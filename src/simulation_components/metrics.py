from collections import Counter
import re
from typing import Any

import networkx as nx
import numpy as np

from simulation_components.entities import Entity, User
from simulation_components.type_aliases import Thread


def normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.array([count / total for count in counts.values() if count > 0], dtype=float)
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(probs.size))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def compute_model_opinion_scores(
    model_probabilities: list[dict[str, float]] | dict[str, dict[str, float]] | None,
) -> dict[int, float]:
    if not model_probabilities:
        return {}

    scores: dict[int, float] = {}
    if isinstance(model_probabilities, dict):
        iterable = model_probabilities.items()
    else:
        iterable = enumerate(model_probabilities)

    for model_key, distribution in iterable:
        if not isinstance(distribution, dict):
            continue

        if isinstance(model_key, int):
            model_id = model_key
        else:
            match = re.search(r"(\d+)$", str(model_key))
            if match is None:
                continue
            model_id = int(match.group(1))

        probs = np.array(list(distribution.values()), dtype=float)
        total = probs.sum()
        if total <= 0:
            scores[model_id] = 0.0
            continue

        probs = probs / total
        support = np.arange(len(probs), dtype=float)
        scores[model_id] = float(np.dot(support, probs))

    return scores


def compute_echo_chamber_metrics(
    survey_results: list[dict[str, Any]],
    visualizer_data: dict[str, Any],
    threads: list[Thread],
) -> dict[str, Any]:
    if not survey_results:
        return {
            "status": "insufficient_data",
            "reason": "No survey results available.",
            "survey_count": 0,
            "by_survey": [],
        }

    per_survey = [
        compute_echo_chamber_metrics_for_survey(survey, visualizer_data, threads)
        for survey in survey_results
    ]

    latest = per_survey[-1]
    return {
        **latest,
        "survey_count": len(survey_results),
        "by_survey": per_survey,
    }


def compute_echo_chamber_metrics_for_survey(
    survey: dict[str, Any],
    visualizer_data: dict[str, Any],
    threads: list[Thread],
) -> dict[str, Any]:
    if not isinstance(survey, dict):
        return {
            "status": "insufficient_data",
            "reason": "Invalid survey snapshot.",
        }

    survey_step = int(survey.get("step", -1))
    survey_answers = survey.get("results", {})
    if not survey_answers:
        return {
            "status": "insufficient_data",
            "reason": "Survey has no user responses.",
            "survey_step": survey_step,
        }

    nodes = visualizer_data.get("nodes", [])
    user_names = {node["name"] for node in nodes if node.get("type") == "User"}
    name_to_option = {
        name: option
        for name, option in survey_answers.items()
        if name in user_names
    }

    if len(name_to_option) < 2:
        return {
            "status": "insufficient_data",
            "reason": "Need at least two surveyed users.",
            "survey_step": survey_step,
        }

    options = sorted({option for option in name_to_option.values()})
    option_to_id = {option: idx for idx, option in enumerate(options)}

    graph = nx.DiGraph()
    for name, option in name_to_option.items():
        graph.add_node(name, opinion=option_to_id[option])

    edge_count_considered = 0
    cross_cutting_edges = 0
    edges = visualizer_data.get("edges", [])
    for edge in edges:
        source_idx = edge.get("source")
        target_idx = edge.get("target")
        if source_idx is None or target_idx is None:
            continue

        if source_idx < 0 or target_idx < 0:
            continue

        if source_idx >= len(nodes) or target_idx >= len(nodes):
            continue

        source_name = nodes[source_idx]["name"]
        target_name = nodes[target_idx]["name"]
        if source_name not in name_to_option or target_name not in name_to_option:
            continue

        graph.add_edge(source_name, target_name)
        edge_count_considered += 1
        if name_to_option[source_name] != name_to_option[target_name]:
            cross_cutting_edges += 1

    if graph.number_of_edges() == 0:
        return {
            "status": "insufficient_data",
            "reason": "No user-user edges with survey labels.",
            "survey_step": survey_step,
        }

    try:
        assortativity = nx.attribute_assortativity_coefficient(graph, "opinion")
        if np.isnan(assortativity):
            assortativity = 0.0
    except Exception:
        assortativity = 0.0

    local_agreements = []
    for user_name in graph.nodes:
        neighbors = list(graph.successors(user_name))
        if not neighbors:
            continue
        same = sum(
            1
            for neighbor in neighbors
            if name_to_option[neighbor] == name_to_option[user_name]
        )
        local_agreements.append(float(same / len(neighbors)))

    thread_messages = {
        int(thread.id): sorted(thread.content, key=lambda m: int(m.get("step", -1)))
        for thread in threads
    }

    exposure_same_option_shares = []
    exposure_diversities = []
    for obs in visualizer_data.get("observations", []):
        observer = obs.get("entity_name")
        thread_id = int(obs.get("thread_id", -1))
        obs_step = int(obs.get("step", -1))
        if observer not in name_to_option or thread_id not in thread_messages:
            continue
        if obs_step > survey_step:
            continue

        option_counts: Counter[str] = Counter()
        for message in thread_messages[thread_id]:
            msg_step = int(message.get("step", -1))
            if msg_step < 0 or msg_step >= obs_step:
                break
            author = message.get("role")
            if author in name_to_option:
                option_counts[name_to_option[author]] += 1

        total_exposed = sum(option_counts.values())
        if total_exposed == 0:
            continue

        observer_option = name_to_option[observer]
        same_option_count = option_counts.get(observer_option, 0)
        exposure_same_option_shares.append(float(same_option_count / total_exposed))
        exposure_diversities.append(normalized_entropy(option_counts))

    return {
        "status": "ok",
        "survey_step": survey_step,
        "labeled_user_count": len(name_to_option),
        "labeled_edge_count": edge_count_considered,
        "network_assortativity": float(assortativity),
        "mean_local_agreement": float(np.mean(local_agreements)) if local_agreements else 0.0,
        "cross_cutting_edge_fraction": float(cross_cutting_edges / edge_count_considered) if edge_count_considered else 0.0,
        "mean_same_option_exposure_share": float(np.mean(exposure_same_option_shares)) if exposure_same_option_shares else 0.0,
        "mean_exposure_diversity": float(np.mean(exposure_diversities)) if exposure_diversities else 0.0,
        "option_distribution": dict(Counter(name_to_option.values())),
    }


def compute_herd_effect_metrics(
    survey_results: list[dict[str, Any]],
    visualizer_data: dict[str, Any],
    social_graph: list[set[int]] | None,
    name_to_idx: dict[str, int] | None,
) -> dict[str, Any]:
    surveys = survey_results
    if len(surveys) < 2:
        return {
            "status": "insufficient_data",
            "reason": "Need at least two survey snapshots.",
            "survey_count": len(surveys),
        }

    transitions = []
    herd_follow_rates = []
    shift_rates = []
    consensus_gains = []
    neighbor_alignment_shift_rates = []

    node_names = [node.get("name") for node in visualizer_data.get("nodes", [])]

    def neighbor_majority_option(
        user_name: str,
        prev_results: dict[str, str],
        shared_users: set[str],
    ) -> str | None:
        if not social_graph or name_to_idx is None:
            return None

        user_idx = name_to_idx.get(user_name)
        if user_idx is None or user_idx < 0 or user_idx >= len(social_graph):
            return None

        neighbor_options: list[str] = []
        for neighbor_idx in social_graph[user_idx]:
            if neighbor_idx < 0 or neighbor_idx >= len(node_names):
                continue
            neighbor_name = node_names[neighbor_idx]
            if neighbor_name is None or neighbor_name not in shared_users:
                continue
            option = prev_results.get(neighbor_name)
            if option is not None:
                neighbor_options.append(option)

        if not neighbor_options:
            return None

        option_counts = Counter(neighbor_options)
        ranked = option_counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    for prev, curr in zip(surveys[:-1], surveys[1:]):
        prev_results = prev.get("results", {})
        curr_results = curr.get("results", {})
        shared_users = set(prev_results).intersection(curr_results)
        if not shared_users:
            continue

        changed_users = [
            name for name in shared_users if prev_results[name] != curr_results[name]
        ]

        prev_counts = Counter(prev_results[name] for name in shared_users)
        curr_counts = Counter(curr_results[name] for name in shared_users)
        prev_majority_option, prev_majority_count = max(
            prev_counts.items(), key=lambda item: item[1]
        )
        curr_majority_option, curr_majority_count = max(
            curr_counts.items(), key=lambda item: item[1]
        )

        moved_to_curr_majority = sum(
            1 for name in changed_users if curr_results[name] == curr_majority_option
        )
        moved_to_prev_majority = sum(
            1 for name in changed_users if curr_results[name] == prev_majority_option
        )

        neighbor_majority_eligible_users = 0
        changed_to_neighbor_majority = 0
        for name in shared_users:
            neighbor_majority = neighbor_majority_option(name, prev_results, shared_users)
            if neighbor_majority is None:
                continue
            neighbor_majority_eligible_users += 1
            if prev_results[name] != curr_results[name] and curr_results[name] == neighbor_majority:
                changed_to_neighbor_majority += 1

        n_shared = len(shared_users)
        n_changed = len(changed_users)
        shift_rate = float(n_changed / n_shared)
        herd_follow_rate = float(moved_to_curr_majority / n_changed) if n_changed else 0.0
        neighbor_alignment_shift_rate = float(changed_to_neighbor_majority / n_shared)
        prev_consensus = float(prev_majority_count / n_shared)
        curr_consensus = float(curr_majority_count / n_shared)
        consensus_gain = curr_consensus - prev_consensus

        transitions.append(
            {
                "from_step": int(prev.get("step", -1)),
                "to_step": int(curr.get("step", -1)),
                "shared_users": n_shared,
                "changed_users": n_changed,
                "opinion_shift_rate": shift_rate,
                "current_majority_follow_rate": herd_follow_rate,
                "moved_to_previous_majority_rate": float(moved_to_prev_majority / n_changed)
                if n_changed
                else 0.0,
                "neighbor_majority_eligible_user_count": neighbor_majority_eligible_users,
                "changed_to_neighbor_majority_count": changed_to_neighbor_majority,
                "neighbor_alignment_shift_rate": neighbor_alignment_shift_rate,
                "neighbor_alignment_among_changers_rate": float(changed_to_neighbor_majority / n_changed)
                if n_changed
                else 0.0,
                "neighbor_alignment_eligible_shift_rate": float(
                    changed_to_neighbor_majority / neighbor_majority_eligible_users
                )
                if neighbor_majority_eligible_users
                else 0.0,
                "previous_majority_option": prev_majority_option,
                "current_majority_option": curr_majority_option,
                "previous_consensus": prev_consensus,
                "current_consensus": curr_consensus,
                "consensus_gain": consensus_gain,
                "previous_diversity": normalized_entropy(prev_counts),
                "current_diversity": normalized_entropy(curr_counts),
            }
        )

        herd_follow_rates.append(herd_follow_rate)
        shift_rates.append(shift_rate)
        consensus_gains.append(consensus_gain)
        neighbor_alignment_shift_rates.append(neighbor_alignment_shift_rate)

    if not transitions:
        return {
            "status": "insufficient_data",
            "reason": "No overlapping users between consecutive surveys.",
            "survey_count": len(surveys),
        }

    first_counts = Counter(surveys[0].get("results", {}).values())
    last_counts = Counter(surveys[-1].get("results", {}).values())
    first_consensus = (
        max(first_counts.values()) / max(sum(first_counts.values()), 1)
        if first_counts
        else 0.0
    )
    last_consensus = (
        max(last_counts.values()) / max(sum(last_counts.values()), 1)
        if last_counts
        else 0.0
    )

    return {
        "status": "ok",
        "survey_count": len(surveys),
        "transition_count": len(transitions),
        "mean_opinion_shift_rate": float(np.mean(shift_rates)) if shift_rates else 0.0,
        "mean_current_majority_follow_rate": float(np.mean(herd_follow_rates))
        if herd_follow_rates
        else 0.0,
        "mean_consensus_gain": float(np.mean(consensus_gains)) if consensus_gains else 0.0,
        "mean_neighbor_alignment_shift_rate": float(np.mean(neighbor_alignment_shift_rates))
        if neighbor_alignment_shift_rates
        else 0.0,
        "initial_consensus": float(first_consensus),
        "final_consensus": float(last_consensus),
        "net_consensus_change": float(last_consensus - first_consensus),
        "initial_diversity": normalized_entropy(first_counts),
        "final_diversity": normalized_entropy(last_counts),
        "transitions": transitions,
    }


def compute_behavioral_metrics(
    survey_results: list[dict[str, Any]],
    visualizer_data: dict[str, Any],
    threads: list[Thread],
    social_graph: list[set[int]] | None,
    name_to_idx: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "echo_chamber_metrics": compute_echo_chamber_metrics(
            survey_results, visualizer_data, threads
        ),
        "herd_effect_metrics": compute_herd_effect_metrics(
            survey_results, visualizer_data, social_graph, name_to_idx
        ),
    }


def compute_homophily_metrics(
    entities: list[Entity] | tuple[Entity, ...],
    social_graph: list[set[int]] | None,
    model_probabilities: list[dict[str, float]] | dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    model_scores = compute_model_opinion_scores(model_probabilities)

    user_indices = [i for i, entity in enumerate(entities) if isinstance(entity, User)]
    if not user_indices:
        return {
            "edge_similarity": 0.0,
            "random_similarity": 0.0,
            "excess_similarity": 0.0,
            "same_model_edge_fraction": 0.0,
            "edge_count": 0,
        }

    if social_graph is None:
        return {
            "edge_similarity": 0.0,
            "random_similarity": 0.0,
            "excess_similarity": 0.0,
            "same_model_edge_fraction": 0.0,
            "edge_count": 0,
        }

    user_set = set(user_indices)
    user_scores = {
        idx: model_scores.get(getattr(entities[idx], "model_id", None), 0.0)
        for idx in user_indices
    }

    min_score = min(user_scores.values()) if user_scores else 0.0
    max_score = max(user_scores.values()) if user_scores else 0.0
    score_range = max(max_score - min_score, 1e-12)

    edge_similarities = []
    same_model_count = 0
    edge_count = 0

    for u_idx in user_indices:
        u_model_id = getattr(entities[u_idx], "model_id", None)
        u_score = user_scores[u_idx]
        for v_idx in social_graph[u_idx]:
            if v_idx not in user_set:
                continue

            v_model_id = getattr(entities[v_idx], "model_id", None)
            v_score = user_scores[v_idx]

            similarity = 1.0 - (abs(u_score - v_score) / score_range)
            edge_similarities.append(float(np.clip(similarity, 0.0, 1.0)))

            if u_model_id == v_model_id:
                same_model_count += 1
            edge_count += 1

    if edge_count == 0:
        return {
            "edge_similarity": 0.0,
            "random_similarity": 0.0,
            "excess_similarity": 0.0,
            "same_model_edge_fraction": 0.0,
            "edge_count": 0,
        }

    edge_similarity = float(np.mean(edge_similarities))

    rng = np.random.default_rng(123)
    sampled_pair_count = min(5000, max(1000, edge_count))
    random_similarities = []
    user_scores_list = [user_scores[idx] for idx in user_indices]
    n_users = len(user_scores_list)

    if n_users > 1:
        for _ in range(sampled_pair_count):
            i = int(rng.integers(0, n_users))
            j = int(rng.integers(0, n_users - 1))
            if j >= i:
                j += 1
            similarity = 1.0 - (abs(user_scores_list[i] - user_scores_list[j]) / score_range)
            random_similarities.append(float(np.clip(similarity, 0.0, 1.0)))

    random_similarity = float(np.mean(random_similarities)) if random_similarities else 0.0

    return {
        "edge_similarity": edge_similarity,
        "random_similarity": random_similarity,
        "excess_similarity": edge_similarity - random_similarity,
        "same_model_edge_fraction": float(same_model_count / edge_count),
        "edge_count": edge_count,
    }
