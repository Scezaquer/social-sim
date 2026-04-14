from collections.abc import Callable, Mapping, Sequence
from typing import Any
from simulation_components.type_aliases import Thread
from simulation_components.entities import NewsSource, User, Entity
import numpy as np
from tqdm import tqdm
import networkx as nx
import time
import re
from collections import Counter

class SimEngine:
    """Engine interface."""

    def __init__(self,
                 threads: list[Thread] = None,
                 graph: Any = None,
                 survey_config: dict[str, Any] = None,
                 homophily: bool = False,
                 model_probabilities: list[dict[str, float]] | dict[str, dict[str, float]] | None = None,
                 ):
        self._threads = [] if threads is None else threads
        self._entity_activity_probs = None
        self._social_graph = None
        self._acting_entities = None
        self._acting_probs = None
        self._name_to_idx = None
        self._initial_graph = graph
        self._survey_config = survey_config
        self._step_count = 0
        self._survey_results = []
        self._homophily = homophily
        self._model_probabilities = model_probabilities
        self._homophily_metrics = None
        self._behavioral_metrics = None
        self._visualizer_data = {
            "nodes": [],
            "edges": [],
            "threads": [],
            "observations": [],
            "survey_results": [],
            "news_posts": [],
            "behavioral_metrics": {},
        }

    @staticmethod
    def _normalized_entropy(counts: Counter[str]) -> float:
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

    def _compute_echo_chamber_metrics(self) -> dict[str, Any]:
        if not self._survey_results:
            return {
                "status": "insufficient_data",
                "reason": "No survey results available.",
                "survey_count": 0,
                "by_survey": [],
            }

        per_survey = [
            self._compute_echo_chamber_metrics_for_survey(survey)
            for survey in self._survey_results
        ]

        latest = per_survey[-1]
        return {
            **latest,
            "survey_count": len(self._survey_results),
            "by_survey": per_survey,
        }

    def _compute_echo_chamber_metrics_for_survey(
        self,
        survey: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(survey, dict):
            return {
                "status": "insufficient_data",
                "reason": "Invalid survey snapshot.",
            }

        survey_step = int(survey.get("step", -1))
        survey_results = survey.get("results", {})
        if not survey_results:
            return {
                "status": "insufficient_data",
                "reason": "Survey has no user responses.",
                "survey_step": survey_step,
            }

        user_names = {
            node["name"]
            for node in self._visualizer_data.get("nodes", [])
            if node.get("type") == "User"
        }
        name_to_option = {
            name: option
            for name, option in survey_results.items()
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
        for edge in self._visualizer_data.get("edges", []):
            source_idx = edge.get("source")
            target_idx = edge.get("target")
            if source_idx is None or target_idx is None:
                continue

            if source_idx < 0 or target_idx < 0:
                continue

            if source_idx >= len(self._visualizer_data["nodes"]) or target_idx >= len(self._visualizer_data["nodes"]):
                continue

            source_name = self._visualizer_data["nodes"][source_idx]["name"]
            target_name = self._visualizer_data["nodes"][target_idx]["name"]
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
            same = sum(1 for neighbor in neighbors if name_to_option[neighbor] == name_to_option[user_name])
            local_agreements.append(float(same / len(neighbors)))

        thread_messages = {
            int(thread.id): sorted(thread.content, key=lambda m: int(m.get("step", -1)))
            for thread in self._threads
        }

        exposure_same_option_shares = []
        exposure_diversities = []
        for obs in self._visualizer_data.get("observations", []):
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
            exposure_diversities.append(self._normalized_entropy(option_counts))

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

    def _compute_herd_effect_metrics(self) -> dict[str, Any]:
        surveys = self._survey_results
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

        node_names = [node.get("name") for node in self._visualizer_data.get("nodes", [])]

        def _neighbor_majority_option(
            user_name: str,
            prev_results: dict[str, str],
            shared_users: set[str],
        ) -> str | None:
            if not self._social_graph or self._name_to_idx is None:
                return None

            user_idx = self._name_to_idx.get(user_name)
            if user_idx is None or user_idx < 0 or user_idx >= len(self._social_graph):
                return None

            neighbor_options: list[str] = []
            for neighbor_idx in self._social_graph[user_idx]:
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
                name for name in shared_users
                if prev_results[name] != curr_results[name]
            ]

            prev_counts = Counter(prev_results[name] for name in shared_users)
            curr_counts = Counter(curr_results[name] for name in shared_users)
            prev_majority_option, prev_majority_count = max(prev_counts.items(), key=lambda item: item[1])
            curr_majority_option, curr_majority_count = max(curr_counts.items(), key=lambda item: item[1])

            moved_to_curr_majority = sum(
                1 for name in changed_users if curr_results[name] == curr_majority_option
            )
            moved_to_prev_majority = sum(
                1 for name in changed_users if curr_results[name] == prev_majority_option
            )

            neighbor_majority_eligible_users = 0
            changed_to_neighbor_majority = 0
            for name in shared_users:
                neighbor_majority = _neighbor_majority_option(name, prev_results, shared_users)
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

            transitions.append({
                "from_step": int(prev.get("step", -1)),
                "to_step": int(curr.get("step", -1)),
                "shared_users": n_shared,
                "changed_users": n_changed,
                "opinion_shift_rate": shift_rate,
                "current_majority_follow_rate": herd_follow_rate,
                "moved_to_previous_majority_rate": float(moved_to_prev_majority / n_changed) if n_changed else 0.0,
                "neighbor_majority_eligible_user_count": neighbor_majority_eligible_users,
                "changed_to_neighbor_majority_count": changed_to_neighbor_majority,
                "neighbor_alignment_shift_rate": neighbor_alignment_shift_rate,
                "neighbor_alignment_among_changers_rate": float(changed_to_neighbor_majority / n_changed) if n_changed else 0.0,
                "neighbor_alignment_eligible_shift_rate": float(changed_to_neighbor_majority / neighbor_majority_eligible_users) if neighbor_majority_eligible_users else 0.0,
                "previous_majority_option": prev_majority_option,
                "current_majority_option": curr_majority_option,
                "previous_consensus": prev_consensus,
                "current_consensus": curr_consensus,
                "consensus_gain": consensus_gain,
                "previous_diversity": self._normalized_entropy(prev_counts),
                "current_diversity": self._normalized_entropy(curr_counts),
            })

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
            if first_counts else 0.0
        )
        last_consensus = (
            max(last_counts.values()) / max(sum(last_counts.values()), 1)
            if last_counts else 0.0
        )

        return {
            "status": "ok",
            "survey_count": len(surveys),
            "transition_count": len(transitions),
            "mean_opinion_shift_rate": float(np.mean(shift_rates)) if shift_rates else 0.0,
            "mean_current_majority_follow_rate": float(np.mean(herd_follow_rates)) if herd_follow_rates else 0.0,
            "mean_consensus_gain": float(np.mean(consensus_gains)) if consensus_gains else 0.0,
            "mean_neighbor_alignment_shift_rate": float(np.mean(neighbor_alignment_shift_rates)) if neighbor_alignment_shift_rates else 0.0,
            "initial_consensus": float(first_consensus),
            "final_consensus": float(last_consensus),
            "net_consensus_change": float(last_consensus - first_consensus),
            "initial_diversity": self._normalized_entropy(first_counts),
            "final_diversity": self._normalized_entropy(last_counts),
            "transitions": transitions,
        }

    def _compute_behavioral_metrics(self) -> dict[str, Any]:
        return {
            "echo_chamber_metrics": self._compute_echo_chamber_metrics(),
            "herd_effect_metrics": self._compute_herd_effect_metrics(),
        }

    def get_behavioral_metrics(self) -> dict[str, Any]:
        """Returns computed echo-chamber and herd-effect metrics."""
        self._behavioral_metrics = self._compute_behavioral_metrics()
        return self._behavioral_metrics

    def _compute_model_opinion_scores(self) -> dict[int, float]:
        if not self._model_probabilities:
            return {}

        scores = {}
        if isinstance(self._model_probabilities, dict):
            iterable = self._model_probabilities.items()
        else:
            iterable = enumerate(self._model_probabilities)

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

    def _assign_with_homophily(
        self,
        graph: Any,
        remaining_nodes: list[int],
        remaining_indices: list[int],
        entities: Sequence[Entity],
    ) -> dict[int, int]:
        if not remaining_nodes or not remaining_indices:
            return {}

        if graph.is_directed():
            undirected_graph = graph.to_undirected(as_view=False)
        else:
            undirected_graph = graph

        graph_subset = undirected_graph.subgraph(remaining_nodes).copy()
        positions = nx.spring_layout(graph_subset, seed=42, dim=2)
        ordered_nodes = sorted(
            remaining_nodes,
            key=lambda node: float(np.ravel(np.asarray(positions.get(node, [0.0])))[0])
        )

        model_scores = self._compute_model_opinion_scores()
        model_to_entity_indices: dict[int | None, list[int]] = {}
        for entity_idx in remaining_indices:
            model_id = getattr(entities[entity_idx], 'model_id', None)
            model_to_entity_indices.setdefault(model_id, []).append(entity_idx)

        rng = np.random.default_rng(42)
        for indices in model_to_entity_indices.values():
            rng.shuffle(indices)

        ordered_model_ids = sorted(
            model_to_entity_indices.keys(),
            key=lambda model_id: (model_scores.get(model_id, float('inf')), str(model_id))
        )

        ordered_entity_indices = []
        for model_id in ordered_model_ids:
            ordered_entity_indices.extend(model_to_entity_indices[model_id])

        node_to_entity_idx = {}
        for node, entity_idx in zip(ordered_nodes, ordered_entity_indices):
            node_to_entity_idx[node] = entity_idx

        return node_to_entity_idx

    def _compute_homophily_metrics(self, entities: Sequence[Entity]) -> dict[str, float]:
        model_scores = self._compute_model_opinion_scores()

        user_indices = [i for i, entity in enumerate(entities) if isinstance(entity, User)]
        if not user_indices:
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
            for v_idx in self._social_graph[u_idx]:
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

    def get_survey_results(self) -> list[dict[str, Any]]:
        """Returns the results of the surveys."""
        return self._survey_results

    def get_visualizer_data(self) -> dict[str, Any]:
        """Returns the data needed for the visualizer."""
        self._visualizer_data["threads"] = [
            {"id": t.id, "messages": t.content} for t in self._threads
        ]
        self._behavioral_metrics = self._compute_behavioral_metrics()
        self._visualizer_data["behavioral_metrics"] = self._behavioral_metrics
        return self._visualizer_data

    def _initialize_social_context(self, entities: Sequence[Entity]):
        n_entities = len(entities)
        entity_names = [e.name for e in entities]
        self._name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
        # Identify NewsSource
        news_source_indices = [i for i, e in enumerate(entities) if isinstance(e, NewsSource)]

        # Power-law activity distribution
        ranks = np.arange(1, n_entities + 1)
        activity_weights = 1.0 / (ranks ** 0.5) # Zipf's law
        activity_weights /= activity_weights.sum()
        
        # Assign highest weights to NewsSource
        activity_weights = np.sort(activity_weights)[::-1]
        final_weights = np.zeros(n_entities)
        
        for i, idx in enumerate(news_source_indices):
            if i < len(activity_weights):
                final_weights[idx] = activity_weights[i]
        
        remaining_indices = [i for i in range(n_entities) if i not in news_source_indices]
        remaining_weights = activity_weights[len(news_source_indices):]
        np.random.shuffle(remaining_weights)
        
        for i, idx in enumerate(remaining_indices):
            if i < len(remaining_weights):
                final_weights[idx] = remaining_weights[i]

        self._entity_activity_probs = dict(zip(entity_names, final_weights))
        
        # Cache for next_acting efficiency
        self._acting_entities = list(entities)
        self._acting_probs = final_weights
        
        # Social Graph using NetworkX
        self._social_graph = [set() for _ in range(n_entities)]

        if self._initial_graph is not None:
            G = self._initial_graph
            
            # Calculate degrees to identify hubs
            if G.is_directed():
                degree_dict = dict(G.in_degree())
            else:
                degree_dict = dict(G.degree())
                
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            
            node_to_entity_idx = {}
            used_nodes = set()
            
            # Assign NewsSources to highest degree nodes
            for i, ns_idx in enumerate(news_source_indices):
                if i < len(sorted_nodes):
                    node = sorted_nodes[i]
                    node_to_entity_idx[node] = ns_idx
                    used_nodes.add(node)
            
            # Assign remaining entities
            remaining_indices = [i for i in range(n_entities) if i not in news_source_indices]
            remaining_nodes = [n for n in sorted_nodes if n not in used_nodes]

            if self._homophily:
                homophily_assignment = self._assign_with_homophily(
                    graph=G,
                    remaining_nodes=remaining_nodes,
                    remaining_indices=remaining_indices,
                    entities=entities,
                )
                node_to_entity_idx.update(homophily_assignment)
            else:
                # Shuffle remaining indices to avoid correlation
                np.random.shuffle(remaining_indices)

                for i, r_idx in enumerate(remaining_indices):
                    if i < len(remaining_nodes):
                        node = remaining_nodes[i]
                        node_to_entity_idx[node] = r_idx
            
            # Build social graph from G edges
            for u, v in G.edges():
                if u in node_to_entity_idx and v in node_to_entity_idx:
                    u_idx = node_to_entity_idx[u]
                    v_idx = node_to_entity_idx[v]
                    if u_idx != v_idx:
                        self._social_graph[u_idx].add(v_idx)
        else:
            # Scale-free graph for realistic social media topology (hubs/poles)
            G = nx.scale_free_graph(n_entities, seed=42)
            
            for u, v in G.edges():
                if u != v and u < n_entities and v < n_entities:
                    self._social_graph[u].add(v)
            
            # Force many users to follow NewsSource
            for ns_idx in news_source_indices:
                for i in range(n_entities):
                    if i != ns_idx and np.random.rand() < 0.8:
                        self._social_graph[i].add(ns_idx)
            
            # Clear G to free memory
            del G

        # Populate visualizer data for nodes and edges
        for i, entity in enumerate(entities):
            self._visualizer_data["nodes"].append({
                "id": i,
                "name": entity.name,
                "type": "NewsSource" if isinstance(entity, NewsSource) else "User",
                "model_id": getattr(entity, "model_id", None)
            })
        for u_idx, neighbors in enumerate(self._social_graph):
            for v_idx in neighbors:
                self._visualizer_data["edges"].append({
                    "source": u_idx,
                    "target": v_idx
                })

        self._homophily_metrics = self._compute_homophily_metrics(entities)

    def make_observation(
        self,
        game_master: Entity,
        entity: Entity,
        make_new_thread: bool = True,
        step: int = 0
    ) -> Thread:
        """Make an observation for an entity."""
        
        if make_new_thread and isinstance(entity, NewsSource):
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            self._visualizer_data["observations"].append({
                "step": step,
                "entity_name": entity.name,
                "thread_id": new_thread.id
            })
            return new_thread

        if make_new_thread and np.random.rand() < 0.3:
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            self._visualizer_data["observations"].append({
                "step": step,
                "entity_name": entity.name,
                "thread_id": new_thread.id
            })
            return new_thread

        if not self._threads:
            if make_new_thread:
                new_thread = Thread(id=0, content=[])
                self._threads.append(new_thread)
                self._visualizer_data["observations"].append({
                    "step": step,
                    "entity_name": entity.name,
                    "thread_id": new_thread.id
                })
                return new_thread
            else:
                return None

        # Only consider the last 50 threads for efficiency
        relevant_threads = self._threads[-50:]

        # Calculate weights proportional to thread length
        thread_lengths = np.array([float(len(thread.content) + 2) for thread in relevant_threads])
        # Add 1 to normalize and prevent zero-length threads from having zero probability, and +1 more to ensure even very short threads have some chance of being selected.
        weights = thread_lengths

        # Zero out weights for threads with more than 20 messages
        weights[np.array([len(thread.content) > 20 for thread in relevant_threads])] = 0.0

        # Prevent self-replies
        for i, thread in enumerate(relevant_threads):
            if thread.content and thread.content[-1]['role'] == entity.name:
                weights[i] = 0.0
        
        # Social and Participation boost
        boosts = np.ones(len(relevant_threads))
        
        entity_idx = self._name_to_idx.get(entity.name)
        following = set()
        if entity_idx is not None and self._social_graph:
            following = self._social_graph[entity_idx]
            
        for i, thread in enumerate(relevant_threads):
            participants_indices = set()
            for msg in thread.content:
                p_idx = self._name_to_idx.get(msg['role'])
                if p_idx is not None:
                    participants_indices.add(p_idx)
            
            # Social boost
            if not following.isdisjoint(participants_indices):
                boosts[i] *= 10.0
            
            # Participation boost
            if entity_idx in participants_indices:
                age = len(relevant_threads) - i # Age relative to the most recent thread
                boosts[i] *= (1.0 + (20.0 / age))
        
        weights *= boosts

        # Normalize weights to sum to 1
        weights = weights.clip(min=0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        elif make_new_thread:
            # If no valid threads (e.g. all are self-replies), create a new one
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            self._visualizer_data["observations"].append({
                "step": step,
                "entity_name": entity.name,
                "thread_id": new_thread.id
            })
            return new_thread
        else:
            return None
        
        random_thread_idx = np.random.choice(len(relevant_threads), p=weights)
        random_thread = relevant_threads[random_thread_idx]
        entity.observe(random_thread)
        self._visualizer_data["observations"].append({
            "step": step,
            "entity_name": entity.name,
            "thread_id": random_thread.id
        })
        return random_thread

    def next_acting(
        self,
        game_master: Entity,
        entities: Sequence[Entity],
    ) -> Entity:
        """Return the next entity or entities to act."""
        
        if self._entity_activity_probs is None:
            self._initialize_social_context(entities)

        # Use cached probabilities for efficiency
        for _ in range(100):
            idx = np.random.choice(len(self._acting_entities), p=self._acting_probs)
            acting_entity = self._acting_entities[idx]
            
            if isinstance(acting_entity, NewsSource) and not acting_entity.has_news():
                continue
            
            return acting_entity
            
        # Fallback: pick a non-NewsSource entity if possible
        non_news_indices = [i for i, e in enumerate(self._acting_entities) if not isinstance(e, NewsSource)]
        if non_news_indices:
             probs = self._acting_probs[non_news_indices]
             if probs.sum() > 0:
                 probs /= probs.sum()
                 idx = np.random.choice(len(non_news_indices), p=probs)
                 return self._acting_entities[non_news_indices[idx]]
        
        return self._acting_entities[0]

    def resolve(
        self,
        game_master: Entity,
        event: str,
    ) -> None:
        """Resolve the event."""
        return

    def terminate(
        self,
        game_master: Entity,
    ) -> bool:
        """Decide if the episode should terminate or continue."""
        return False

    def next_game_master(
        self,
        game_master: Entity,
        game_masters: Sequence[Entity],
    ) -> Entity:
        """Return the game master that will be responsible for the next step."""
        return game_master

    def run_loop(
        self,
        game_masters: Sequence[Entity],
        entities: Sequence[Entity],
        premise: str,
        max_steps: int,
        verbose: bool,
        log: list[Mapping[str, Any]] | None,
        checkpoint_callback: Callable[[int], None] | None = None,
        start_time: float | None = None,
        duration: float | None = None,
    ) -> None:
        """Run a game loop."""

        steps = 0
        game_master = game_masters[0]

        if self._entity_activity_probs is None:
            self._initialize_social_context(entities)

        if self._homophily_metrics is not None:
            metrics = self._homophily_metrics
            print(
                "Homophily metrics: "
                f"edge_similarity={metrics['edge_similarity']:.4f}, "
                f"random_similarity={metrics['random_similarity']:.4f}, "
                f"excess_similarity={metrics['excess_similarity']:.4f}, "
                f"same_model_edge_fraction={metrics['same_model_edge_fraction']:.4f}, "
                f"edges={metrics['edge_count']}"
            )

        with tqdm(total=max_steps, disable=not verbose, desc="Game Steps") as pbar:
            while not self.terminate(game_master) and steps <= max_steps:
                if start_time is not None and duration is not None:
                    if time.time() - start_time > duration:
                        if verbose:
                            print(f"Time limit of {duration} seconds reached. Stopping simulation.")
                        break

                if verbose:
                    print(f"Step {steps}")

                # Survey Logic
                if self._survey_config and steps % self._survey_config['interval'] == 0:
                    question = self._survey_config['question']
                    options = self._survey_config['options']
                    results = {}
                    print(f"\n--- Running Survey at Step {steps} ---")
                    for entity in tqdm(entities, desc="Surveying Entities", leave=False):
                        if isinstance(entity, User):
                            response = entity.survey_response(question, options)
                            results[entity.name] = response
                    
                    self._survey_results.append({
                        'step': steps,
                        'question': question,
                        'results': results
                    })
                    self._visualizer_data["survey_results"].append({
                        'step': steps,
                        'question': question,
                        'results': results
                    })
                    print(f"--- Survey Completed ---\n")
                    if steps == max_steps:
                        break

                # 10 observations for every action
                for i in range(10):
                    acting_entity = self.next_acting(
                        game_master, entities)
                    
                    make_new_thread = ( i == 9 )  # Only create new thread on the last observation
                    observation = self.make_observation(game_master, acting_entity, make_new_thread, step=steps)
                action = acting_entity.act()
                observation.content.append({'role': acting_entity.name, 'content': action, 'step': steps})
                
                if isinstance(acting_entity, NewsSource):
                    self._visualizer_data["news_posts"].append({
                        "step": steps,
                        "thread_id": observation.id,
                        "content": action
                    })
                
                self.resolve(game_master, action)
                game_master = self.next_game_master(game_master, game_masters)
                steps += 1
                pbar.update(1)

        return
