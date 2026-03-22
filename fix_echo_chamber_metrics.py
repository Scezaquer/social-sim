import json
import os
import numpy as np
import networkx as nx
from collections import Counter
from tqdm import tqdm

def normalized_entropy(counts: Counter) -> float:
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

def compute_echo_chamber_metrics_for_survey(
    survey: dict,
    nodes: list,
    edges: list,
    threads: list
) -> dict:
    if not isinstance(survey, dict):
        return {"status": "insufficient_data", "reason": "Invalid survey snapshot."}

    survey_step = int(survey.get("step", -1))
    survey_results = survey.get("results", {})
    if not survey_results:
        return {
            "status": "insufficient_data",
            "reason": "Survey has no user responses.",
            "survey_step": survey_step,
        }

    user_names = {node["name"] for node in nodes if node.get("type") == "User"}
    name_to_option = {name: option for name, option in survey_results.items() if name in user_names}

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
    
    # We need to map node index to name
    idx_to_name = {node["id"]: node["name"] for node in nodes}

    for edge in edges:
        source_idx = edge.get("source")
        target_idx = edge.get("target")
        if source_idx is None or target_idx is None:
            continue

        source_name = idx_to_name.get(source_idx)
        target_name = idx_to_name.get(target_idx)
        
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
        int(thread["id"]): sorted(thread["messages"], key=lambda m: int(m.get("step", -1)))
        for thread in threads
    }

    exposure_same_option_shares = []
    exposure_diversities = []
    
    # observations in visualizer data
    observations = [obs for obs in observations_all if int(obs.get("step", -1)) <= survey_step]
    
    # Actually, we should iterate over observations and filter by step
    # Wait, observations_all is not defined yet. Let's pass it.

def compute_echo_chamber_metrics(data: dict) -> dict:
    survey_results = data.get("survey_results", [])
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    threads = data.get("threads", [])
    observations = data.get("observations", [])

    if not survey_results:
        return {
            "status": "insufficient_data",
            "reason": "No survey results available.",
            "survey_count": 0,
            "by_survey": [],
        }

    idx_to_name = {node["id"]: node["name"] for node in nodes}
    user_names = {node["name"] for node in nodes if node.get("type") == "User"}
    
    thread_messages = {
        int(thread["id"]): sorted(thread["messages"], key=lambda m: int(m.get("step", -1)))
        for thread in threads
    }

    per_survey = []
    for survey in survey_results:
        survey_step = int(survey.get("step", -1))
        results = survey.get("results", {})
        name_to_option = {name: option for name, option in results.items() if name in user_names}
        
        if len(name_to_option) < 2:
            per_survey.append({
                "status": "insufficient_data",
                "reason": "Need at least two surveyed users.",
                "survey_step": survey_step,
            })
            continue

        options = sorted({option for option in name_to_option.values()})
        option_to_id = {option: idx for idx, option in enumerate(options)}

        graph = nx.DiGraph()
        for name, option in name_to_option.items():
            graph.add_node(name, opinion=option_to_id[option])

        edge_count_considered = 0
        cross_cutting_edges = 0
        for edge in edges:
            s_name = idx_to_name.get(edge["source"])
            t_name = idx_to_name.get(edge["target"])
            if s_name in name_to_option and t_name in name_to_option:
                graph.add_edge(s_name, t_name)
                edge_count_considered += 1
                if name_to_option[s_name] != name_to_option[t_name]:
                    cross_cutting_edges += 1

        if graph.number_of_edges() == 0:
            per_survey.append({
                "status": "insufficient_data",
                "reason": "No user-user edges with survey labels.",
                "survey_step": survey_step,
            })
            continue

        try:
            assortativity = nx.attribute_assortativity_coefficient(graph, "opinion")
            if np.isnan(assortativity):
                assortativity = 0.0
        except:
            assortativity = 0.0

        local_agreements = []
        for user_name in graph.nodes:
            neighbors = list(graph.successors(user_name))
            if neighbors:
                same = sum(1 for n in neighbors if name_to_option[n] == name_to_option[user_name])
                local_agreements.append(same / len(neighbors))

        exposure_same_option_shares = []
        exposure_diversities = []
        
        current_observations = [obs for obs in observations if int(obs.get("step", -1)) <= survey_step]
        
        for obs in current_observations:
            observer = obs.get("entity_name")
            thread_id = int(obs.get("thread_id", -1))
            obs_step = int(obs.get("step", -1))
            
            if observer not in name_to_option or thread_id not in thread_messages:
                continue

            option_counts = Counter()
            for msg in thread_messages[thread_id]:
                if int(msg.get("step", -1)) < obs_step:
                    author = msg.get("role")
                    if author in name_to_option:
                        option_counts[name_to_option[author]] += 1
            
            total_exposed = sum(option_counts.values())
            if total_exposed > 0:
                observer_option = name_to_option[observer]
                same_option_count = option_counts.get(observer_option, 0)
                exposure_same_option_shares.append(same_option_count / total_exposed)
                exposure_diversities.append(normalized_entropy(option_counts))

        per_survey.append({
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
        })

    latest = per_survey[-1]
    return {
        **latest,
        "survey_count": len(survey_results),
        "by_survey": per_survey,
    }

def main():
    files = [f for f in os.listdir('.') if f.startswith('visualizer_randomized_') and f.endswith('.json')]
    for filename in tqdm(files):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding {filename}")
                continue
        
        if "behavioral_metrics" not in data:
            print(f"Skipping {filename}: no behavioral_metrics")
            continue
            
        metrics = data["behavioral_metrics"]
        if "echo_chamber_metrics" not in metrics:
            print(f"Skipping {filename}: no echo_chamber_metrics")
            continue
            
        echo_metrics = metrics["echo_chamber_metrics"]
        # If it only has one step and it's 2500, or doesn't have 'by_survey'
        if "by_survey" not in echo_metrics or len(echo_metrics.get("by_survey", [])) <= 1:
            print(f"Updating {filename}...")
            new_echo_metrics = compute_echo_chamber_metrics(data)
            data["behavioral_metrics"]["echo_chamber_metrics"] = new_echo_metrics
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
