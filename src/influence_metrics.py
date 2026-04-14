from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import networkx as nx
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


EMBEDDING_MODEL_NAME = "vinai/bertweet-base"


@dataclass(slots=True)
class MessageRecord:
    id: int
    thread_id: int
    author: str
    author_node_id: int | None
    content: str
    timestamp: int
    position: int
    embedding: np.ndarray | None = None
    candidate_parent_ids: list[int] = field(default_factory=list)
    viewers: set[str] = field(default_factory=set)
    impact_score: float = 0.0
    drift_score: float = 0.0


class BertweetEmbedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 128,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 768))
            return np.zeros((0, hidden_size), dtype=np.float32)

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self.model(**tokens)
                pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(batches)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = masked_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def similarity(left_embedding: np.ndarray, right_embedding: np.ndarray) -> float:
    return float(np.dot(left_embedding, right_embedding))


def clean_message_text(text: str) -> str:
    cleaned = text.replace("<|im_end|>", " ").replace("<URL>", "URL")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def load_simulation_data(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_messages(sim_data: dict[str, Any]) -> list[MessageRecord]:
    nodes_by_name = {node["name"]: node["id"] for node in sim_data.get("nodes", [])}
    messages: list[MessageRecord] = []

    for thread in sim_data.get("threads", []):
        thread_id = int(thread["id"])
        for position, message in enumerate(thread.get("messages", [])):
            timestamp = int(message.get("step", -1))
            if timestamp < 0:
                continue
            author = message.get("role", "")
            content = clean_message_text(message.get("content", ""))
            messages.append(
                MessageRecord(
                    id=len(messages),
                    thread_id=thread_id,
                    author=author,
                    author_node_id=nodes_by_name.get(author),
                    content=content,
                    timestamp=timestamp,
                    position=position,
                )
            )

    messages.sort(key=lambda msg: (msg.timestamp, msg.thread_id, msg.position, msg.id))
    for new_id, message in enumerate(messages):
        message.id = new_id
    return messages


def compute_embeddings(
    messages: list[MessageRecord],
    embed: Callable[[list[str]], np.ndarray] | None = None,
    *,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    if not messages:
        return np.zeros((0, 0), dtype=np.float32)

    embed_fn = embed or BertweetEmbedder(model_name=model_name, batch_size=batch_size, device=device)
    texts = [message.content for message in messages]
    embeddings = embed_fn(texts)
    if embeddings.shape[0] != len(messages):
        raise ValueError("Embedding function returned an unexpected number of vectors.")

    for message, embedding in zip(messages, embeddings):
        message.embedding = embedding.astype(np.float32)
    return embeddings.astype(np.float32)


def build_exposure_index(
    sim_data: dict[str, Any],
    messages: list[MessageRecord],
) -> tuple[dict[str, dict[int, int]], dict[int, set[str]]]:
    messages_by_thread: dict[int, list[MessageRecord]] = defaultdict(list)
    posts_by_author: dict[str, list[MessageRecord]] = defaultdict(list)
    observations_by_author: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for message in messages:
        messages_by_thread[message.thread_id].append(message)
        posts_by_author[message.author].append(message)

    for thread_messages in messages_by_thread.values():
        thread_messages.sort(key=lambda msg: (msg.timestamp, msg.position, msg.id))

    for author_messages in posts_by_author.values():
        author_messages.sort(key=lambda msg: (msg.timestamp, msg.position, msg.id))

    for observation in sim_data.get("observations", []):
        entity_name = observation.get("entity_name")
        if entity_name is None:
            continue
        observations_by_author[entity_name].append(
            (int(observation["step"]), int(observation["thread_id"]))
        )

    for author_observations in observations_by_author.values():
        author_observations.sort(key=lambda item: (item[0], item[1]))

    exposure_steps_by_user: dict[str, dict[int, int]] = defaultdict(dict)
    viewers_by_message: dict[int, set[str]] = defaultdict(set)
    all_authors = set(posts_by_author) | set(observations_by_author)

    for author in all_authors:
        thread_indices: dict[int, int] = defaultdict(int)
        exposures_for_author = exposure_steps_by_user[author]
        observations = observations_by_author.get(author, [])
        authored_posts = posts_by_author.get(author, [])
        observation_index = 0

        for post in authored_posts:
            while observation_index < len(observations) and observations[observation_index][0] <= post.timestamp:
                observe_thread_messages(
                    author=author,
                    observation=observations[observation_index],
                    messages_by_thread=messages_by_thread,
                    thread_indices=thread_indices,
                    exposures_for_author=exposures_for_author,
                    viewers_by_message=viewers_by_message,
                )
                observation_index += 1
            post.candidate_parent_ids = list(exposures_for_author.keys())

        while observation_index < len(observations):
            observe_thread_messages(
                author=author,
                observation=observations[observation_index],
                messages_by_thread=messages_by_thread,
                thread_indices=thread_indices,
                exposures_for_author=exposures_for_author,
                viewers_by_message=viewers_by_message,
            )
            observation_index += 1

    for message in messages:
        message.viewers = viewers_by_message.get(message.id, set())
    return exposure_steps_by_user, viewers_by_message


def observe_thread_messages(
    *,
    author: str,
    observation: tuple[int, int],
    messages_by_thread: dict[int, list[MessageRecord]],
    thread_indices: dict[int, int],
    exposures_for_author: dict[int, int],
    viewers_by_message: dict[int, set[str]],
) -> None:
    step, thread_id = observation
    thread_messages = messages_by_thread.get(thread_id, [])
    index = thread_indices[thread_id]

    while index < len(thread_messages) and thread_messages[index].timestamp < step:
        message = thread_messages[index]
        if message.id not in exposures_for_author:
            exposures_for_author[message.id] = step
            viewers_by_message[message.id].add(author)
        index += 1

    thread_indices[thread_id] = index


def compute_influence_graph(
    messages: list[MessageRecord],
    *,
    beta: float = 0.05,
    edge_threshold: float = 0.05,
) -> dict[str, Any]:
    if not messages:
        empty_graph = nx.DiGraph()
        return {
            "graph": empty_graph,
            "filtered_graph": empty_graph.copy(),
            "edges": [],
            "filtered_edges": [],
            "top_parents_by_message": {},
            "top_children_by_message": {},
        }

    embeddings = np.vstack([message.embedding for message in messages]).astype(np.float32)
    timestamps = np.array([message.timestamp for message in messages], dtype=np.float32)
    full_graph = nx.DiGraph()
    filtered_graph = nx.DiGraph()
    full_graph.add_nodes_from(message.id for message in messages)
    filtered_graph.add_nodes_from(message.id for message in messages)

    edges: list[dict[str, Any]] = []
    filtered_edges: list[dict[str, Any]] = []
    top_parents_by_message: dict[int, list[dict[str, Any]]] = {}

    for message in messages:
        candidates = [candidate_id for candidate_id in message.candidate_parent_ids if candidate_id != message.id]
        if not candidates:
            top_parents_by_message[message.id] = []
            continue

        candidate_indices = np.array(candidates, dtype=np.int32)
        deltas = message.timestamp - timestamps[candidate_indices]
        valid_mask = deltas > 0
        if not np.any(valid_mask):
            top_parents_by_message[message.id] = []
            continue

        candidate_indices = candidate_indices[valid_mask]
        deltas = deltas[valid_mask]
        similarities = np.array(
            [similarity(embeddings[candidate_id], embeddings[message.id]) for candidate_id in candidate_indices],
            dtype=np.float32,
        )
        similarities = np.clip(similarities, a_min=0.0, a_max=None)
        weights = similarities * np.exp(-beta * deltas)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            top_parents_by_message[message.id] = []
            continue

        probabilities = weights / total_weight
        parent_records: list[dict[str, Any]] = []
        for candidate_id, similarity_score, delta, weight, probability in zip(
            candidate_indices.tolist(),
            similarities.tolist(),
            deltas.tolist(),
            weights.tolist(),
            probabilities.tolist(),
        ):
            edge_payload = {
                "source": int(candidate_id),
                "target": message.id,
                "probability": float(probability),
                "similarity": float(similarity_score),
                "time_delta": float(delta),
                "weight": float(weight),
            }
            edges.append(edge_payload)
            parent_records.append(edge_payload)
            full_graph.add_edge(candidate_id, message.id, probability=float(probability), similarity=float(similarity_score))
            if probability >= edge_threshold:
                filtered_edges.append(edge_payload)
                filtered_graph.add_edge(candidate_id, message.id, probability=float(probability), similarity=float(similarity_score))

        parent_records.sort(key=lambda item: item["probability"], reverse=True)
        top_parents_by_message[message.id] = parent_records[:10]

    top_children_by_message: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        top_children_by_message[edge["source"]].append(edge)

    top_children_by_message = {
        message_id: sorted(children, key=lambda item: item["probability"], reverse=True)[:10]
        for message_id, children in top_children_by_message.items()
    }

    return {
        "graph": full_graph,
        "filtered_graph": filtered_graph,
        "edges": edges,
        "filtered_edges": filtered_edges,
        "top_parents_by_message": top_parents_by_message,
        "top_children_by_message": top_children_by_message,
    }


def compute_message_impact(
    messages: list[MessageRecord],
    influence_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    impact_scores = np.zeros(len(messages), dtype=np.float32)
    for edge in influence_graph["edges"]:
        impact_scores[edge["source"]] += float(edge["probability"])

    results = []
    for message in messages:
        message.impact_score = float(impact_scores[message.id])
        results.append(
            {
                "message_id": message.id,
                "impact": float(impact_scores[message.id]),
                "author": message.author,
                "timestamp": message.timestamp,
                "thread_id": message.thread_id,
            }
        )

    results.sort(key=lambda item: item["impact"], reverse=True)
    return results


def compute_cascades(
    messages: list[MessageRecord],
    influence_graph: dict[str, Any],
    *,
    edge_threshold: float = 0.05,
) -> dict[str, Any]:
    filtered_graph: nx.DiGraph = influence_graph["filtered_graph"]
    best_parent_tree = nx.DiGraph()
    best_parent_tree.add_nodes_from(message.id for message in messages)

    incoming_edges: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in influence_graph["edges"]:
        incoming_edges[edge["target"]].append(edge)

    for target_id, edges in incoming_edges.items():
        best_edge = max(edges, key=lambda item: item["probability"])
        if best_edge["probability"] >= edge_threshold:
            best_parent_tree.add_edge(
                best_edge["source"],
                target_id,
                probability=float(best_edge["probability"]),
                similarity=float(best_edge["similarity"]),
            )

    cascades_by_message: dict[str, dict[str, Any]] = {}
    for message in messages:
        descendants = sorted(nx.descendants(best_parent_tree, message.id))
        direct_children = sorted(best_parent_tree.successors(message.id))
        subtree_nodes = {message.id, *descendants}
        subtree_weight = 0.0
        for source, target, attrs in best_parent_tree.edges(data=True):
            if source in subtree_nodes and target in subtree_nodes:
                subtree_weight += float(attrs.get("probability", 0.0))

        cascades_by_message[str(message.id)] = {
            "message_id": message.id,
            "cascade_size": len(descendants),
            "direct_children": direct_children,
            "descendants": descendants,
            "tree_probability_mass": subtree_weight,
        }

    roots = [node for node in best_parent_tree.nodes if best_parent_tree.in_degree(node) == 0 and best_parent_tree.out_degree(node) > 0]
    roots.sort(key=lambda node_id: cascades_by_message[str(node_id)]["cascade_size"], reverse=True)

    return {
        "filtered_edge_threshold": edge_threshold,
        "filtered_edges": influence_graph["filtered_edges"],
        "best_parent_tree_edges": [
            {
                "source": int(source),
                "target": int(target),
                "probability": float(attrs.get("probability", 0.0)),
                "similarity": float(attrs.get("similarity", 0.0)),
            }
            for source, target, attrs in best_parent_tree.edges(data=True)
        ],
        "roots": roots,
        "by_message": cascades_by_message,
        "filtered_component_sizes": sorted(
            (len(component) for component in nx.weakly_connected_components(filtered_graph) if component),
            reverse=True,
        ),
    }


def compute_drift(
    messages: list[MessageRecord],
    exposure_steps_by_user: dict[str, dict[int, int]],
    *,
    window_size: int = 50,
) -> dict[str, Any]:
    if not messages:
        return {"per_message": [], "global": []}

    user_messages: dict[str, list[MessageRecord]] = defaultdict(list)
    for message in messages:
        user_messages[message.author].append(message)

    per_message_totals = np.zeros(len(messages), dtype=np.float32)
    per_message_counts = np.zeros(len(messages), dtype=np.int32)

    for author, authored_messages in user_messages.items():
        authored_messages.sort(key=lambda msg: (msg.timestamp, msg.position, msg.id))
        if not authored_messages:
            continue

        authored_times = [message.timestamp for message in authored_messages]
        authored_ids = [message.id for message in authored_messages]
        authored_embeddings = np.vstack([messages[message_id].embedding for message_id in authored_ids]).astype(np.float32)
        prefix_sums = np.cumsum(authored_embeddings, axis=0)

        for exposed_message_id, exposure_step in exposure_steps_by_user.get(author, {}).items():
            previous_count = bisect_left(authored_times, exposure_step)
            next_index = bisect_left(authored_times, exposure_step)
            if previous_count == 0 or next_index >= len(authored_messages):
                continue

            previous_embedding = prefix_sums[previous_count - 1] / float(previous_count)
            next_embedding = authored_embeddings[next_index]
            delta = next_embedding - previous_embedding
            source_embedding = messages[exposed_message_id].embedding
            drift_value = float(np.dot(delta, source_embedding))
            per_message_totals[exposed_message_id] += drift_value
            per_message_counts[exposed_message_id] += 1

    per_message = []
    for message in messages:
        message.drift_score = float(per_message_totals[message.id])
        contribution_count = int(per_message_counts[message.id])
        per_message.append(
            {
                "message_id": message.id,
                "drift": float(per_message_totals[message.id]),
                "mean_drift": float(per_message_totals[message.id] / contribution_count) if contribution_count else 0.0,
                "contribution_count": contribution_count,
                "viewer_count": len(message.viewers),
                "timestamp": message.timestamp,
                "author": message.author,
            }
        )

    per_message.sort(key=lambda item: abs(item["drift"]), reverse=True)
    global_drift = compute_global_semantic_drift(messages, window_size=window_size)
    return {"per_message": per_message, "global": global_drift}


def compute_global_semantic_drift(messages: list[MessageRecord], *, window_size: int = 50) -> list[dict[str, Any]]:
    if not messages:
        return []

    windows: dict[int, list[np.ndarray]] = defaultdict(list)
    for message in messages:
        window_index = message.timestamp // max(window_size, 1)
        windows[window_index].append(message.embedding)

    sorted_windows = sorted(windows.items())
    series: list[dict[str, Any]] = []
    previous_mean: np.ndarray | None = None

    for window_index, embeddings in sorted_windows:
        mean_embedding = np.mean(np.vstack(embeddings), axis=0)
        drift_value = 0.0 if previous_mean is None else float(np.linalg.norm(mean_embedding - previous_mean))
        series.append(
            {
                "window_index": int(window_index),
                "window_start": int(window_index * window_size),
                "window_end": int((window_index + 1) * window_size - 1),
                "message_count": int(len(embeddings)),
                "drift": drift_value,
            }
        )
        previous_mean = mean_embedding

    return series


def build_serialized_messages(
    messages: list[MessageRecord],
    influence_graph: dict[str, Any],
    drift_metrics: dict[str, Any],
    *,
    include_embeddings: bool = True,
) -> list[dict[str, Any]]:
    drift_by_message = {item["message_id"]: item for item in drift_metrics["per_message"]}
    serialized_messages: list[dict[str, Any]] = []

    for message in messages:
        drift_entry = drift_by_message.get(message.id, {})
        top_parents = influence_graph["top_parents_by_message"].get(message.id, [])
        top_children = influence_graph["top_children_by_message"].get(message.id, [])
        payload = {
            "id": message.id,
            "thread_id": message.thread_id,
            "author": message.author,
            "author_node_id": message.author_node_id,
            "timestamp": message.timestamp,
            "position": message.position,
            "content": message.content,
            "impact_score": float(message.impact_score),
            "drift_score": float(drift_entry.get("drift", 0.0)),
            "mean_drift": float(drift_entry.get("mean_drift", 0.0)),
            "drift_contribution_count": int(drift_entry.get("contribution_count", 0)),
            "viewer_count": len(message.viewers),
            "candidate_parent_count": len(message.candidate_parent_ids),
            "top_parents": top_parents,
            "top_children": top_children,
        }
        if include_embeddings:
            payload["embedding"] = np.asarray(message.embedding, dtype=np.float32).round(6).tolist()
        serialized_messages.append(payload)

    return serialized_messages


def build_output_payload(
    sim_data: dict[str, Any],
    messages: list[MessageRecord],
    influence_graph: dict[str, Any],
    impact_scores: list[dict[str, Any]],
    cascades: dict[str, Any],
    drift_metrics: dict[str, Any],
    *,
    beta: float,
    edge_threshold: float,
    model_name: str,
    window_size: int,
    include_embeddings: bool,
) -> dict[str, Any]:
    output = dict(sim_data)
    output["influence_analysis"] = {
        "config": {
            "embedding_model": model_name,
            "beta": beta,
            "edge_threshold": edge_threshold,
            "window_size": window_size,
            "stores_embeddings": include_embeddings,
        },
        "messages": build_serialized_messages(
            messages,
            influence_graph,
            drift_metrics,
            include_embeddings=include_embeddings,
        ),
        "influence_graph": {
            "edge_count": len(influence_graph["edges"]),
            "filtered_edge_count": len(influence_graph["filtered_edges"]),
            "edges": influence_graph["edges"],
            "filtered_edges": influence_graph["filtered_edges"],
        },
        "impact_scores": impact_scores,
        "cascades": cascades,
        "discourse_drift": drift_metrics,
    }
    return output


def default_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}_influence{ext or '.json'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute semantic influence metrics from visualizer JSON.")
    parser.add_argument("inputs", nargs="+", help="Path(s) to simulator visualizer JSON file(s).")
    parser.add_argument("--output", default=None, help="Path to the enriched output JSON file.")
    parser.add_argument("--beta", type=float, default=0.05, help="Temporal decay parameter for Hawkes-style influence.")
    parser.add_argument("--edge-threshold", type=float, default=0.05, help="Minimum influence probability to keep in filtered cascade views.")
    parser.add_argument("--window-size", type=int, default=50, help="Time window size for global semantic drift.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--device", default=None, help="Embedding device override, for example cpu or cuda.")
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME, help="Transformer model used for message embeddings.")
    parser.add_argument("--omit-embeddings", action="store_true", help="Do not serialize per-message embedding vectors in the output file.")
    return parser.parse_args()


def resolve_output_path(input_path: str, *, output_arg: str | None, multiple_inputs: bool) -> str:
    if output_arg is None:
        return default_output_path(input_path)

    if multiple_inputs:
        raise ValueError("--output supports only a single input file. Omit --output for batch mode.")

    return output_arg


def process_file(
    input_path: str,
    *,
    output_path: str,
    embed_fn: Callable[[list[str]], np.ndarray],
    beta: float,
    edge_threshold: float,
    model_name: str,
    window_size: int,
    include_embeddings: bool,
) -> None:
    print(f"Loading simulator data from {input_path}")
    sim_data = load_simulation_data(input_path)
    messages = extract_messages(sim_data)
    print(f"Extracted {len(messages)} messages from {len(sim_data.get('threads', []))} threads.")

    print(f"Embedding messages with {model_name}")
    compute_embeddings(messages, embed=embed_fn)

    print("Reconstructing exposure histories")
    exposure_steps_by_user, _ = build_exposure_index(sim_data, messages)

    print("Computing semantic influence graph")
    influence_graph = compute_influence_graph(
        messages,
        beta=beta,
        edge_threshold=edge_threshold,
    )

    print("Computing impact scores")
    impact_scores = compute_message_impact(messages, influence_graph)

    print("Computing cascades")
    cascades = compute_cascades(
        messages,
        influence_graph,
        edge_threshold=edge_threshold,
    )

    print("Computing discourse drift")
    drift_metrics = compute_drift(
        messages,
        exposure_steps_by_user,
        window_size=window_size,
    )

    output_payload = build_output_payload(
        sim_data,
        messages,
        influence_graph,
        impact_scores,
        cascades,
        drift_metrics,
        beta=beta,
        edge_threshold=edge_threshold,
        model_name=model_name,
        window_size=window_size,
        include_embeddings=include_embeddings,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print(f"Wrote enriched influence analysis to {output_path}")


def main() -> None:
    args = parse_args()
    multiple_inputs = len(args.inputs) > 1

    print(f"Initializing embedder {args.model_name}")
    embedder = BertweetEmbedder(
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    for input_path in args.inputs:
        output_path = resolve_output_path(input_path, output_arg=args.output, multiple_inputs=multiple_inputs)
        process_file(
            input_path,
            output_path=output_path,
            embed_fn=embedder,
            beta=args.beta,
            edge_threshold=args.edge_threshold,
            model_name=args.model_name,
            window_size=args.window_size,
            include_embeddings=not args.omit_embeddings,
        )


if __name__ == "__main__":
    main()