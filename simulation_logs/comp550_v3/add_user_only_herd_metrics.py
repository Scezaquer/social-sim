#!/usr/bin/env python3
"""Add user_only_herd_effect_metrics to one or more visualizer JSON files.

This script computes herd-effect metrics while excluding adversarial agents from
user-level counts (shared users, changed users, consensus, diversity, etc.).
Adversarial neighbors are still allowed to influence neighborhood-majority
calculations for non-adversarial users.

By default, the script updates files in place and appends metrics at:
  behavioral_metrics.user_only_herd_effect_metrics

Examples:
  python add_user_only_herd_metrics.py visualizer_comp550_9317783_13.json
  python add_user_only_herd_metrics.py visualizer_comp550_*.json
  python add_user_only_herd_metrics.py "visualizer_comp550_*.json" --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any


def normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    probs = [count / total for count in counts.values() if count > 0]
    if len(probs) <= 1:
        return 0.0

    import math

    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return 0.0

    return float(entropy / max_entropy)


def merge_run_parameters(data: dict[str, Any]) -> dict[str, Any]:
    run_parameters = data.get("run_parameters", {})
    if not isinstance(run_parameters, dict):
        return {}

    merged = dict(run_parameters)
    cli_args = run_parameters.get("cli_args")
    if isinstance(cli_args, dict):
        for key, value in cli_args.items():
            merged.setdefault(key, value)
    return merged


def build_name_to_idx(nodes: list[dict[str, Any]]) -> dict[str, int]:
    name_to_idx: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        name = node.get("name")
        if isinstance(name, str):
            name_to_idx[name] = idx
    return name_to_idx


def build_social_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[set[int]]:
    graph: list[set[int]] = [set() for _ in range(len(nodes))]

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        if source < 0 or target < 0:
            continue
        if source >= len(nodes) or target >= len(nodes):
            continue

        graph[source].add(target)

    return graph


def resolve_node_name_for_entity_id(nodes: list[dict[str, Any]], entity_id: int) -> str | None:
    if 0 <= entity_id < len(nodes):
        by_index = nodes[entity_id].get("name")
        if isinstance(by_index, str):
            return by_index

    for node in nodes:
        node_name = node.get("name")
        if not isinstance(node_name, str):
            continue

        if node.get("entity_id") == entity_id:
            return node_name
        if node.get("id") == entity_id:
            return node_name

    return None


def identify_adversarial_names(data: dict[str, Any]) -> set[str]:
    run_parameters = merge_run_parameters(data)
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        nodes = []

    adversarial_names: set[str] = set()

    # 1) Explicit adversarial names.
    names = run_parameters.get("adversarial_agent_names", [])
    if isinstance(names, list):
        for name in names:
            if isinstance(name, str) and name:
                adversarial_names.add(name)

    # 2) Structured adversarial agent records.
    agents = run_parameters.get("adversarial_agents", [])
    if isinstance(agents, list):
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            name = agent.get("name")
            if isinstance(name, str) and name:
                adversarial_names.add(name)

    # 3) Adversarial entity IDs mapped to node names.
    ids = run_parameters.get("adversarial_agent_entity_ids", [])
    if isinstance(ids, list):
        for entity_id in ids:
            if not isinstance(entity_id, int):
                continue
            maybe_name = resolve_node_name_for_entity_id(nodes, entity_id)
            if maybe_name is not None:
                adversarial_names.add(maybe_name)

    # 4) Conservative fallback from known contiguous user indexing.
    if not adversarial_names:
        n_normal = run_parameters.get("number_normal_users")
        n_adversarial = run_parameters.get("number_adversarial_users")
        if isinstance(n_normal, int) and isinstance(n_adversarial, int):
            for entity_id in range(n_normal, n_normal + n_adversarial):
                maybe_name = resolve_node_name_for_entity_id(nodes, entity_id)
                if maybe_name is not None:
                    adversarial_names.add(maybe_name)

    return adversarial_names


def compute_user_only_herd_effect_metrics(data: dict[str, Any]) -> dict[str, Any]:
    surveys = data.get("survey_results", [])
    if not isinstance(surveys, list):
        surveys = []

    if len(surveys) < 2:
        return {
            "status": "insufficient_data",
            "reason": "Need at least two survey snapshots.",
            "survey_count": len(surveys),
            "transition_count": 0,
        }

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []

    node_names: list[str | None] = [
        node.get("name") if isinstance(node, dict) else None for node in nodes
    ]
    name_to_idx = build_name_to_idx(nodes)  # maps name -> node index
    social_graph = build_social_graph(nodes, edges)
    adversarial_names = identify_adversarial_names(data)

    transitions: list[dict[str, Any]] = []
    herd_follow_rates: list[float] = []
    shift_rates: list[float] = []
    consensus_gains: list[float] = []
    neighbor_alignment_shift_rates: list[float] = []

    def neighbor_majority_option(
        user_name: str,
        prev_results: dict[str, str],
        shared_users_all: set[str],
    ) -> str | None:
        user_idx = name_to_idx.get(user_name)
        if user_idx is None or user_idx < 0 or user_idx >= len(social_graph):
            return None

        neighbor_options: list[str] = []
        for neighbor_idx in social_graph[user_idx]:
            if neighbor_idx < 0 or neighbor_idx >= len(node_names):
                continue
            neighbor_name = node_names[neighbor_idx]
            if neighbor_name is None:
                continue
            if neighbor_name not in shared_users_all:
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
        if not isinstance(prev, dict) or not isinstance(curr, dict):
            continue

        prev_results = prev.get("results", {})
        curr_results = curr.get("results", {})
        if not isinstance(prev_results, dict) or not isinstance(curr_results, dict):
            continue

        shared_users_all = set(prev_results).intersection(curr_results)
        shared_users = {u for u in shared_users_all if u not in adversarial_names}
        if not shared_users:
            continue

        changed_users = [name for name in shared_users if prev_results[name] != curr_results[name]]

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
            neighbor_majority = neighbor_majority_option(name, prev_results, shared_users_all)
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
            "reason": "No overlapping non-adversarial users between consecutive surveys.",
            "survey_count": len(surveys),
            "transition_count": 0,
            "excluded_adversarial_user_count": len(adversarial_names),
        }

    first_results = surveys[0].get("results", {}) if isinstance(surveys[0], dict) else {}
    last_results = surveys[-1].get("results", {}) if isinstance(surveys[-1], dict) else {}
    if not isinstance(first_results, dict):
        first_results = {}
    if not isinstance(last_results, dict):
        last_results = {}

    first_counts = Counter(
        option for name, option in first_results.items() if name not in adversarial_names
    )
    last_counts = Counter(
        option for name, option in last_results.items() if name not in adversarial_names
    )

    first_total = sum(first_counts.values())
    last_total = sum(last_counts.values())

    first_consensus = (
        max(first_counts.values()) / first_total if first_total > 0 and first_counts else 0.0
    )
    last_consensus = (
        max(last_counts.values()) / last_total if last_total > 0 and last_counts else 0.0
    )

    return {
        "status": "ok",
        "survey_count": len(surveys),
        "transition_count": len(transitions),
        "mean_opinion_shift_rate": float(sum(shift_rates) / len(shift_rates)) if shift_rates else 0.0,
        "mean_current_majority_follow_rate": float(sum(herd_follow_rates) / len(herd_follow_rates))
        if herd_follow_rates
        else 0.0,
        "mean_consensus_gain": float(sum(consensus_gains) / len(consensus_gains)) if consensus_gains else 0.0,
        "mean_neighbor_alignment_shift_rate": float(
            sum(neighbor_alignment_shift_rates) / len(neighbor_alignment_shift_rates)
        )
        if neighbor_alignment_shift_rates
        else 0.0,
        "initial_consensus": float(first_consensus),
        "final_consensus": float(last_consensus),
        "net_consensus_change": float(last_consensus - first_consensus),
        "initial_diversity": normalized_entropy(first_counts),
        "final_diversity": normalized_entropy(last_counts),
        "excluded_adversarial_user_count": len(adversarial_names),
        "transitions": transitions,
    }


def add_metrics_to_file(path: Path, indent: int, dry_run: bool) -> tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover
        return False, f"{path}: failed to read JSON ({exc})"

    if not isinstance(data, dict):
        return False, f"{path}: top-level JSON is not an object"

    user_only_metrics = compute_user_only_herd_effect_metrics(data)

    if dry_run:
        status = user_only_metrics.get("status")
        transitions = user_only_metrics.get("transition_count")
        excluded = user_only_metrics.get("excluded_adversarial_user_count", 0)
        return True, (
            f"{path}: dry-run status={status}, transitions={transitions}, "
            f"excluded_adversarial_user_count={excluded}"
        )

    behavioral_metrics = data.get("behavioral_metrics")
    if not isinstance(behavioral_metrics, dict):
        behavioral_metrics = {}
        data["behavioral_metrics"] = behavioral_metrics

    behavioral_metrics["user_only_herd_effect_metrics"] = user_only_metrics

    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.write("\n")
        os.replace(temp_path, path)
    except Exception as exc:  # pragma: no cover
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False, f"{path}: failed to write updated JSON ({exc})"

    status = user_only_metrics.get("status")
    transitions = user_only_metrics.get("transition_count")
    excluded = user_only_metrics.get("excluded_adversarial_user_count", 0)
    return True, (
        f"{path}: updated user_only_herd_effect_metrics "
        f"(status={status}, transitions={transitions}, "
        f"excluded_adversarial_user_count={excluded})"
    )


def expand_input_paths(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        matches = sorted(glob(raw))
        candidates = [Path(m) for m in matches] if matches else [Path(raw)]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            resolved.append(candidate)

    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and append user-only herd-effect metrics to one or more "
            "visualizer JSON files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON paths and/or glob patterns (supports multiple files).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics and print summaries without modifying files.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation for in-place writes (default: 4).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    paths = expand_input_paths(args.inputs)
    if not paths:
        print("No input files provided.")
        return 1

    failures = 0
    for path in paths:
        if not path.exists():
            print(f"{path}: file not found")
            failures += 1
            continue
        if not path.is_file():
            print(f"{path}: not a file")
            failures += 1
            continue

        ok, message = add_metrics_to_file(path, indent=args.indent, dry_run=args.dry_run)
        print(message)
        if not ok:
            failures += 1

    if failures:
        print(f"Done with {failures} failure(s).")
        return 1

    print(f"Done. Processed {len(paths)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
