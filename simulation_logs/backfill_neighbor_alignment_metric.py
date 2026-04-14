#!/usr/bin/env python3
"""Backfill neighbor-alignment herd metrics in visualizer_randomized JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add herd metric neighbor_alignment_shift_rate to existing "
            "visualizer_randomized_*.json files."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="visualizer_randomized_*.json",
        help="Glob pattern for visualizer files (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates but do not write files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite even if metric already exists.",
    )
    return parser.parse_args()


def unique_majority_option(neighbor_options: list[str]) -> str | None:
    if not neighbor_options:
        return None

    counts = Counter(neighbor_options)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def compute_neighbor_alignment_for_transition(
    transition: dict[str, Any],
    survey_by_step: dict[int, dict[str, str]],
    adjacency: list[set[int]],
    index_to_name: list[str | None],
    name_to_index: dict[str, int],
) -> tuple[int, int, float, float, float]:
    from_step_raw = transition.get("from_step")
    to_step_raw = transition.get("to_step")
    if from_step_raw is None or to_step_raw is None:
        return 0, 0, 0.0, 0.0, 0.0

    try:
        from_step = int(from_step_raw)
        to_step = int(to_step_raw)
    except (TypeError, ValueError):
        return 0, 0, 0.0, 0.0, 0.0

    prev_results = survey_by_step.get(from_step, {})
    curr_results = survey_by_step.get(to_step, {})
    if not prev_results or not curr_results:
        return 0, 0, 0.0, 0.0, 0.0

    shared_users = set(prev_results).intersection(curr_results)
    if not shared_users:
        return 0, 0, 0.0, 0.0, 0.0

    changed_users = {
        name for name in shared_users if prev_results[name] != curr_results[name]
    }

    eligible_users = 0
    changed_to_neighbor_majority = 0

    for user_name in shared_users:
        user_index = name_to_index.get(user_name)
        if user_index is None or user_index < 0 or user_index >= len(adjacency):
            continue

        neighbor_options: list[str] = []
        for neighbor_index in adjacency[user_index]:
            if neighbor_index < 0 or neighbor_index >= len(index_to_name):
                continue
            neighbor_name = index_to_name[neighbor_index]
            if neighbor_name is None or neighbor_name not in shared_users:
                continue
            option = prev_results.get(neighbor_name)
            if option is not None:
                neighbor_options.append(option)

        majority_option = unique_majority_option(neighbor_options)
        if majority_option is None:
            continue

        eligible_users += 1
        if user_name in changed_users and curr_results[user_name] == majority_option:
            changed_to_neighbor_majority += 1

    n_shared = int(transition.get("shared_users", len(shared_users)))
    n_changed = int(transition.get("changed_users", len(changed_users)))

    shift_rate = (
        float(changed_to_neighbor_majority / n_shared)
        if n_shared > 0 else 0.0
    )
    among_changers_rate = (
        float(changed_to_neighbor_majority / n_changed)
        if n_changed > 0 else 0.0
    )
    eligible_shift_rate = (
        float(changed_to_neighbor_majority / eligible_users)
        if eligible_users > 0 else 0.0
    )

    return (
        eligible_users,
        changed_to_neighbor_majority,
        shift_rate,
        among_changers_rate,
        eligible_shift_rate,
    )


def backfill_file(file_path: str, force: bool = False, dry_run: bool = False) -> tuple[bool, str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"read_error: {exc}"

    if not isinstance(payload, dict):
        return False, "invalid_payload"

    behavioral = payload.get("behavioral_metrics")
    if not isinstance(behavioral, dict):
        return False, "missing_behavioral_metrics"

    herd = behavioral.get("herd_effect_metrics")
    if not isinstance(herd, dict):
        return False, "missing_herd_effect_metrics"

    transitions = herd.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return False, "missing_transitions"

    missing_metric = any(
        isinstance(tr, dict) and "neighbor_alignment_shift_rate" not in tr
        for tr in transitions
    )
    if not force and not missing_metric and "mean_neighbor_alignment_shift_rate" in herd:
        return False, "already_present"

    surveys = payload.get("survey_results", [])
    if not isinstance(surveys, list):
        return False, "missing_survey_results"

    survey_by_step: dict[int, dict[str, str]] = {}
    for survey in surveys:
        if not isinstance(survey, dict):
            continue
        step_raw = survey.get("step")
        results = survey.get("results", {})
        if not isinstance(results, dict):
            continue
        try:
            step = int(step_raw)
        except (TypeError, ValueError):
            continue
        survey_by_step[step] = {
            str(name): str(option)
            for name, option in results.items()
        }

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False, "missing_graph"

    max_index = max(len(nodes) - 1, 0)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        s = edge.get("source")
        t = edge.get("target")
        if isinstance(s, int):
            max_index = max(max_index, s)
        if isinstance(t, int):
            max_index = max(max_index, t)

    index_to_name: list[str | None] = [None] * (max_index + 1)
    name_to_index: dict[str, int] = {}

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_name = node.get("name")
        if not isinstance(node_name, str):
            continue
        node_index = node.get("id", index)
        if not isinstance(node_index, int) or node_index < 0:
            continue
        if node_index >= len(index_to_name):
            index_to_name.extend([None] * (node_index - len(index_to_name) + 1))
        index_to_name[node_index] = node_name
        name_to_index[node_name] = node_index

    adjacency: list[set[int]] = [set() for _ in range(len(index_to_name))]
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, int) or not isinstance(target, int):
            continue
        if source < 0 or target < 0:
            continue
        if source >= len(adjacency):
            adjacency.extend([set() for _ in range(source - len(adjacency) + 1)])
        adjacency[source].add(target)

    shift_rates: list[float] = []
    changed_any = False

    for transition in transitions:
        if not isinstance(transition, dict):
            continue
        (
            eligible_users,
            changed_to_neighbor_majority,
            shift_rate,
            among_changers_rate,
            eligible_shift_rate,
        ) = compute_neighbor_alignment_for_transition(
            transition,
            survey_by_step,
            adjacency,
            index_to_name,
            name_to_index,
        )

        transition["neighbor_majority_eligible_user_count"] = eligible_users
        transition["changed_to_neighbor_majority_count"] = changed_to_neighbor_majority
        transition["neighbor_alignment_shift_rate"] = shift_rate
        transition["neighbor_alignment_among_changers_rate"] = among_changers_rate
        transition["neighbor_alignment_eligible_shift_rate"] = eligible_shift_rate

        shift_rates.append(shift_rate)
        changed_any = True

    if not changed_any:
        return False, "no_valid_transitions"

    herd["mean_neighbor_alignment_shift_rate"] = float(mean(shift_rates)) if shift_rates else 0.0

    if not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return True, "updated"


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    updated = 0
    skipped = 0
    errors = 0

    for file_path in files:
        changed, status = backfill_file(file_path, force=args.force, dry_run=args.dry_run)
        if changed:
            updated += 1
            if args.dry_run:
                print(f"[DRY RUN] would update: {file_path}")
            else:
                print(f"updated: {file_path}")
        else:
            if status.startswith("read_error"):
                errors += 1
                print(f"error: {file_path} -> {status}")
            else:
                skipped += 1

    print(
        f"done files={len(files)} updated={updated} skipped={skipped} errors={errors} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
