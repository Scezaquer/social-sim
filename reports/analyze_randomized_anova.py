#!/usr/bin/env python3
# Output quick guide:
# - Writes anova_randomized_q<question>.json and anova_randomized_q<question>.csv.
# - For each metric/parameter pair, eta_squared estimates the fraction of
#   within-timestep variance explained by that parameter (higher means stronger).
# - sigma2_estimate approximates run-to-run Gaussian noise variance for fixed
#   full parameter settings when repeated runs are available.
"""ANOVA-style variance attribution for randomized visualizer runs.

This script analyzes visualizer_randomized_*.json files and estimates how much of
metric variance is attributable to each simulation parameter while controlling for
timestep (i.e., only comparing runs at identical steps).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


PARAMETERS = [
    "base_model",
    "num_agents",
    "num_news_agents",
    "graph_type",
    "homophily",
    "add_survey_to_context",
    "proportions",
]

BASE_METRICS = [
    "majority_follow",
    "consensus",
    "diversity",
    "opinion_shift_rate",
    "neighbor_alignment_shift_rate",
]


@dataclass
class RunRecord:
    file_path: str
    question_number: int
    params: Dict[str, Any]
    full_param_signature: Tuple[Any, ...]
    vote_pct_by_step: Dict[int, Dict[str, float]]
    majority_follow_by_step: Dict[int, float]
    consensus_by_step: Dict[int, float]
    diversity_by_step: Dict[int, float]
    opinion_shift_by_step: Dict[int, float]
    consensus_gain_by_step: Dict[int, float]
    moved_to_prev_majority_by_step: Dict[int, float]
    neighbor_alignment_shift_by_step: Dict[int, float]
    echo_assortativity_by_step: Dict[int, float]
    echo_local_agreement_by_step: Dict[int, float]
    echo_cross_cutting_by_step: Dict[int, float]
    echo_same_option_exposure_by_step: Dict[int, float]
    echo_exposure_diversity_by_step: Dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-parameter contribution to metric variance for "
            "visualizer_randomized_*.json runs."
        )
    )
    parser.add_argument(
        "--input-glob",
        default="visualizer_randomized_*.json",
        help="Glob pattern used to find visualizer JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--question-number",
        type=int,
        required=True,
        help="Only analyze runs with this question_number.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path (default: anova_randomized_q<question>.json)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: anova_randomized_q<question>.csv)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of files to process (0 means all).",
    )
    return parser.parse_args()


def canonicalize_param_value(param_name: str, value: Any) -> Any:
    if param_name == "proportions":
        if value is None:
            return None
        if not isinstance(value, list):
            return value
        # Round to stabilize floating-point serialization for exact comparisons.
        return tuple(round(float(v), 12) for v in value)
    return value


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def extract_vote_percentages(survey_results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    vote_pct_by_step: Dict[int, Dict[str, float]] = {}

    for entry in survey_results:
        step = entry.get("step")
        results = entry.get("results", {})
        if step is None or not isinstance(results, dict):
            continue

        counts = Counter(results.values())
        total = sum(counts.values())
        if total <= 0:
            continue

        percentages = {
            str(option): (count / total) * 100.0 for option, count in counts.items()
        }
        vote_pct_by_step[int(step)] = percentages

    return vote_pct_by_step


def extract_herd_metrics(behavioral_metrics: Dict[str, Any]) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]
]:
    majority_follow_by_step: Dict[int, float] = {}
    consensus_by_step: Dict[int, float] = {}
    diversity_by_step: Dict[int, float] = {}
    opinion_shift_by_step: Dict[int, float] = {}
    consensus_gain_by_step: Dict[int, float] = {}
    moved_to_prev_majority_by_step: Dict[int, float] = {}
    neighbor_alignment_shift_by_step: Dict[int, float] = {}

    herd = behavioral_metrics.get("herd_effect_metrics", {}) if isinstance(behavioral_metrics, dict) else {}
    transitions = herd.get("transitions", []) if isinstance(herd, dict) else []

    initial_consensus = safe_float(herd.get("initial_consensus")) if isinstance(herd, dict) else None
    initial_diversity = safe_float(herd.get("initial_diversity")) if isinstance(herd, dict) else None

    if transitions and isinstance(transitions, list):
        first = transitions[0]
        if isinstance(first, dict):
            from_step = first.get("from_step")
            if from_step is not None:
                if initial_consensus is not None:
                    consensus_by_step[int(from_step)] = initial_consensus
                if initial_diversity is not None:
                    diversity_by_step[int(from_step)] = initial_diversity

    for tr in transitions:
        if not isinstance(tr, dict):
            continue
        to_step = tr.get("to_step")
        if to_step is None:
            continue
        step = int(to_step)

        val = safe_float(tr.get("current_majority_follow_rate"))
        if val is not None:
            majority_follow_by_step[step] = val

        val = safe_float(tr.get("current_consensus"))
        if val is not None:
            consensus_by_step[step] = val

        val = safe_float(tr.get("current_diversity"))
        if val is not None:
            diversity_by_step[step] = val

        val = safe_float(tr.get("opinion_shift_rate"))
        if val is not None:
            opinion_shift_by_step[step] = val

        val = safe_float(tr.get("consensus_gain"))
        if val is not None:
            consensus_gain_by_step[step] = val

        val = safe_float(tr.get("moved_to_previous_majority_rate"))
        if val is not None:
            moved_to_prev_majority_by_step[step] = val

        val = safe_float(tr.get("neighbor_alignment_shift_rate"))
        if val is not None:
            neighbor_alignment_shift_by_step[step] = val

    return (
        majority_follow_by_step,
        consensus_by_step,
        diversity_by_step,
        opinion_shift_by_step,
        consensus_gain_by_step,
        moved_to_prev_majority_by_step,
        neighbor_alignment_shift_by_step,
    )


def extract_echo_metrics(behavioral_metrics: Dict[str, Any]) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]
]:
    assortativity_by_step: Dict[int, float] = {}
    local_agreement_by_step: Dict[int, float] = {}
    cross_cutting_by_step: Dict[int, float] = {}
    same_option_exposure_by_step: Dict[int, float] = {}
    exposure_diversity_by_step: Dict[int, float] = {}

    echo = behavioral_metrics.get("echo_chamber_metrics", {}) if isinstance(behavioral_metrics, dict) else {}
    if not isinstance(echo, dict):
        return (
            assortativity_by_step,
            local_agreement_by_step,
            cross_cutting_by_step,
            same_option_exposure_by_step,
            exposure_diversity_by_step,
        )

    if isinstance(echo.get("by_survey"), list):
        snapshots = [s for s in echo.get("by_survey", []) if isinstance(s, dict)]
    elif echo.get("survey_step") is not None:
        snapshots = [echo]
    else:
        snapshots = []

    for snapshot in snapshots:
        step = snapshot.get("survey_step")
        if step is None:
            continue

        try:
            step_int = int(step)
        except (TypeError, ValueError):
            continue

        val = safe_float(snapshot.get("network_assortativity"))
        if val is not None:
            assortativity_by_step[step_int] = val

        val = safe_float(snapshot.get("mean_local_agreement"))
        if val is not None:
            local_agreement_by_step[step_int] = val

        val = safe_float(snapshot.get("cross_cutting_edge_fraction"))
        if val is not None:
            cross_cutting_by_step[step_int] = val

        val = safe_float(snapshot.get("mean_same_option_exposure_share"))
        if val is not None:
            same_option_exposure_by_step[step_int] = val

        val = safe_float(snapshot.get("mean_exposure_diversity"))
        if val is not None:
            exposure_diversity_by_step[step_int] = val

    return (
        assortativity_by_step,
        local_agreement_by_step,
        cross_cutting_by_step,
        same_option_exposure_by_step,
        exposure_diversity_by_step,
    )


def load_run_record(file_path: str) -> Optional[RunRecord]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    run_params = payload.get("run_parameters", {})
    if not isinstance(run_params, dict):
        return None

    question_number_raw = run_params.get("question_number")
    try:
        question_number = int(question_number_raw)
    except (TypeError, ValueError):
        return None

    params: Dict[str, Any] = {}
    for name in PARAMETERS:
        if name == "proportions":
            proportions_option = run_params.get("proportions_option")
            if proportions_option is not None:
                params[name] = str(proportions_option)
            else:
                params[name] = canonicalize_param_value(name, run_params.get(name))
        else:
            params[name] = canonicalize_param_value(name, run_params.get(name))
    full_param_signature = tuple(params[name] for name in PARAMETERS)

    survey_results = payload.get("survey_results", [])
    if not isinstance(survey_results, list):
        survey_results = []

    vote_pct_by_step = extract_vote_percentages(survey_results)

    behavioral_metrics = payload.get("behavioral_metrics", {})
    (
        majority_follow_by_step,
        consensus_by_step,
        diversity_by_step,
        opinion_shift_by_step,
        consensus_gain_by_step,
        moved_to_prev_majority_by_step,
        neighbor_alignment_shift_by_step,
    ) = extract_herd_metrics(behavioral_metrics)

    (
        echo_assortativity_by_step,
        echo_local_agreement_by_step,
        echo_cross_cutting_by_step,
        echo_same_option_exposure_by_step,
        echo_exposure_diversity_by_step,
    ) = extract_echo_metrics(behavioral_metrics)

    return RunRecord(
        file_path=file_path,
        question_number=question_number,
        params=params,
        full_param_signature=full_param_signature,
        vote_pct_by_step=vote_pct_by_step,
        majority_follow_by_step=majority_follow_by_step,
        consensus_by_step=consensus_by_step,
        diversity_by_step=diversity_by_step,
        opinion_shift_by_step=opinion_shift_by_step,
        consensus_gain_by_step=consensus_gain_by_step,
        moved_to_prev_majority_by_step=moved_to_prev_majority_by_step,
        neighbor_alignment_shift_by_step=neighbor_alignment_shift_by_step,
        echo_assortativity_by_step=echo_assortativity_by_step,
        echo_local_agreement_by_step=echo_local_agreement_by_step,
        echo_cross_cutting_by_step=echo_cross_cutting_by_step,
        echo_same_option_exposure_by_step=echo_same_option_exposure_by_step,
        echo_exposure_diversity_by_step=echo_exposure_diversity_by_step,
    )


def build_metric_observations(
    runs: List[RunRecord],
) -> Dict[str, List[Tuple[int, float, Dict[str, Any], Tuple[Any, ...]]]]:
    """Return metric_name -> list of (step, value, params, full_signature)."""
    observations: Dict[str, List[Tuple[int, float, Dict[str, Any], Tuple[Any, ...]]]] = defaultdict(list)

    # For vote percentages, build per-step option universe so missing options become 0%.
    vote_options_by_step: Dict[int, set] = defaultdict(set)
    for run in runs:
        for step, pct_map in run.vote_pct_by_step.items():
            vote_options_by_step[step].update(str(opt) for opt in pct_map.keys())

    for run in runs:
        for step, options in vote_options_by_step.items():
            pct_map = run.vote_pct_by_step.get(step, {})
            for option in options:
                value = safe_float(pct_map.get(option, 0.0))
                if value is None:
                    continue
                metric_name = f"vote_pct::{option}"
                observations[metric_name].append((step, value, run.params, run.full_param_signature))

        for step, value in run.majority_follow_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["majority_follow"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.consensus_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["consensus"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.diversity_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["diversity"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.opinion_shift_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["opinion_shift_rate"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.consensus_gain_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["consensus_gain"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.moved_to_prev_majority_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["moved_to_previous_majority_rate"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.neighbor_alignment_shift_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["neighbor_alignment_shift_rate"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.echo_assortativity_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["echo_network_assortativity"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.echo_local_agreement_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["echo_mean_local_agreement"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.echo_cross_cutting_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["echo_cross_cutting_edge_fraction"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.echo_same_option_exposure_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["echo_mean_same_option_exposure_share"].append((step, val, run.params, run.full_param_signature))

        for step, value in run.echo_exposure_diversity_by_step.items():
            val = safe_float(value)
            if val is not None:
                observations["echo_mean_exposure_diversity"].append((step, val, run.params, run.full_param_signature))

    return observations


def anova_one_parameter_stratified_by_step(
    records: List[Tuple[int, float, Dict[str, Any], Tuple[Any, ...]]],
    parameter: str,
) -> Dict[str, Any]:
    by_step: Dict[int, List[Tuple[float, Any]]] = defaultdict(list)
    for step, value, params, _sig in records:
        by_step[step].append((value, params.get(parameter)))

    ss_total = 0.0
    ss_param = 0.0
    ss_residual = 0.0
    steps_used = 0

    per_step_details = []
    all_levels = set()
    singleton_levels = 0
    level_counts: Counter = Counter()

    for step, vals in sorted(by_step.items()):
        if len(vals) < 2:
            continue

        ys = [y for y, _ in vals]
        mean_all = sum(ys) / len(ys)
        sst = sum((y - mean_all) ** 2 for y in ys)

        groups: Dict[Any, List[float]] = defaultdict(list)
        for y, lvl in vals:
            groups[lvl].append(y)

        ss_between = 0.0
        for lvl, grp in groups.items():
            m = sum(grp) / len(grp)
            ss_between += len(grp) * ((m - mean_all) ** 2)
            all_levels.add(lvl)
            level_counts[lvl] += len(grp)

        ss_within = max(0.0, sst - ss_between)

        if sst > 0:
            steps_used += 1
            ss_total += sst
            ss_param += ss_between
            ss_residual += ss_within

            per_step_details.append(
                {
                    "step": step,
                    "n": len(vals),
                    "levels": len(groups),
                    "eta_squared": ss_between / sst if sst > 0 else None,
                    "ss_total": sst,
                    "ss_parameter": ss_between,
                    "ss_residual": ss_within,
                }
            )

    for lvl in all_levels:
        if level_counts[lvl] == 1:
            singleton_levels += 1

    eta_squared = (ss_param / ss_total) if ss_total > 0 else None

    return {
        "parameter": parameter,
        "n_observations": len(records),
        "n_steps_total": len(by_step),
        "n_steps_used": steps_used,
        "n_levels": len(all_levels),
        "n_singleton_levels": singleton_levels,
        "singleton_level_fraction": (
            singleton_levels / len(all_levels) if len(all_levels) > 0 else None
        ),
        "ss_total": ss_total,
        "ss_parameter": ss_param,
        "ss_residual": ss_residual,
        "eta_squared": eta_squared,
        "per_step": per_step_details,
    }


def estimate_noise_variance(
    records: List[Tuple[int, float, Dict[str, Any], Tuple[Any, ...]]]
) -> Dict[str, Any]:
    """Estimate sigma^2 from repeated full-parameter signatures at each timestep."""
    by_step_sig: Dict[int, Dict[Tuple[Any, ...], List[float]]] = defaultdict(lambda: defaultdict(list))

    for step, value, _params, signature in records:
        by_step_sig[step][signature].append(value)

    weighted_ss = 0.0
    weighted_df = 0
    repeated_groups = 0

    for step, groups in by_step_sig.items():
        for _sig, ys in groups.items():
            n = len(ys)
            if n < 2:
                continue
            repeated_groups += 1
            mean_y = sum(ys) / n
            group_ss = sum((y - mean_y) ** 2 for y in ys)
            weighted_ss += group_ss
            weighted_df += (n - 1)

    sigma2 = (weighted_ss / weighted_df) if weighted_df > 0 else None

    return {
        "sigma2_estimate": sigma2,
        "sigma_estimate": math.sqrt(sigma2) if sigma2 is not None else None,
        "repeated_groups": repeated_groups,
        "degrees_of_freedom": weighted_df,
    }


def to_csv_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric_name, metric_result in results["metrics"].items():
        noise = metric_result["noise_estimate"]
        for anova in metric_result["anova_by_parameter"]:
            rows.append(
                {
                    "metric": metric_name,
                    "parameter": anova["parameter"],
                    "eta_squared": anova["eta_squared"],
                    "ss_total": anova["ss_total"],
                    "ss_parameter": anova["ss_parameter"],
                    "ss_residual": anova["ss_residual"],
                    "n_observations": anova["n_observations"],
                    "n_steps_total": anova["n_steps_total"],
                    "n_steps_used": anova["n_steps_used"],
                    "n_levels": anova["n_levels"],
                    "n_singleton_levels": anova["n_singleton_levels"],
                    "singleton_level_fraction": anova["singleton_level_fraction"],
                    "sigma2_estimate": noise["sigma2_estimate"],
                    "sigma_estimate": noise["sigma_estimate"],
                    "noise_repeated_groups": noise["repeated_groups"],
                    "noise_df": noise["degrees_of_freedom"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()

    input_files = sorted(glob.glob(args.input_glob))
    if args.max_files and args.max_files > 0:
        input_files = input_files[: args.max_files]

    if not input_files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    runs: List[RunRecord] = []
    skipped = 0
    for file_path in input_files:
        rec = load_run_record(file_path)
        if rec is None:
            skipped += 1
            continue
        if rec.question_number != args.question_number:
            continue
        runs.append(rec)

    if not runs:
        raise SystemExit(
            "No valid runs found for question_number="
            f"{args.question_number} in pattern {args.input_glob}."
        )

    observations = build_metric_observations(runs)

    results: Dict[str, Any] = {
        "question_number": args.question_number,
        "input_glob": args.input_glob,
        "run_count": len(runs),
        "file_count_matched": len(input_files),
        "file_count_skipped_invalid": skipped,
        "parameters": PARAMETERS,
        "metrics": {},
    }

    for metric_name, metric_records in sorted(observations.items()):
        metric_noise = estimate_noise_variance(metric_records)
        anova = [
            anova_one_parameter_stratified_by_step(metric_records, p) for p in PARAMETERS
        ]
        anova.sort(
            key=lambda row: (
                -1.0 if row["eta_squared"] is None else -row["eta_squared"],
                row["parameter"],
            )
        )

        results["metrics"][metric_name] = {
            "n_observations": len(metric_records),
            "noise_estimate": metric_noise,
            "anova_by_parameter": anova,
        }

    output_json = (
        args.output_json
        if args.output_json
        else f"anova_randomized_q{args.question_number}.json"
    )
    output_csv = (
        args.output_csv
        if args.output_csv
        else f"anova_randomized_q{args.question_number}.csv"
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    csv_rows = to_csv_rows(results)
    fieldnames = [
        "metric",
        "parameter",
        "eta_squared",
        "ss_total",
        "ss_parameter",
        "ss_residual",
        "n_observations",
        "n_steps_total",
        "n_steps_used",
        "n_levels",
        "n_singleton_levels",
        "singleton_level_fraction",
        "sigma2_estimate",
        "sigma_estimate",
        "noise_repeated_groups",
        "noise_df",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("ANOVA analysis complete")
    print(f"question_number={args.question_number}")
    print(f"runs_used={len(runs)}")
    print(f"json_output={output_json}")
    print(f"csv_output={output_csv}")

    for metric_name, metric_result in sorted(results["metrics"].items()):
        print(f"\n[{metric_name}]")
        for row in metric_result["anova_by_parameter"]:
            eta = row["eta_squared"]
            eta_text = "NA" if eta is None else f"{eta:.4f}"
            print(
                "  "
                f"{row['parameter']:<22} eta^2={eta_text:<8} "
                f"steps_used={row['n_steps_used']:<3} "
                f"levels={row['n_levels']:<4} "
                f"singleton_frac={row['singleton_level_fraction']}"
            )


# Interpretation note:
# Example CSV row:
# metric=consensus, parameter=num_agents, eta_squared=0.42, sigma2_estimate=0.003
# Read as: at fixed timestep and fixed question, num_agents explains about 42% of
# consensus variance across runs; about 58% remains as residual/unexplained variance.
# sigma2_estimate is an empirical within-configuration noise estimate from repeated
# runs that share the full parameter signature.
# Field definitions used in console/CSV output:
# - levels: number of distinct observed values of a parameter (for the metric slice).
# - singleton_frac: fraction of levels seen exactly once (higher means weaker
#   replication for that parameter effect estimate).
# - steps_used: number of informative timesteps included in ANOVA aggregation
#   (requires at least 2 observations and non-zero variance at the timestep).
if __name__ == "__main__":
    main()
