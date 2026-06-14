#!/usr/bin/env python3
"""Builds the tidy run-level dataset for all V2 analyses.

One row per simulation run, parsed from visualizer_v2_*.json:
- every design factor from run_parameters (incl. seed, stimulus mode, empirical
  graph mean degree);
- behavioral metrics (herd + echo chamber, incl. delta assortativity from the
  first to the last survey);
- measurement-validity columns computed from the dual-order survey details
  (mean order-consistency, mean |log-prob margin|, low-margin share);
- per-run BERT detector accuracy if eval_detector.py has annotated the file.

Usage:
    python analysis/build_dataset.py --input-glob 'visualizer_v2_*.json' --output v2_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

FACTOR_FIELDS = [
    "job_id",
    "array_id",
    "seed",
    "base_model",
    "base_only",
    "num_agents",
    "num_news_agents",
    "graph_type",
    "graph_mean_degree",
    "homophily",
    "question_number",
    "add_survey_to_context",
    "proportions_option",
    "activity_exponent",
    "stimulus_mode",
    "survey_order_mode",
    "max_steps",
    "survey_interval",
]

HERD_FIELDS = [
    "mean_opinion_shift_rate",
    "mean_current_majority_follow_rate",
    "mean_consensus_gain",
    "mean_neighbor_alignment_shift_rate",
    "initial_consensus",
    "final_consensus",
    "net_consensus_change",
    "initial_diversity",
    "final_diversity",
]

ECHO_FIELDS = [
    "network_assortativity",
    "mean_local_agreement",
    "cross_cutting_edge_fraction",
    "mean_same_option_exposure_share",
    "mean_exposure_diversity",
]

LOW_MARGIN_THRESHOLD = 1.0  # nats; |margin| below this = low-confidence answer


def model_label(params: dict) -> str:
    base = str(params.get("base_model", "unknown")).split("/")[-1]
    if params.get("base_only") or not params.get("loras_path"):
        return f"{base}-base"
    return f"{base}-BluePrint"


def survey_validity_columns(survey_results: list[dict]) -> dict:
    """Order-consistency and margin statistics across all surveys of a run."""
    consistent_flags = []
    margins = []
    flip_margins = []  # margins of answers that later changed
    per_agent_prev = {}
    flips_low_margin = 0
    flips_total = 0

    for survey in survey_results:
        details = survey.get("details") or {}
        results = survey.get("results") or {}
        for name, det in details.items():
            if isinstance(det, dict):
                if det.get("order_consistent") is not None:
                    consistent_flags.append(bool(det["order_consistent"]))
                if det.get("margin") is not None:
                    margins.append(abs(float(det["margin"])))
        for name, choice in results.items():
            prev = per_agent_prev.get(name)
            if prev is not None and choice != prev["choice"]:
                flips_total += 1
                if prev["margin"] is not None:
                    flip_margins.append(abs(prev["margin"]))
                    if abs(prev["margin"]) < LOW_MARGIN_THRESHOLD:
                        flips_low_margin += 1
            det = details.get(name) if isinstance(details.get(name), dict) else {}
            per_agent_prev[name] = {
                "choice": choice,
                "margin": float(det["margin"]) if det.get("margin") is not None else None,
            }

    out = {
        "order_consistency_rate": float(np.mean(consistent_flags)) if consistent_flags else None,
        "mean_abs_margin": float(np.mean(margins)) if margins else None,
        "flip_count": flips_total,
        "low_margin_flip_share": (flips_low_margin / flips_total) if flips_total else None,
        "mean_abs_margin_before_flip": float(np.mean(flip_margins)) if flip_margins else None,
    }
    return out


def delta_assortativity(echo_metrics: dict) -> float | None:
    by_survey = echo_metrics.get("by_survey") or []
    values = [
        s.get("network_assortativity")
        for s in by_survey
        if isinstance(s, dict) and s.get("network_assortativity") is not None
    ]
    if len(values) < 2:
        return None
    return float(values[-1]) - float(values[0])


def parse_run(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Skipping {path}: {e}", file=sys.stderr)
        return None

    params = data.get("run_parameters") or {}
    if not params:
        print(f"Skipping {path}: no run_parameters", file=sys.stderr)
        return None

    row = {"file": os.path.basename(path)}
    # visualizer_v2_<experiment>_<runid>.json
    stem = os.path.basename(path).replace("visualizer_v2_", "").replace(".json", "")
    if "_" in stem:
        row["experiment"], _, run_id = stem.rpartition("_")
        row["run_id"] = int(run_id) if run_id.isdigit() else None
    else:
        row["experiment"], row["run_id"] = None, None
    for field in FACTOR_FIELDS:
        row[field] = params.get(field)
    row["model"] = model_label(params)
    row["model_family"] = str(params.get("base_model", "unknown")).split("/")[-1]
    row["lora_finetuned"] = not (params.get("base_only") or not params.get("loras_path"))

    behavioral = data.get("behavioral_metrics") or {}
    herd = behavioral.get("herd_effect_metrics") or {}
    echo = behavioral.get("echo_chamber_metrics") or {}
    for field in HERD_FIELDS:
        row[field] = herd.get(field)
    for field in ECHO_FIELDS:
        row[field] = echo.get(field)
    row["delta_assortativity"] = delta_assortativity(echo)

    row.update(survey_validity_columns(data.get("survey_results") or []))

    bert = data.get("bert_real_vs_llm_classifier")
    if isinstance(bert, dict):
        row["bert_accuracy"] = bert.get("accuracy")
        row["bert_auc"] = bert.get("auc")
    else:
        row["bert_accuracy"] = bert if isinstance(bert, (int, float)) else None
        row["bert_auc"] = None

    n_surveys = len(data.get("survey_results") or [])
    row["survey_count"] = n_surveys
    expected = (int(params.get("max_steps") or 0) // int(params.get("survey_interval") or 1)) + 1
    row["run_complete"] = n_surveys >= expected
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default="visualizer_v2_*.json")
    parser.add_argument("--output", default="v2_runs.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        print(f"No files matched {args.input_glob}", file=sys.stderr)
        sys.exit(1)

    rows = [r for r in (parse_run(p) for p in paths) if r is not None]
    if not rows:
        print("No parsable runs.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    complete = sum(1 for r in rows if r["run_complete"])
    print(f"Wrote {len(rows)} runs ({complete} complete) -> {args.output}")


if __name__ == "__main__":
    main()
