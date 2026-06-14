#!/usr/bin/env python3
"""Measurement-validity analysis (E4 + dual-order surveys), per V2_PLAN Priority 3.

Answers, with new data, R4-W2/W4 and R3's prompt-sensitivity concern:
1. Noise floors: opinion-shift rate under stimulus_mode 'none' (self-feedback
   only; expected exactly 0 when ctx=off) and 'scrambled' (socially meaningless
   context) vs matched real runs (E1, N=256). The socially-driven component is
   OSR_normal - OSR_scrambled, reported per model x ctx with Welch t, Cohen's d,
   Mann-Whitney and BH-FDR across the family.
2. Order effects: distribution of per-run order-consistency rates (from the
   dual-order surveys present in every V2 run).
3. Margin-stratified flips: P(answer flips by next survey | log-prob margin
   bin), computed from raw visualizer survey details. Low-margin flips are
   response noise; high-margin flips indicate genuine context-driven change.

Usage:
    python analysis/noise_floor_analysis.py --dataset v2_runs.csv \
        --visualizer-glob 'visualizer_v2_*.json' --outdir analysis_out
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

MARGIN_BINS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf]


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def noise_floor_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model_family, ctx): normal-vs-scrambled and normal-vs-none OSR."""
    real = df[(df["stimulus_mode"] == "normal") & (df["num_agents"] == 256)]
    rows = []
    for baseline in ["scrambled", "none"]:
        base = df[df["stimulus_mode"] == baseline]
        for model in sorted(base["model_family"].dropna().unique()):
            for ctx in [True, False]:
                a = real[(real["model_family"] == model) & (real["add_survey_to_context"] == ctx)]["mean_opinion_shift_rate"].dropna()
                b = base[(base["model_family"] == model) & (base["add_survey_to_context"] == ctx)]["mean_opinion_shift_rate"].dropna()
                if len(a) < 3 or len(b) < 2:
                    continue
                t, p = stats.ttest_ind(a, b, equal_var=False)
                try:
                    u_p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
                except ValueError:
                    u_p = float("nan")
                rows.append({
                    "baseline": baseline,
                    "model_family": model,
                    "ctx": ctx,
                    "n_normal": len(a),
                    "n_baseline": len(b),
                    "osr_normal": float(a.mean()),
                    "osr_baseline": float(b.mean()),
                    "socially_driven_excess": float(a.mean() - b.mean()),
                    "welch_t": float(t),
                    "p_welch": float(p),
                    "p_mannwhitney": float(u_p),
                    "cohens_d": cohens_d(a, b),
                })
    out = pd.DataFrame(rows)
    if len(out):
        out["q_bh"] = multipletests(out["p_welch"], method="fdr_bh")[1]
    return out


def margin_flip_curve(visualizer_paths: list[str]) -> pd.DataFrame:
    """P(flip at next survey | margin bin at current survey), pooled per stimulus mode."""
    counts: dict[tuple, list[int]] = {}
    for path in visualizer_paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        params = data.get("run_parameters") or {}
        mode = params.get("stimulus_mode", "normal") or "normal"
        surveys = data.get("survey_results") or []
        for prev, nxt in zip(surveys, surveys[1:]):
            prev_details = prev.get("details") or {}
            nxt_results = nxt.get("results") or {}
            for name, det in prev_details.items():
                if not isinstance(det, dict) or det.get("margin") is None or name not in nxt_results:
                    continue
                margin = abs(float(det["margin"]))
                flipped = nxt_results[name] != det.get("choice")
                bin_idx = int(np.digitize(margin, MARGIN_BINS) - 1)
                key = (mode, bin_idx)
                counts.setdefault(key, [0, 0])
                counts[key][0] += int(flipped)
                counts[key][1] += 1

    rows = []
    for (mode, bin_idx), (flips, total) in sorted(counts.items()):
        lo = MARGIN_BINS[bin_idx]
        hi = MARGIN_BINS[bin_idx + 1] if bin_idx + 1 < len(MARGIN_BINS) else np.inf
        ci_lo, ci_hi = stats.beta.interval(0.95, flips + 0.5, total - flips + 0.5)  # Jeffreys
        rows.append({
            "stimulus_mode": mode,
            "margin_bin": f"[{lo}, {hi})",
            "n": total,
            "flip_rate": flips / total,
            "flip_rate_ci95": [float(ci_lo), float(ci_hi)],
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v2_runs.csv")
    parser.add_argument("--visualizer-glob", default="visualizer_v2_*.json")
    parser.add_argument("--outdir", default="analysis_out")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.dataset)

    floors = noise_floor_tests(df)
    floors_path = os.path.join(args.outdir, "noise_floor_tests.csv")
    floors.to_csv(floors_path, index=False)
    print(f"Wrote {floors_path} ({len(floors)} contrasts)")
    if len(floors):
        print(floors.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    consistency = df[df["order_consistency_rate"].notna()].groupby(
        ["model_family", "lora_finetuned"]
    )["order_consistency_rate"].agg(["mean", "std", "min", "count"])
    cons_path = os.path.join(args.outdir, "order_consistency.csv")
    consistency.to_csv(cons_path)
    print(f"\nOrder-consistency by model (wrote {cons_path}):")
    print(consistency.to_string(float_format=lambda v: f"{v:.4f}"))

    curve = margin_flip_curve(sorted(glob.glob(args.visualizer_glob)))
    curve_path = os.path.join(args.outdir, "margin_flip_curve.csv")
    curve.to_csv(curve_path, index=False)
    print(f"\nMargin-stratified flip rates (wrote {curve_path}):")
    if len(curve):
        print(curve.to_string(index=False))


if __name__ == "__main__":
    main()
