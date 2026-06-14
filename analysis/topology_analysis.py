#!/usr/bin/env python3
"""First-class topology analysis for E3 (V2_PLAN Priority 4, R2-W1).

For each behavioral metric:
1. eta^2 for topology with a PERMUTATION p-value, shuffling topology labels
   within (model_family x question) strata so the test is exact under the
   stratified design.
2. Per-model robustness: the same test within each model family, BH-FDR
   corrected across models x metrics.
3. Density check: among parameterizable (density-matched) topologies, does the
   empirical mean degree predict the metric? If topology effects survive while
   density does not, the effect is structural rather than density-driven.

Usage:
    python analysis/topology_analysis.py --dataset v2_runs.csv --outdir analysis_out
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

METRICS = [
    "mean_neighbor_alignment_shift_rate",
    "net_consensus_change",
    "mean_opinion_shift_rate",
    "mean_current_majority_follow_rate",
    "delta_assortativity",
    "mean_local_agreement",
    "cross_cutting_edge_fraction",
]

# cycle and fully_connected are intentional structural extremes, not density-matched.
PARAMETERIZABLE = ["random", "powerlaw_cluster", "barabasi_albert", "stochastic_block", "forest_fire"]


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    # Guard against effectively-constant metrics, where float residue would
    # otherwise produce arbitrary ratios.
    if ss_total <= 1e-12 * max(1.0, grand ** 2) * len(values):
        return 0.0
    ss_between = sum(
        len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
        for g in np.unique(groups)
    )
    return float(ss_between / ss_total)


def stratified_permutation_eta2(df, metric, n_perm, rng):
    data = df.dropna(subset=[metric, "graph_type"]).copy()
    if data["graph_type"].nunique() < 2 or len(data) < 10:
        return None
    values = data[metric].to_numpy(float)
    groups = data["graph_type"].to_numpy()
    observed = eta_squared(values, groups)

    strata = data.groupby(["model_family", "question_number"]).indices
    exceed = 0
    for _ in range(n_perm):
        permuted = groups.copy()
        for idx in strata.values():
            idx = np.asarray(idx)
            permuted[idx] = rng.permutation(permuted[idx])
        if eta_squared(values, permuted) >= observed:
            exceed += 1
    p = (exceed + 1) / (n_perm + 1)
    return {
        "n": int(len(data)),
        "eta2": observed,
        "p_perm": float(p),
        "group_means": {
            g: float(values[groups == g].mean()) for g in np.unique(groups)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v2_runs.csv")
    parser.add_argument("--outdir", default="analysis_out")
    parser.add_argument("--experiment", default="e3_topology")
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.dataset)
    df = df[df["experiment"] == args.experiment]
    if not len(df):
        raise SystemExit(f"No runs found for experiment {args.experiment}")
    print(f"E3 runs: {len(df)}; topologies: {sorted(df['graph_type'].dropna().unique())}")

    results = {"overall": {}, "per_model": {}, "density_check": {}}

    # 1. Overall stratified permutation test per metric.
    for metric in METRICS:
        res = stratified_permutation_eta2(df, metric, args.n_perm, rng)
        if res:
            results["overall"][metric] = res
            print(f"{metric:45s} eta2={res['eta2']:.3f} p_perm={res['p_perm']:.4g}")

    # 2. Per-model robustness with BH-FDR across the whole family.
    family = []
    for model in sorted(df["model_family"].dropna().unique()):
        sub = df[df["model_family"] == model]
        for metric in METRICS:
            res = stratified_permutation_eta2(sub, metric, args.n_perm, rng)
            if res:
                family.append({"model_family": model, "metric": metric, **res})
    if family:
        pvals = [f["p_perm"] for f in family]
        reject, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        for f, q, r in zip(family, qvals, reject):
            f["q_bh"] = float(q)
            f["significant_fdr"] = bool(r)
            f.pop("group_means", None)
        results["per_model"] = family

    # 3. Density check among parameterizable topologies.
    sub = df[df["graph_type"].isin(PARAMETERIZABLE)].dropna(subset=["graph_mean_degree"])
    for metric in METRICS:
        d = sub.dropna(subset=[metric])
        if len(d) < 10 or d["graph_mean_degree"].nunique() < 3 or d[metric].nunique() < 2:
            continue
        r, p = stats.pearsonr(d["graph_mean_degree"], d[metric])
        results["density_check"][metric] = {"pearson_r": float(r), "p": float(p), "n": int(len(d))}

    out_path = os.path.join(args.outdir, "topology_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
