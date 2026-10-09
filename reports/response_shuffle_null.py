#!/usr/bin/env python3
"""
Response-shuffle null model for the topology paper rebuttal (reviewer "Option A").

Post-hoc permutation test on existing Phase 2 raw run files
(tmp/visualizer_randomized_network_*.json, 156 runs). Does NOT re-run any
simulations. For each run, holds the graph (nodes/edges) fixed and permutes
the mapping of survey-response *trajectories* onto user node names once per
run (same permutation reused across all 11 snapshots, so each agent's own
opinion trajectory is preserved but its spatial position in the graph is
randomized). Recomputes the neighbor-structured behavioral metrics on the
shuffled data using src/simulation_components/metrics.py verbatim, and
compares real vs. null.

Metrics: NASR (mean_neighbor_alignment_shift_rate), final network
assortativity, mean local agreement, cross-cutting edge fraction, mean
same-option exposure share. OSR/NCC/MFR are intentionally excluded: they are
population-level counts invariant to node relabeling, so this shuffle is
uninformative for them.

Usage:
    python reports/response_shuffle_null.py [--n-perm 1000] [--workers 16]
"""

import argparse
import glob
import json
import math
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from simulation_components.metrics import (  # noqa: E402
    compute_echo_chamber_metrics_for_survey,
    compute_herd_effect_metrics,
)
from simulation_components.type_aliases import Thread  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "reports"))
from phase2_analysis import eta_squared, permutation_anova_p  # noqa: E402

PHASE2_GLOB = str(REPO_ROOT / "tmp" / "visualizer_randomized_network_*.json")
OUT_DIR = REPO_ROOT / "analysis_out"
OUT_CSV = OUT_DIR / "response_shuffle_null.csv"

GRAPH_ORDER = [
    "cycle", "forest_fire", "fully_connected", "stochastic_block",
    "random", "powerlaw_cluster", "barabasi_albert",
]

METRICS = ["nasr", "assortativity", "local_agreement", "cross_cutting", "same_option_exposure"]


def load_run_data(f: str) -> Dict[str, Any]:
    with open(f) as fh:
        d = json.load(fh)
    nodes = d["nodes"]
    edges = d["edges"]
    name_to_idx = {n["name"]: i for i, n in enumerate(nodes)}
    social_graph: List[set] = [set() for _ in nodes]
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s is None or t is None:
            continue
        social_graph[s].add(t)
    threads = [Thread(id=t["id"], content=t["messages"]) for t in d.get("threads", [])]
    survey_results = d["survey_results"]
    graph_type = d.get("run_parameters", {}).get("graph_type")
    return {
        "file": f,
        "graph_type": graph_type,
        "visualizer_data": d,
        "social_graph": social_graph,
        "name_to_idx": name_to_idx,
        "threads": threads,
        "survey_results": survey_results,
    }


def compute_metrics(survey_results, visualizer_data, social_graph, name_to_idx, threads):
    herd = compute_herd_effect_metrics(survey_results, visualizer_data, social_graph, name_to_idx)
    nasr = herd.get("mean_neighbor_alignment_shift_rate") if herd.get("status") == "ok" else None

    echo = compute_echo_chamber_metrics_for_survey(survey_results[-1], visualizer_data, threads)
    if echo.get("status") == "ok":
        assort = echo.get("network_assortativity")
        local_agr = echo.get("mean_local_agreement")
        cross = echo.get("cross_cutting_edge_fraction")
        exposure = echo.get("mean_same_option_exposure_share")
    else:
        assort = local_agr = cross = exposure = None

    return {
        "nasr": nasr,
        "assortativity": assort,
        "local_agreement": local_agr,
        "cross_cutting": cross,
        "same_option_exposure": exposure,
    }


def shuffle_survey_results(survey_results, rng):
    all_names = sorted({name for snap in survey_results for name in snap["results"]})
    perm_idx = rng.permutation(len(all_names))
    mapping = {all_names[i]: all_names[perm_idx[i]] for i in range(len(all_names))}
    shuffled = []
    for snap in survey_results:
        new_results = {mapping[name]: val for name, val in snap["results"].items()}
        shuffled.append({"step": snap["step"], "question": snap.get("question"), "results": new_results})
    return shuffled


def process_run(args):
    f, run_idx, n_perm = args
    data = load_run_data(f)
    real = compute_metrics(
        data["survey_results"], data["visualizer_data"], data["social_graph"],
        data["name_to_idx"], data["threads"],
    )

    rng = np.random.default_rng(42 + run_idx)
    null_draws = {m: [] for m in METRICS}
    for _ in range(n_perm):
        shuffled_sr = shuffle_survey_results(data["survey_results"], rng)
        m = compute_metrics(
            shuffled_sr, data["visualizer_data"], data["social_graph"],
            data["name_to_idx"], data["threads"],
        )
        for k in METRICS:
            null_draws[k].append(m[k])

    return {
        "file": f,
        "graph_type": data["graph_type"],
        "real": real,
        "null": null_draws,
    }


def clean(vals):
    return [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]


def mean(vals):
    v = clean(vals)
    return statistics.mean(v) if v else float("nan")


def sd(vals):
    v = clean(vals)
    return statistics.stdev(v) if len(v) > 1 else 0.0


def kendall_tau(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    cli_args = ap.parse_args()

    files = sorted(glob.glob(PHASE2_GLOB))
    print(f"Found {len(files)} Phase 2 raw run files.")

    jobs = [(f, i, cli_args.n_perm) for i, f in enumerate(files)]
    with Pool(cli_args.workers) as pool:
        results = pool.map(process_run, jobs)

    OUT_DIR.mkdir(exist_ok=True)

    # ---- per-run CSV rows ----
    per_run_rows = []
    for r in results:
        row = {"file": r["file"], "graph_type": r["graph_type"], "row_type": "run"}
        for k in METRICS:
            real_v = r["real"][k]
            null_vals = clean(r["null"][k])
            row[f"{k}_real"] = real_v
            row[f"{k}_null_mean"] = mean(null_vals)
            row[f"{k}_null_sd"] = sd(null_vals)
            if null_vals and real_v is not None:
                z = (real_v - mean(null_vals)) / sd(null_vals) if sd(null_vals) > 0 else float("nan")
                ge = sum(1 for v in null_vals if v >= real_v)
                le = sum(1 for v in null_vals if v <= real_v)
                p_two = 2 * min(ge, le) / (len(null_vals) + 1)
                p_two = min(p_two, 1.0)
            else:
                z, p_two = float("nan"), float("nan")
            row[f"{k}_z"] = z
            row[f"{k}_p_perm"] = p_two
        per_run_rows.append(row)

    # ---- per-topology aggregation ----
    by_topo: Dict[str, List[Dict]] = {gt: [] for gt in GRAPH_ORDER}
    for r in results:
        gt = r["graph_type"]
        if gt in by_topo:
            by_topo[gt].append(r)

    topo_rows = []
    topo_summary = {}
    for gt in GRAPH_ORDER:
        runs = by_topo[gt]
        if not runs:
            continue
        row = {"file": "", "graph_type": gt, "row_type": "topology", "n_runs": len(runs)}
        summary_metrics = {}
        for k in METRICS:
            real_vals = [r["real"][k] for r in runs]
            real_clean = clean(real_vals)
            real_topo_mean = mean(real_clean)

            # per-permutation topology-mean null distribution (paired across runs by perm index)
            n_perm = cli_args.n_perm
            null_topo_means = []
            for kperm in range(n_perm):
                vals_k = [r["null"][k][kperm] for r in runs if r["null"][k][kperm] is not None]
                if vals_k:
                    null_topo_means.append(sum(vals_k) / len(vals_k))
            null_mean_of_means = mean(null_topo_means)
            null_sd_of_means = sd(null_topo_means)
            if null_sd_of_means > 0:
                z = (real_topo_mean - null_mean_of_means) / null_sd_of_means
            else:
                z = float("nan")
            ge = sum(1 for v in null_topo_means if v >= real_topo_mean)
            le = sum(1 for v in null_topo_means if v <= real_topo_mean)
            p_two = min(2 * min(ge, le) / (len(null_topo_means) + 1), 1.0) if null_topo_means else float("nan")

            row[f"{k}_real_mean"] = real_topo_mean
            row[f"{k}_real_sd"] = sd(real_clean)
            row[f"{k}_null_mean"] = null_mean_of_means
            row[f"{k}_null_sd"] = null_sd_of_means
            row[f"{k}_z"] = z
            row[f"{k}_p_perm"] = p_two
            summary_metrics[k] = {
                "real_mean": real_topo_mean, "real_sd": sd(real_clean),
                "null_mean": null_mean_of_means, "null_sd": null_sd_of_means,
                "z": z, "p_perm": p_two,
            }
        topo_rows.append(row)
        topo_summary[gt] = summary_metrics

    # ---- cross-topology picture: Kendall tau + eta^2 collapse ----
    tau_lines = []
    eta_lines = []
    for k in METRICS:
        real_groups = [clean([r["real"][k] for r in by_topo[gt]]) for gt in GRAPH_ORDER if by_topo[gt]]
        gt_present = [gt for gt in GRAPH_ORDER if by_topo[gt]]
        real_topo_means = [mean(g) for g in real_groups]

        n_perm = cli_args.n_perm
        eta2_null_draws = []
        null_topo_mean_matrix = []  # [gt][perm] -> mean across runs
        for gt in gt_present:
            runs = by_topo[gt]
            row_means = []
            for kperm in range(n_perm):
                vals_k = clean([r["null"][k][kperm] for r in runs])
                row_means.append(mean(vals_k) if vals_k else float("nan"))
            null_topo_mean_matrix.append(row_means)

        for kperm in range(n_perm):
            groups_k = []
            for gi, gt in enumerate(gt_present):
                runs = by_topo[gt]
                vals_k = clean([r["null"][k][kperm] for r in runs])
                if vals_k:
                    groups_k.append(vals_k)
            if len(groups_k) >= 2:
                eta2_null_draws.append(eta_squared(groups_k))

        real_groups_clean = [g for g in real_groups if g]
        eta2_real = eta_squared(real_groups_clean) if len(real_groups_clean) >= 2 else float("nan")
        p_real = permutation_anova_p(real_groups_clean) if len(real_groups_clean) >= 2 else float("nan")

        # topology-mean null values averaged over all permutations, for tau
        null_topo_means_avg = [mean(row) for row in null_topo_mean_matrix]

        tau = kendall_tau(real_topo_means, null_topo_means_avg)
        eta2_null_draws_clean = clean(eta2_null_draws)

        tau_lines.append((k, tau, real_topo_means, null_topo_means_avg, gt_present))
        eta_lines.append((k, eta2_real, p_real, mean(eta2_null_draws_clean), sd(eta2_null_draws_clean)))

    # ---- write CSV ----
    fieldnames = ["row_type", "file", "graph_type", "n_runs"]
    for k in METRICS:
        fieldnames += [f"{k}_real", f"{k}_null_mean", f"{k}_null_sd", f"{k}_z", f"{k}_p_perm"]
    for k in METRICS:
        fieldnames += [f"{k}_real_mean", f"{k}_real_sd"]

    import csv
    all_rows = per_run_rows + topo_rows
    all_fieldnames = sorted({key for row in all_rows for key in row})
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Wrote {len(all_rows)} rows to {OUT_CSV}")

    # ---- markdown summary ----
    print()
    print("## Response-shuffle null model: per-topology real vs. shuffled null")
    print()
    for k in METRICS:
        print(f"### {k}")
        print()
        print("| topology | n | real mean | null mean ± sd | z | p (perm) |")
        print("|---|---|---|---|---|---|")
        for gt in GRAPH_ORDER:
            if gt not in topo_summary:
                continue
            s = topo_summary[gt][k]
            print(
                f"| {gt} | {len(by_topo[gt])} | {s['real_mean']:.4f} | "
                f"{s['null_mean']:.4f} ± {s['null_sd']:.4f} | {s['z']:.2f} | {s['p_perm']:.4f} |"
            )
        print()

    print("## Cross-topology ordering: Kendall tau (real per-topology means vs. shuffled-null per-topology means)")
    print()
    print("| metric | tau |")
    print("|---|---|")
    for k, tau, _, _, _ in tau_lines:
        print(f"| {k} | {tau:.3f} |")
    print()

    print("## eta^2 collapse under the null (topology explains variance in real vs. shuffled data)")
    print()
    print("| metric | eta^2 real | p (real, anova-perm) | eta^2 null mean ± sd |")
    print("|---|---|---|---|")
    for k, e_real, p_real, e_null_mean, e_null_sd in eta_lines:
        print(f"| {k} | {e_real:.4f} | {p_real:.4f} | {e_null_mean:.4f} ± {e_null_sd:.4f} |")
    print()

    print("### Per-topology means used for NASR Kendall tau")
    print()
    nasr_tau_entry = [t for t in tau_lines if t[0] == "nasr"][0]
    _, tau_nasr, real_means, null_means, gts = nasr_tau_entry
    print("| topology | real NASR mean | null NASR mean |")
    print("|---|---|---|")
    for gt, rm, nm in zip(gts, real_means, null_means):
        print(f"| {gt} | {rm:.4f} | {nm:.4f} |")


if __name__ == "__main__":
    main()
