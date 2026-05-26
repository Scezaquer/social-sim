#!/usr/bin/env python3
"""
Phase 2 analysis script (EMNLP 2026 paper).

Computes per-topology means + standard deviations and ANOVA effect sizes
for every metric reported in the paper's Phase 2 section. Outputs LaTeX-
ready snippets so every number in the paper can be traced back to this
script.

Inputs: /home/aurelienbk/social-sim/tmp/visualizer_randomized_network_*.json
        (156 Phase 2 runs).

Usage:
    python reports/phase2_analysis.py
"""

import collections
import glob
import json
import math
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE2_GLOB = "/home/aurelienbk/social-sim/tmp/visualizer_randomized_network_*.json"

GRAPH_LABEL = {
    "random": "Random (ER)",
    "powerlaw_cluster": "PL Cluster",
    "barabasi_albert": "Barab\\'asi--Albert",
    "stochastic_block": "Stoch.\\ Block",
    "forest_fire": "Forest Fire",
    "fully_connected": "Fully Connected",
    "cycle": "Cycle",
}

# Plotting/reporting order: Cycle, FF, FC, SBM, ER, PLC, BA (matches paper).
GRAPH_ORDER = [
    "cycle", "forest_fire", "fully_connected", "stochastic_block",
    "random", "powerlaw_cluster", "barabasi_albert",
]

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _clean(vals):
    return [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]

def mean(vals):
    v = _clean(vals)
    return statistics.mean(v) if v else float("nan")

def stdev(vals):
    v = _clean(vals)
    return statistics.stdev(v) if len(v) > 1 else 0.0

def variance(vals):
    v = _clean(vals)
    return statistics.variance(v) if len(v) > 1 else 0.0

def welch_t(a, b) -> Tuple[float, float]:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p

def cohens_d(a, b) -> float:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    pooled = math.sqrt(
        (statistics.variance(a) * (len(a) - 1) + statistics.variance(b) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    return (ma - mb) / pooled if pooled > 0 else float("nan")

def f_stat(groups: List[List[float]]) -> float:
    all_vals = [v for g in groups for v in g]
    if len(all_vals) < 2:
        return float("nan")
    grand = statistics.mean(all_vals)
    k, N = len(groups), len(all_vals)
    if N <= k:
        return float("nan")
    ss_b = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups if g)
    ss_w = sum((v - statistics.mean(g)) ** 2 for g in groups for v in g if g)
    df_b, df_w = k - 1, N - k
    if df_b <= 0 or df_w <= 0 or ss_w == 0:
        return float("nan")
    return (ss_b / df_b) / (ss_w / df_w)

def permutation_anova_p(groups: List[List[float]], n_perm: int = 2000) -> float:
    all_vals = [v for g in groups for v in g]
    obs_f = f_stat(groups)
    if math.isnan(obs_f):
        return float("nan")
    sizes = [len(g) for g in groups]
    count = 0
    for _ in range(n_perm):
        random.shuffle(all_vals)
        pg, idx = [], 0
        for s in sizes:
            pg.append(all_vals[idx:idx + s])
            idx += s
        pf = f_stat(pg)
        if not math.isnan(pf) and pf >= obs_f:
            count += 1
    return (count + 1) / (n_perm + 1)

def eta_squared(groups: List[List[float]]) -> float:
    all_vals = [v for g in groups for v in g]
    if len(all_vals) < 2:
        return float("nan")
    grand = statistics.mean(all_vals)
    ss_t = sum((v - grand) ** 2 for v in all_vals)
    if ss_t == 0:
        return float("nan")
    ss_b = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups if g)
    return ss_b / ss_t

def permutation_two_sample_p(a: List[float], b: List[float], n_perm: int = 2000) -> float:
    """Permutation p-value for a two-sample mean difference (two-sided)."""
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = a + b
    obs = abs(statistics.mean(a) - statistics.mean(b))
    count = 0
    na = len(a)
    for _ in range(n_perm):
        random.shuffle(pooled)
        diff = abs(statistics.mean(pooled[:na]) - statistics.mean(pooled[na:]))
        if diff >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)

def pearson_r(xs, ys) -> Tuple[float, float, int]:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 3:
        return float("nan"), float("nan"), len(pairs)
    x_, y_ = zip(*pairs)
    mx, my = statistics.mean(x_), statistics.mean(y_)
    num = sum((a - mx) * (b - my) for a, b in zip(x_, y_))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x_))
    sy = math.sqrt(sum((b - my) ** 2 for b in y_))
    if sx == 0 or sy == 0:
        return 0.0, 1.0, len(pairs)
    r = num / (sx * sy)
    n = len(pairs)
    tv = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r ** 2))
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(tv) / math.sqrt(2))))
    return r, pv, n

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def nasr_from_herd(herd: dict) -> Optional[float]:
    transitions = herd.get("transitions", [])
    vals = [tr.get("neighbor_alignment_shift_rate") for tr in transitions
            if isinstance(tr, dict) and tr.get("neighbor_alignment_shift_rate") is not None]
    if vals:
        return statistics.mean(vals)
    # Some files store it as a scalar at the top level too.
    return herd.get("mean_neighbor_alignment_shift_rate")

def delta_metric(by_survey: list, key: str) -> Optional[float]:
    if not by_survey:
        return None
    v0 = by_survey[0].get(key)
    v1 = by_survey[-1].get(key)
    if v0 is None or v1 is None:
        return None
    return v1 - v0

def load_runs() -> List[Dict[str, Any]]:
    runs = []
    files = sorted(glob.glob(PHASE2_GLOB))
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        rp = d.get("run_parameters", {})
        bm_raw = d.get("behavioral_metrics", {})
        herd = bm_raw.get("herd_effect_metrics", {})
        echo = bm_raw.get("echo_chamber_metrics", {})
        bert = d.get("bert_real_vs_llm_classifier", {})
        by_survey = echo.get("by_survey", [])
        nodes = d.get("nodes", [])
        edges = d.get("edges", [])
        n_nodes = len(nodes)
        n_edges = len(edges)
        runs.append({
            "file": f,
            "graph_type": rp.get("graph_type"),
            "homophily": rp.get("homophily"),
            "question": rp.get("question_number"),
            "num_agents": rp.get("num_agents"),
            "num_news": rp.get("num_news_agents"),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "mean_degree": (2 * n_edges / n_nodes) if n_nodes > 0 else float("nan"),
            # Behavioral
            "initial_consensus": herd.get("initial_consensus"),
            "final_consensus": herd.get("final_consensus"),
            "ncc": herd.get("net_consensus_change"),
            "osr": herd.get("mean_opinion_shift_rate"),
            "mfr": herd.get("mean_current_majority_follow_rate"),
            "nasr": nasr_from_herd(herd),
            # Echo (final snapshot)
            "assortativity": echo.get("network_assortativity"),
            "local_agreement": echo.get("mean_local_agreement"),
            "cross_cutting": echo.get("cross_cutting_edge_fraction"),
            "same_option_exposure": echo.get("mean_same_option_exposure_share"),
            "exposure_diversity": echo.get("mean_exposure_diversity"),
            # Echo (initial snapshot)
            "init_assortativity": (by_survey[0].get("network_assortativity") if by_survey else None),
            "init_local_agreement": (by_survey[0].get("mean_local_agreement") if by_survey else None),
            "init_cross_cutting": (by_survey[0].get("cross_cutting_edge_fraction") if by_survey else None),
            # Echo deltas (last - first survey snapshot)
            "d_assortativity": delta_metric(by_survey, "network_assortativity"),
            "d_local_agreement": delta_metric(by_survey, "mean_local_agreement"),
            "d_cross_cutting": delta_metric(by_survey, "cross_cutting_edge_fraction"),
            "d_same_option_exposure": delta_metric(by_survey, "mean_same_option_exposure_share"),
            # BERT
            "bert_accuracy": bert.get("accuracy"),
        })
    return runs

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def group_by_graph(runs: List[Dict], key: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = collections.defaultdict(list)
    for r in runs:
        gt = r.get("graph_type")
        v = r.get(key)
        if gt and v is not None and not (isinstance(v, float) and math.isnan(v)):
            out[gt].append(v)
    return dict(out)

def topology_anova(runs: List[Dict], key: str, n_perm: int = 2000):
    grp = group_by_graph(runs, key)
    g_list = [g for g in grp.values() if len(g) >= 2]
    if len(g_list) < 2:
        return float("nan"), float("nan"), float("nan"), grp
    F = f_stat(g_list)
    p = permutation_anova_p(g_list, n_perm=n_perm)
    e2 = eta_squared(g_list)
    return F, p, e2, grp

def homophily_test(runs: List[Dict], key: str, n_perm: int = 2000):
    ht = [r.get(key) for r in runs if r.get("homophily") is True]
    hf = [r.get(key) for r in runs if r.get("homophily") is False]
    t, _ = welch_t(ht, hf)
    d = cohens_d(ht, hf)
    p_perm = permutation_two_sample_p(ht, hf, n_perm=n_perm)
    return t, d, p_perm, mean(ht), mean(hf), len(_clean(ht)), len(_clean(hf))

def fmt(v, spec=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "---"
    return format(v, spec)

def fmtp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "---"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} Phase 2 runs")
    print()
    # Sanity checks
    gt_counts = collections.Counter(r["graph_type"] for r in runs)
    hm_counts = collections.Counter(r["homophily"] for r in runs)
    print("Counts by graph_type:")
    for gt in GRAPH_ORDER:
        print(f"  {gt}: {gt_counts.get(gt, 0)}")
    print(f"Homophily: True={hm_counts[True]}, False={hm_counts[False]}")
    print()

    # ---------------------------------------------------------------------
    # tab:phase2_behavioral — Graph-type effect + Homophily effect, both
    # reported as (eta^2 or d) & p_perm.
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("TABLE: phase2_behavioral — Behavioural metrics, both effects with p_perm")
    print("=" * 78)
    print(f"{'Metric':<32} {'eta^2 graph':>12} {'p_perm':>10} {'d hom':>8} {'p_perm':>10}")
    behav_metrics = [
        ("ncc", "Net Consensus Change"),
        ("osr", "Opinion Shift Rate"),
        ("mfr", "Majority Follow Rate"),
        ("nasr", "Neighbor Alignment Shift Rate"),
    ]
    behav_rows = []
    for key, label in behav_metrics:
        F, p_g, e2, _ = topology_anova(runs, key)
        t, d, p_h, mt, mf, nt, nf = homophily_test(runs, key)
        print(f"  {label:<30} {fmt(e2):>12} {fmtp(p_g):>10} {fmt(d):>8} {fmtp(p_h):>10}")
        behav_rows.append((label, F, e2, p_g, t, d, p_h, mt, mf, nt, nf))
    print()

    # ---------------------------------------------------------------------
    # tab:phase2_echo — Echo-chamber metrics (final snapshot)
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("TABLE: phase2_echo — Echo-chamber final snapshots")
    print("=" * 78)
    print(f"{'Metric':<32} {'eta^2 graph':>12} {'p_perm':>10} {'d hom':>8} {'p_perm':>10}")
    echo_metrics = [
        ("assortativity", "Final assortativity"),
        ("local_agreement", "Final local agreement"),
        ("cross_cutting", "Final cross-cutting"),
        ("exposure_diversity", "Exposure diversity"),
    ]
    echo_rows = []
    for key, label in echo_metrics:
        F, p_g, e2, _ = topology_anova(runs, key)
        t, d, p_h, mt, mf, nt, nf = homophily_test(runs, key)
        print(f"  {label:<30} {fmt(e2):>12} {fmtp(p_g):>10} {fmt(d):>8} {fmtp(p_h):>10}")
        echo_rows.append((label, F, e2, p_g, t, d, p_h, mt, mf, nt, nf))
    print()

    # Echo delta metrics (last - first)
    print("=" * 78)
    print("Echo-chamber DELTAs (last - first survey snapshot)")
    print("=" * 78)
    print(f"{'Metric':<32} {'eta^2 graph':>12} {'p_perm':>10} {'d hom':>8} {'p_perm':>10}")
    delta_metrics = [
        ("d_assortativity", "Δ assortativity"),
        ("d_local_agreement", "Δ local agreement"),
        ("d_cross_cutting", "Δ cross-cutting"),
    ]
    delta_rows = []
    for key, label in delta_metrics:
        F, p_g, e2, _ = topology_anova(runs, key)
        t, d, p_h, mt, mf, nt, nf = homophily_test(runs, key)
        print(f"  {label:<30} {fmt(e2):>12} {fmtp(p_g):>10} {fmt(d):>8} {fmtp(p_h):>10}")
        delta_rows.append((label, F, e2, p_g, t, d, p_h, mt, mf, nt, nf))
    print()

    # ---------------------------------------------------------------------
    # tab:topology_attribution — Per-topology mean ± SD for ALL metrics,
    # with Random (ER) separated out (default).
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("PER-TOPOLOGY MEANS ± STD")
    print("=" * 78)
    all_keys = [
        ("ncc", "NCC"),
        ("nasr", "NASR"),
        ("osr", "OSR"),
        ("mfr", "MFR"),
        ("initial_consensus", "Init Cons."),
        ("final_consensus", "Final Cons."),
        ("assortativity", "Final Asst."),
        ("local_agreement", "Local Agr."),
        ("cross_cutting", "Cross-cut."),
        ("exposure_diversity", "Exp. Div."),
        ("same_option_exposure", "Same Opt. Exp."),
        ("init_assortativity", "Init Asst."),
        ("init_local_agreement", "Init Loc. Agr."),
        ("init_cross_cutting", "Init Cross-cut."),
        ("d_assortativity", "Δ Asst."),
        ("d_local_agreement", "Δ Loc. Agr."),
        ("d_cross_cutting", "Δ Cross-cut."),
        ("d_same_option_exposure", "Δ Same Opt. Exp."),
    ]
    # Header
    hdr = f"{'Topology':<22} {'n':>4}"
    for _, lab in all_keys:
        hdr += f" {lab:>18}"
    print(hdr)
    for gt in GRAPH_ORDER:
        gruns = [r for r in runs if r["graph_type"] == gt]
        row = f"{GRAPH_LABEL.get(gt, gt):<22} {len(gruns):>4}"
        for key, _ in all_keys:
            vals = [r.get(key) for r in gruns]
            m = mean(vals)
            s = stdev(vals)
            row += f" {fmt(m):>8}±{fmt(s):>8}"
        print(row)
    print()

    # ---------------------------------------------------------------------
    # Cross-domain correlations (for §6 of paper)
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("CROSS-DOMAIN CORRELATIONS")
    print("=" * 78)
    # (a) Δ local agreement vs NCC
    la_d = [r["d_local_agreement"] for r in runs]
    ncc = [r["ncc"] for r in runs]
    r_la_ncc, p_la_ncc, n_la_ncc = pearson_r(la_d, ncc)
    print(f"Δ local agreement vs NCC: r={fmt(r_la_ncc)} p={fmtp(p_la_ncc)} n={n_la_ncc}")

    # (b) Δ assortativity vs NCC
    as_d = [r["d_assortativity"] for r in runs]
    r_as_ncc, p_as_ncc, n_as_ncc = pearson_r(as_d, ncc)
    print(f"Δ assortativity vs NCC: r={fmt(r_as_ncc)} p={fmtp(p_as_ncc)} n={n_as_ncc}")

    # (c) NASR vs OSR
    nasr_v = [r["nasr"] for r in runs]
    osr_v = [r["osr"] for r in runs]
    r_nasr_osr, p_nasr_osr, n_nasr_osr = pearson_r(nasr_v, osr_v)
    print(f"NASR vs OSR: r={fmt(r_nasr_osr)} p={fmtp(p_nasr_osr)} n={n_nasr_osr}")
    print()

    # ---------------------------------------------------------------------
    # BERT realism in Phase 2
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("BERT REALISM (Phase 2)")
    print("=" * 78)
    F_b, p_b, e2_b, grp_b = topology_anova(runs, "bert_accuracy")
    ba_all = [r["bert_accuracy"] for r in runs]
    print(f"Overall BERT acc: mean={mean(ba_all):.4f} ± {stdev(ba_all):.4f} (n={len(_clean(ba_all))})")
    print(f"BERT topology ANOVA: F={fmt(F_b)} eta^2={fmt(e2_b)} p_perm={fmtp(p_b)}")
    for gt in GRAPH_ORDER:
        vals = grp_b.get(gt, [])
        if vals:
            print(f"  {GRAPH_LABEL.get(gt, gt):<22} n={len(vals)} mean={mean(vals):.4f} ± {stdev(vals):.4f}")
    print()

    # ---------------------------------------------------------------------
    # LaTeX-ready per-topology table (a compact version for the main body)
    # ---------------------------------------------------------------------
    print("=" * 78)
    print("LATEX: compact tab:topology_attribution (NCC, NASR, OSR, MFR, ΔAsst., ΔLocAgr.)")
    print("=" * 78)
    sub_keys = [
        ("ncc", "NCC"),
        ("nasr", "NASR"),
        ("osr", "OSR"),
        ("mfr", "MFR"),
        ("d_assortativity", "$\\Delta$Asst."),
        ("d_local_agreement", "$\\Delta$LocAgr."),
    ]
    # Order: Cycle, FF, FC, SBM, PLC, BA, then ER (default) separately
    primary = ["cycle", "forest_fire", "fully_connected", "stochastic_block",
               "powerlaw_cluster", "barabasi_albert"]
    default = "random"
    for gt in primary + [default]:
        gruns = [r for r in runs if r["graph_type"] == gt]
        row = f"{GRAPH_LABEL.get(gt, gt):<22} (n={len(gruns):>3})"
        for key, lab in sub_keys:
            vals = [r.get(key) for r in gruns]
            row += f" & {mean(vals):.3f}$\\pm${stdev(vals):.3f}"
        row += " \\\\"
        print(row)
        if gt == "barabasi_albert":
            print("  \\midrule")

if __name__ == "__main__":
    main()
