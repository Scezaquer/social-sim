#!/usr/bin/env python3
"""
Reviewer response analysis – EMNLP 2026.

Two-pass design:
  Pass 1: run all analyses, register every hypothesis test in ALL_TESTS.
  Pass 2: apply BH FDR across the full battery, then emit the report.

Usage:
    python reports/reviewer_response_analysis.py
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

GRAPH_LABEL = {
    "random": "Random (ER)",
    "powerlaw_cluster": "PL Cluster",
    "barabasi_albert": "Barabási–Albert",
    "stochastic_block": "Stoch. Block",
    "forest_fire": "Forest Fire",
    "fully_connected": "Fully Connected",
    "cycle": "Cycle",
}

GRAPH_ORDER = [
    "cycle", "forest_fire", "fully_connected", "stochastic_block",
    "random", "powerlaw_cluster", "barabasi_albert",
]

MODEL_LABEL = {
    "marcelbinz/Llama-3.1-Minitaur-8B": "minitaur",
    "meta-llama/Llama-3.1-8B": "llama3.1",
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "google/gemma-3-4b-pt": "gemma",
}

# ---------------------------------------------------------------------------
# Central test registry
# ---------------------------------------------------------------------------

# Each entry: { "label": str, "p": float, "extra": dict }
ALL_TESTS: List[Dict] = []

def register(label: str, p: float, **extra) -> str:
    """Register a test and return its label (for later q-value lookup)."""
    ALL_TESTS.append({"label": label, "p": p, **extra})
    return label


def bh_correct(tests: List[Dict], alpha: float = 0.05) -> Dict[str, float]:
    """Apply Benjamini-Hochberg FDR correction. Returns {label: q_value}."""
    valid = [(t["p"], t["label"]) for t in tests if not math.isnan(t["p"])]
    valid.sort(key=lambda x: x[0])
    m = len(valid)
    q_vals: Dict[str, float] = {}
    # Compute q-values as BH-adjusted p: q_(i) = min_{j>=i} (m/j * p_(j))
    min_so_far = 1.0
    for rank in range(m, 0, -1):
        p_i, lbl = valid[rank - 1]
        q_i = min(1.0, (m / rank) * p_i)
        min_so_far = min(min_so_far, q_i)
        q_vals[lbl] = min_so_far
    # NaN tests get NaN
    for t in tests:
        if math.isnan(t["p"]):
            q_vals[t["label"]] = float("nan")
    return q_vals

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _clean(vals):
    return [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]

def mean(vals): v = _clean(vals); return statistics.mean(v) if v else float("nan")
def stdev(vals): v = _clean(vals); return statistics.stdev(v) if len(v) > 1 else 0.0
def n_valid(vals): return len(_clean(vals))

def welch_t(a, b) -> Tuple[float, float]:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2: return float("nan"), float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0: return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p

def cohens_d(a, b) -> float:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    pooled = math.sqrt(
        (statistics.variance(a) * (len(a) - 1) + statistics.variance(b) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    return (ma - mb) / pooled if pooled > 0 else float("nan")

def pearson_r(xs, ys) -> Tuple[float, float, int]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 3: return float("nan"), float("nan"), len(pairs)
    x_, y_ = zip(*pairs)
    mx, my = statistics.mean(x_), statistics.mean(y_)
    num = sum((a - mx) * (b - my) for a, b in zip(x_, y_))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x_))
    sy = math.sqrt(sum((b - my) ** 2 for b in y_))
    if sx == 0 or sy == 0: return 0.0, 1.0, len(pairs)
    r = num / (sx * sy)
    n = len(pairs)
    tv = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r ** 2))
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(tv) / math.sqrt(2))))
    return r, pv, n

def f_stat(groups: List[List[float]]) -> float:
    all_vals = [v for g in groups for v in g]
    if len(all_vals) < 2: return float("nan")
    grand = statistics.mean(all_vals)
    k, N = len(groups), len(all_vals)
    if N <= k: return float("nan")
    ss_b = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups if g)
    ss_w = sum((v - statistics.mean(g)) ** 2 for g in groups for v in g if g)
    df_b, df_w = k - 1, N - k
    if df_b <= 0 or df_w <= 0 or ss_w == 0: return float("nan")
    return (ss_b / df_b) / (ss_w / df_w)

def permutation_anova_p(groups: List[List[float]], n_perm: int = 2000) -> float:
    all_vals = [v for g in groups for v in g]
    obs_f = f_stat(groups)
    if math.isnan(obs_f): return float("nan")
    sizes = [len(g) for g in groups]
    count = 0
    for _ in range(n_perm):
        random.shuffle(all_vals)
        pg, idx = [], 0
        for s in sizes:
            pg.append(all_vals[idx:idx + s]); idx += s
        pf = f_stat(pg)
        if not math.isnan(pf) and pf >= obs_f: count += 1
    return (count + 1) / (n_perm + 1)

def eta_squared(groups: List[List[float]]) -> float:
    all_vals = [v for g in groups for v in g]
    if len(all_vals) < 2: return float("nan")
    grand = statistics.mean(all_vals)
    ss_t = sum((v - grand) ** 2 for v in all_vals)
    if ss_t == 0: return float("nan")
    ss_b = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups if g)
    return ss_b / ss_t

def kendall_tau(x: List[float], y: List[float]) -> Tuple[float, float]:
    n = len(x)
    if n < 3: return float("nan"), float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx * dy > 0: conc += 1
            elif dx * dy < 0: disc += 1
    tau = (conc - disc) / (n * (n - 1) / 2)
    se = math.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    z = tau / se if se > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return tau, p

def spearman_r(ranks_a, ranks_b) -> float:
    n = len(ranks_a)
    if n < 3: return float("nan")
    d2 = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - 6 * d2 / (n * (n * n - 1))

def topology_anova(runs: List[Dict], metric: str,
                   extra_filter=None, n_perm: int = 2000):
    """Return (F, p_perm, eta2, groups_dict)."""
    groups: Dict[str, List[float]] = collections.defaultdict(list)
    for r in runs:
        if extra_filter and not extra_filter(r): continue
        v = r.get(metric)
        gt = r.get("graph_type")
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and gt:
            groups[gt].append(v)
    g_list = [g for g in groups.values() if len(g) >= 2]
    if len(g_list) < 2:
        return float("nan"), float("nan"), float("nan"), dict(groups)
    F = f_stat(g_list)
    p = permutation_anova_p(g_list, n_perm=n_perm)
    eta2 = eta_squared(g_list)
    return F, p, eta2, dict(groups)

def fmt(v, spec=".3f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "—"
    return format(v, spec)

def fmtp(p: float) -> str:
    if math.isnan(p): return "—"
    if p < 0.001: return "< 0.001"
    return f"{p:.3f}"

def sig(p: float) -> str:
    if math.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "(ns)"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def norm_model(base_model: str) -> str:
    for key, name in MODEL_LABEL.items():
        if key in base_model: return name
    return f"unknown:{base_model[:40]}"

def nasr_from_herd(herd: dict) -> Optional[float]:
    transitions = herd.get("transitions", [])
    vals = [tr.get("neighbor_alignment_shift_rate") for tr in transitions
            if isinstance(tr, dict) and tr.get("neighbor_alignment_shift_rate") is not None]
    return statistics.mean(vals) if vals else None

def delta_echo(by_survey: list, key: str) -> Optional[float]:
    if not by_survey: return None
    v0 = by_survey[0].get(key); v1 = by_survey[-1].get(key)
    return (v1 - v0) if v0 is not None and v1 is not None else None

def load_runs(patterns: List[str]) -> List[Dict[str, Any]]:
    runs = []
    for pattern in patterns:
        for f in sorted(glob.glob(pattern)):
            try:
                with open(f) as fh: d = json.load(fh)
            except Exception: continue
            rp = d.get("run_parameters", {})
            bm_raw = d.get("behavioral_metrics", {})
            herd = bm_raw.get("herd_effect_metrics", {})
            echo = bm_raw.get("echo_chamber_metrics", {})
            bert = d.get("bert_real_vs_llm_classifier", {})
            nodes = d.get("nodes", []); edges = d.get("edges", [])
            by_survey = echo.get("by_survey", [])
            n_nodes = len(nodes); n_edges = len(edges)
            runs.append({
                "file": f,
                "model": norm_model(rp.get("base_model", "")),
                "graph_type": rp.get("graph_type"),
                "homophily": rp.get("homophily"),
                "question": rp.get("question_number"),
                "num_news": rp.get("num_news_agents"),
                "num_agents": rp.get("num_agents"),
                "n_nodes": n_nodes, "n_edges": n_edges,
                "density": n_edges / n_nodes if n_nodes > 0 else float("nan"),
                "initial_consensus": herd.get("initial_consensus"),
                "final_consensus": herd.get("final_consensus"),
                "ncc": herd.get("net_consensus_change"),
                "osr": herd.get("mean_opinion_shift_rate"),
                "mfr": herd.get("mean_current_majority_follow_rate"),
                "nasr": nasr_from_herd(herd),
                "assortativity": echo.get("network_assortativity"),
                "local_agreement": echo.get("mean_local_agreement"),
                "cross_cutting": echo.get("cross_cutting_edge_fraction"),
                "d_assortativity": delta_echo(by_survey, "network_assortativity"),
                "d_local_agreement": delta_echo(by_survey, "mean_local_agreement"),
                "d_cross_cutting": delta_echo(by_survey, "cross_cutting_edge_fraction"),
                "bert_accuracy": bert.get("accuracy"),
            })
    return runs

# ---------------------------------------------------------------------------
# PASS 1: Run all analyses, register all tests
# ---------------------------------------------------------------------------

runs_model    = load_runs(["visualizer_randomized_emnlp2026model_*.json"])
runs_nonews   = load_runs(["visualizer_randomized_emnlp2026nonews_*.json"])
runs_question = load_runs(["visualizer_randomized_emnlp2026question_*.json"])
runs_baseline = load_runs(["emnlp_visualizer_baseline/*.json"])

METRICS = [("nasr","NASR"),("ncc","NCC"),("osr","OSR"),("mfr","MFR")]

# ---- Comment 1: model generalizability ----
model_results = {}
for model in ["gemma","llama3.1","qwen"]:
    mr = [r for r in runs_model if r["model"] == model]
    model_results[model] = {}
    for metric, label in METRICS:
        F, p, eta2, grp = topology_anova(mr, metric)
        tid = f"c1_{model}_{metric}_anova"
        register(tid, p, F=F, eta2=eta2)
        model_results[model][metric] = {"F":F,"p":p,"eta2":eta2,"groups":grp,"tid":tid,"n":len(mr)}

# Rank correlations for NASR across model pairs
model_nasr_means = {}
for model in ["gemma","llama3.1","qwen"]:
    g = model_results[model]["nasr"]["groups"]
    model_nasr_means[model] = {gt: mean(g.get(gt,[])) for gt in GRAPH_ORDER}

phase2_nasr = {"cycle":0.121,"forest_fire":0.115,"fully_connected":0.113,
               "stochastic_block":0.098,"random":0.095,"powerlaw_cluster":0.091,"barabasi_albert":0.068}

rank_corr_results = {}
for m1, m2 in [("gemma","llama3.1"),("gemma","qwen"),("llama3.1","qwen")]:
    common = [gt for gt in GRAPH_ORDER
              if not math.isnan(model_nasr_means[m1].get(gt,float("nan")))
              and not math.isnan(model_nasr_means[m2].get(gt,float("nan")))]
    x = [model_nasr_means[m1][gt] for gt in common]
    y = [model_nasr_means[m2][gt] for gt in common]
    tau, p_tau = kendall_tau(x, y)
    rx = [sorted(x).index(v)+1 for v in x]; ry = [sorted(y).index(v)+1 for v in y]
    sp = spearman_r(rx, ry)
    tid = f"c1_rankcorr_{m1}_{m2}"
    register(tid, p_tau, tau=tau, spearman=sp, n=len(common))
    rank_corr_results[(m1,m2)] = {"tau":tau,"sp":sp,"p":p_tau,"n":len(common),"tid":tid}

# Minitaur vs new models rank correlation
rank_corr_phase2 = {}
for model in ["gemma","llama3.1","qwen"]:
    g = model_results[model]["nasr"]["groups"]
    common = [gt for gt in GRAPH_ORDER if gt in phase2_nasr and g.get(gt)]
    p2 = [phase2_nasr[gt] for gt in common]
    mv = [mean(g[gt]) for gt in common]
    tau, p_tau = kendall_tau(p2, mv)
    tid = f"c1_rankcorr_phase2_{model}"
    register(tid, p_tau, tau=tau, n=len(common))
    rank_corr_phase2[model] = {"tau":tau,"p":p_tau,"n":len(common),"tid":tid}

# ---- Comment 2: density ----
all_new = runs_model + runs_nonews + runs_question
param_graphs = ["barabasi_albert","random","stochastic_block","powerlaw_cluster","forest_fire"]

ff_runs = [r for r in all_new if r["graph_type"]=="forest_fire"]
r_ff_nasr, p_ff_nasr, n_ff_nasr = pearson_r([r["density"] for r in ff_runs],[r["nasr"] for r in ff_runs if r.get("nasr") is not None])
# align correctly
ff_dn = [(r["density"],r["nasr"]) for r in ff_runs if r.get("nasr") is not None]
r_ff_nasr, p_ff_nasr, n_ff_nasr = pearson_r([x for x,_ in ff_dn],[y for _,y in ff_dn])
ff_dncc = [(r["density"],r["ncc"]) for r in ff_runs if r.get("ncc") is not None]
r_ff_ncc, p_ff_ncc, n_ff_ncc = pearson_r([x for x,_ in ff_dncc],[y for _,y in ff_dncc])
param_runs = [r for r in all_new if r["graph_type"] in param_graphs]
param_dn = [(r["density"],r["nasr"]) for r in param_runs if r.get("nasr") is not None]
r_param, p_param, n_param = pearson_r([x for x,_ in param_dn],[y for _,y in param_dn])

register("c2_ff_density_nasr", p_ff_nasr, r=r_ff_nasr, n=n_ff_nasr)
register("c2_ff_density_ncc",  p_ff_ncc,  r=r_ff_ncc,  n=n_ff_ncc)
register("c2_param_density_nasr", p_param, r=r_param,   n=n_param)

density_results = {
    "ff_nasr":  {"r":r_ff_nasr, "p":p_ff_nasr, "n":n_ff_nasr, "tid":"c2_ff_density_nasr"},
    "ff_ncc":   {"r":r_ff_ncc,  "p":p_ff_ncc,  "n":n_ff_ncc,  "tid":"c2_ff_density_ncc"},
    "param_nasr":{"r":r_param,  "p":p_param,   "n":n_param,   "tid":"c2_param_density_nasr"},
}

# ---- Comment 3: BERT ----
F_bert, p_bert, eta2_bert, bert_groups = topology_anova(runs_baseline, "bert_accuracy")
register("c3_bert_anova", p_bert, F=F_bert, eta2=eta2_bert, n=len(runs_baseline))
bert_results = {"F":F_bert,"p":p_bert,"eta2":eta2_bert,"groups":bert_groups,"tid":"c3_bert_anova"}

# ---- Comment 4: questions ----
question_results = {}
for q in [25,28,29]:
    qr = [r for r in runs_question if r["question"]==q]
    question_results[q] = {}
    for metric, label in METRICS:
        F, p, eta2, grp = topology_anova(qr, metric)
        tid = f"c4_q{q}_{metric}_anova"
        register(tid, p, F=F, eta2=eta2, n=len(qr))
        question_results[q][metric] = {"F":F,"p":p,"eta2":eta2,"groups":grp,"tid":tid}

question_nasr_means = {}
for q in [25,28,29]:
    g = question_results[q]["nasr"]["groups"]
    question_nasr_means[q] = {gt: mean(g.get(gt,[])) for gt in GRAPH_ORDER if g.get(gt)}

q_rank_results = {}
for q1,q2 in [(25,28),(25,29),(28,29)]:
    common=[gt for gt in GRAPH_ORDER
            if not math.isnan(question_nasr_means[q1].get(gt,float("nan")))
            and not math.isnan(question_nasr_means[q2].get(gt,float("nan")))]
    x=[question_nasr_means[q1][gt] for gt in common]
    y=[question_nasr_means[q2][gt] for gt in common]
    tau, p_tau = kendall_tau(x, y)
    rx=[sorted(x).index(v)+1 for v in x]; ry=[sorted(y).index(v)+1 for v in y]
    sp = spearman_r(rx, ry)
    tid=f"c4_rankcorr_q{q1}_q{q2}"
    register(tid, p_tau, tau=tau, spearman=sp, n=len(common))
    q_rank_results[(q1,q2)]={"tau":tau,"sp":sp,"p":p_tau,"n":len(common),"tid":tid}

# ---- Comment 5: no-news ----
nonews_results = {}
for metric, label in METRICS:
    F, p, eta2, grp = topology_anova(runs_nonews, metric)
    tid = f"c5_nonews_{metric}_anova"
    register(tid, p, F=F, eta2=eta2, n=len(runs_nonews))
    nonews_results[metric] = {"F":F,"p":p,"eta2":eta2,"groups":grp,"tid":tid}

# Homophily in no-news
for metric, label in [("nasr","NASR"),("ncc","NCC")]:
    ht = [r[metric] for r in runs_nonews if r["homophily"] is True and r.get(metric) is not None]
    hf = [r[metric] for r in runs_nonews if r["homophily"] is False and r.get(metric) is not None]
    t_val, p_val = welch_t(ht, hf)
    d_val = cohens_d(ht, hf)
    tid = f"c5_homophily_nonews_{metric}"
    register(tid, p_val, t=t_val, d=d_val, mean_true=mean(ht), mean_false=mean(hf),
             n_true=len(ht), n_false=len(hf))
    nonews_results[f"hom_{metric}"] = {"t":t_val,"p":p_val,"d":d_val,
                                        "mean_true":mean(ht),"mean_false":mean(hf),
                                        "n_true":len(ht),"n_false":len(hf),"tid":tid}

# News vs no-news direct comparison
news_vs_nonews = {}
for metric, label in [("nasr","NASR"),("ncc","NCC"),("osr","OSR")]:
    nv = [r[metric] for r in runs_baseline if r.get(metric) is not None]
    nnv= [r[metric] for r in runs_nonews if r.get(metric) is not None]
    t_val, p_val = welch_t(nv, nnv)
    d_val = cohens_d(nv, nnv)
    tid = f"c5_news_vs_nonews_{metric}"
    register(tid, p_val, t=t_val, d=d_val, mean_news=mean(nv), mean_nonews=mean(nnv),
             n_news=len(nv), n_nonews=len(nnv))
    news_vs_nonews[metric] = {"t":t_val,"p":p_val,"d":d_val,
                               "mean_news":mean(nv),"mean_nonews":mean(nnv),
                               "n_news":len(nv),"n_nonews":len(nnv),"tid":tid}

# Summary baseline and no-news topology ANOVAs (already registered above)
# Compute for Q28 baseline too (already in question_results[28])
# Additional: baseline NASR/NCC anova
F_bl_nasr, p_bl_nasr, eta2_bl_nasr, _ = topology_anova(runs_baseline,"nasr")
F_bl_ncc,  p_bl_ncc,  eta2_bl_ncc,  _ = topology_anova(runs_baseline,"ncc")
register("c2_baseline_nasr_anova", p_bl_nasr, F=F_bl_nasr, eta2=eta2_bl_nasr)
register("c2_baseline_ncc_anova",  p_bl_ncc,  F=F_bl_ncc,  eta2=eta2_bl_ncc)

# ---------------------------------------------------------------------------
# PASS 2: apply BH FDR
# ---------------------------------------------------------------------------

Q_VALS = bh_correct(ALL_TESTS)

def qval(tid: str) -> float:
    return Q_VALS.get(tid, float("nan"))

def fmtpq(p: float, q: float) -> str:
    return f"p={fmtp(p)}, q={fmtp(q)}"

# ---------------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------------

LINES = []
def out(s=""): LINES.append(s); print(s)
def section(t): out(); out("="*80); out(f"## {t}"); out("="*80)
def subsection(t): out(); out(f"### {t}"); out("-"*60)

section("DATA LOADING")
out(f"Model runs loaded       : {len(runs_model)}")
out(f"No-news runs loaded     : {len(runs_nonews)}")
out(f"Question runs loaded    : {len(runs_question)}")
out(f"Baseline (Q28) loaded   : {len(runs_baseline)}")
out(f"Total registered tests  : {len(ALL_TESTS)}")
out()
out("Model runs by model:"); [out(f"  {m}: {sum(1 for r in runs_model if r['model']==m)}") for m in ["gemma","llama3.1","qwen"]]
out("Question runs by question:"); [out(f"  Q{q}: {sum(1 for r in runs_question if r['question']==q)}") for q in [25,28,29]]

# ---- Comment 1 ----
section("COMMENT 1: SINGLE BASE MODEL — GENERALIZABILITY")

subsection("1.1 Per-model topology ANOVA on NASR (permutation, 2000 perms)")
out(f"{'Model':<14} {'N':>5} {'F':>8} {'η²':>7}  {'p (perm)':>10}  {'q (BH)':>10}  {'sig':>5}")
for model in ["gemma","llama3.1","qwen"]:
    r = model_results[model]["nasr"]
    q = qval(r["tid"])
    out(f"  {model:<12} {r['n']:>5} {fmt(r['F']):>8} {fmt(r['eta2']):>7}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {sig(q):>5}")

out()
subsection("1.2 Per-model topology ANOVA on all metrics")
out(f"{'Model':<12} {'Metric':<6} {'F':>8} {'η²':>7}  {'p (perm)':>10}  {'q (BH)':>10}")
for model in ["gemma","llama3.1","qwen"]:
    for metric, label in METRICS:
        r = model_results[model][metric]
        q = qval(r["tid"])
        out(f"  {model:<12} {label:<6} {fmt(r['F']):>8} {fmt(r['eta2']):>7}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {sig(q)}")

out()
subsection("1.3 Per-topology NASR means by model")
out(f"  {'Topology':<22}  {'Phase2':>9}  {'Gemma':>9}  {'Llama3.1':>9}  {'Qwen':>9}")
out("  " + "-"*66)
for gt in GRAPH_ORDER:
    row = f"  {GRAPH_LABEL.get(gt,gt):<22}  {phase2_nasr[gt]:>9.3f}"
    for model in ["gemma","llama3.1","qwen"]:
        vals = model_results[model]["nasr"]["groups"].get(gt,[])
        row += f"  {mean(vals):>6.3f}(n={len(vals):>2})" if vals else f"  {'—':>9}"
    out(row)

out()
subsection("1.4 Rank-ordering consistency of NASR across model pairs")
out(f"  {'Pair':<24} {'Spearman r_s':>12} {'Kendall τ':>10}  {'p':>10}  {'q (BH)':>10}  {'sig':>5}")
for (m1,m2), r in rank_corr_results.items():
    q = qval(r["tid"])
    out(f"  {m1+' vs '+m2:<24} {fmt(r['sp']):>12} {fmt(r['tau']):>10}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {sig(q):>5}")

out()
subsection("1.5 Kendall τ of each new model vs. Phase 2 NASR ordering")
out(f"  {'Model':<14} {'Kendall τ':>10}  {'p':>10}  {'q (BH)':>10}")
for model, r in rank_corr_phase2.items():
    q = qval(r["tid"])
    out(f"  {model:<14} {fmt(r['tau']):>10}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {sig(q)}")

out()
subsection("1.6 NCC per-topology means by model (absolute direction varies)")
out(f"  {'Topology':<22}  {'Gemma':>14}  {'Llama3.1':>14}  {'Qwen':>14}")
out("  " + "-"*68)
for gt in GRAPH_ORDER:
    row = f"  {GRAPH_LABEL.get(gt,gt):<22}"
    for model in ["gemma","llama3.1","qwen"]:
        vals = model_results[model]["ncc"]["groups"].get(gt,[])
        row += f"  {mean(vals):>7.3f}(n={len(vals):>2})" if vals else f"  {'—':>14}"
    out(row)

# ---- Comment 2 ----
section("COMMENT 2: GRAPH DENSITY AND PARAMETERIZATION")

subsection("2.1 Actual edge densities from new runs")
density_by_graph = collections.defaultdict(list)
for r in all_new:
    if r["graph_type"]: density_by_graph[r["graph_type"]].append(r["density"])
# Report mean degree = 2 * (edges / nodes) for undirected graphs
out(f"{'Topology':<22} {'N':>6} {'Mean degree':>12} {'SD':>8} {'Min':>8} {'Max':>8}")
out("-"*75)
for gt in GRAPH_ORDER:
    d = density_by_graph.get(gt,[])
    if d:
        deg = [x * 2 for x in d]
        out(f"{GRAPH_LABEL.get(gt,gt):<22} {len(d):>6} {mean(deg):>12.2f} {stdev(deg):>8.2f} {min(deg):>8.2f} {max(deg):>8.2f}")
out()
out("Parameters (from source code src/main.py):")
out("  ER: p = 1/16 (expected mean degree = 256 × 1/16 = 16.0)")
out("  PLC: m=8, p_triangle=0.4 (expected mean degree ≈ 15.3)")
out("  BA: m=8 (expected mean degree ≈ 2 × 8 × (257-8)/257 ≈ 15.5)")
out("  SBM: 4 balanced blocks (~64 nodes each), p_in=0.17, p_out=0.028 (expected mean degree ≈ 16.1)")
out("  FF: forward_burn_prob=0.21, max_burn_visits=4 (variable; see §2.2)")
out("  Cycle: by definition mean degree=2; cannot be parameterized otherwise.")
out("  Fully Connected: by definition mean degree=N-1; cannot be density-matched.")

subsection("2.2 Density as confound: empirical test")
out(f"  {'Test':<45} {'r':>7}  {'p':>10}  {'q (BH)':>10}  {'n':>5}")
for key, lab in [("ff_nasr","FF density vs NASR"),("ff_ncc","FF density vs NCC"),("param_nasr","All 5 param. topologies: density vs NASR")]:
    dr = density_results[key]
    q = qval(dr["tid"])
    out(f"  {lab:<45} {fmt(dr['r']):>7}  {fmtp(dr['p']):>10}  {fmtp(q):>10}  {dr['n']:>5}  {sig(q)}")

# ---- Comment 3 ----
section("COMMENT 3: REALISM METRIC (BERT)")

subsection("3.1 BERT accuracy by topology (baseline Q28 runs, n=32)")
out(f"{'Topology':<22} {'N':>5} {'Mean acc':>9} {'SD':>7}")
out("-"*48)
for gt in GRAPH_ORDER:
    vals = bert_results["groups"].get(gt,[])
    if vals: out(f"{GRAPH_LABEL.get(gt,gt):<22} {len(vals):>5} {mean(vals):>9.3f} {stdev(vals):>7.3f}")
ba=[r["bert_accuracy"] for r in runs_baseline if r.get("bert_accuracy") is not None]
out()
out(f"Overall: mean={mean(ba):.3f} ± {stdev(ba):.3f}, range=[{min(ba):.3f},{max(ba):.3f}]")
q3 = qval("c3_bert_anova")
out(f"Graph-type effect: F={fmt(bert_results['F'])}, {fmtpq(bert_results['p'],q3)} {sig(q3)}, η²={fmt(bert_results['eta2'])}")

subsection("3.2 BERT from model runs (all 3 new models)")
for model in ["gemma","llama3.1","qwen"]:
    vals=[r["bert_accuracy"] for r in runs_model if r["model"]==model and r.get("bert_accuracy") is not None]
    out(f"  {model:<12}: mean={mean(vals):.3f} ± {stdev(vals):.3f}  (n={len(vals)})")

# ---- Comment 4 ----
section("COMMENT 4: SINGLE SURVEY QUESTION — ROBUSTNESS ACROSS Q25, Q28, Q29")

subsection("4.1 Initial conditions by question")
for q in [25,28,29]:
    qr=[r for r in runs_question if r["question"]==q]
    ic=[r["initial_consensus"] for r in qr if r.get("initial_consensus") is not None]
    out(f"  Q{q}: initial consensus = {mean(ic):.3f} ± {stdev(ic):.3f}  (n={len(ic)})")

subsection("4.2 Topology ANOVAs per question")
out(f"{'Q':>3} {'Metric':<6} {'F':>8} {'η²':>7}  {'p (perm)':>10}  {'q (BH)':>10}  {'sig':>5}")
for q in [25,28,29]:
    for metric, label in METRICS:
        r = question_results[q].get(metric,{})
        if not r: continue
        q_bh = qval(r["tid"])
        out(f"  Q{q} {label:<6} {fmt(r['F']):>8} {fmt(r['eta2']):>7}  {fmtp(r['p']):>10}  {fmtp(q_bh):>10}  {sig(q_bh):>5}")

subsection("4.3 Per-topology NASR means by question")
out(f"  {'Topology':<22}  {'Q25':>14}  {'Q28':>14}  {'Q29':>14}")
out("  " + "-"*68)
for gt in GRAPH_ORDER:
    row = f"  {GRAPH_LABEL.get(gt,gt):<22}"
    for q in [25,28,29]:
        g = question_results[q]["nasr"]["groups"]
        vals = g.get(gt,[])
        row += f"  {mean(vals):>7.3f}(n={len(vals):>2})" if vals else f"  {'—':>14}"
    out(row)

subsection("4.4 NASR rank-ordering consistency across questions")
out(f"  {'Pair':<12} {'Spearman r_s':>12} {'Kendall τ':>10}  {'p':>10}  {'q (BH)':>10}  {'sig':>5}")
for (q1,q2), r in q_rank_results.items():
    q_bh = qval(r["tid"])
    out(f"  Q{q1} vs Q{q2}    {fmt(r['sp']):>12} {fmt(r['tau']):>10}  {fmtp(r['p']):>10}  {fmtp(q_bh):>10}  {sig(q_bh):>5}")

# ---- Comment 5 ----
section("COMMENT 5: NEWS-AGENT PLACEMENT CONFOUND")

subsection("5.1 Topology ANOVAs without news agent (n=70)")
out(f"{'Metric':<6} {'F':>8} {'η²':>7}  {'p (perm)':>10}  {'q (BH)':>10}  {'sig':>5}")
for metric, label in METRICS:
    r = nonews_results[metric]
    q = qval(r["tid"])
    out(f"  {label:<6} {fmt(r['F']):>8} {fmt(r['eta2']):>7}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {sig(q):>5}")

subsection("5.2 Per-topology NASR and NCC: no-news vs Phase 2")
phase2_ncc = {"cycle":-0.150,"forest_fire":-0.080,"fully_connected":-0.064,
              "stochastic_block":-0.121,"random":-0.107,"powerlaw_cluster":-0.112,"barabasi_albert":-0.132}
out(f"  {'Topology':<22}  {'NASR no-news':>13}  {'NASR Phase2':>12}  {'NCC no-news':>12}  {'NCC Phase2':>10}")
out("  " + "-"*76)
for gt in GRAPH_ORDER:
    nn_nasr_vals = nonews_results["nasr"]["groups"].get(gt,[])
    nn_ncc_vals  = nonews_results["ncc"]["groups"].get(gt,[])
    nn_nasr = f"{mean(nn_nasr_vals):>7.3f}(n={len(nn_nasr_vals):>2})" if nn_nasr_vals else "—"
    nn_ncc  = f"{mean(nn_ncc_vals):>7.3f}(n={len(nn_ncc_vals):>2})" if nn_ncc_vals else "—"
    p2_nasr = f"{phase2_nasr.get(gt,float('nan')):.3f}"
    p2_ncc  = f"{phase2_ncc.get(gt,float('nan')):.3f}"
    out(f"  {GRAPH_LABEL.get(gt,gt):<22}  {nn_nasr:>13}  {p2_nasr:>12}  {nn_ncc:>12}  {p2_ncc:>10}")

subsection("5.3 Homophily effect in no-news condition")
out(f"  {'Metric':<6} {'hom=True':>18} {'hom=False':>18}  {'p':>10}  {'q (BH)':>10}  {'d':>7}")
for metric in ["nasr","ncc"]:
    r = nonews_results[f"hom_{metric}"]
    q = qval(r["tid"])
    mt = f"{r['mean_true']:.3f}(n={r['n_true']})"
    mf = f"{r['mean_false']:.3f}(n={r['n_false']})"
    out(f"  {metric.upper():<6} {mt:>18} {mf:>18}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {fmt(r['d']):>7}  {sig(q)}")

subsection("5.4 Direct news vs no-news metric comparison")
out(f"  {'Metric':<6} {'With news':>18} {'No news':>18}  {'Δ':>8}  {'p':>10}  {'q (BH)':>10}  {'d':>7}")
for metric, label in [("nasr","NASR"),("ncc","NCC"),("osr","OSR")]:
    r = news_vs_nonews[metric]
    q = qval(r["tid"])
    mn = f"{r['mean_news']:.3f}(n={r['n_news']})"
    mnn= f"{r['mean_nonews']:.3f}(n={r['n_nonews']})"
    delta = r["mean_news"] - r["mean_nonews"]
    out(f"  {label:<6} {mn:>18} {mnn:>18}  {delta:>+8.3f}  {fmtp(r['p']):>10}  {fmtp(q):>10}  {fmt(r['d']):>7}  {sig(q)}")

# ---- Full test registry summary ----
section("FULL TEST REGISTRY (all registered tests with BH-corrected q-values)")
out(f"Total tests in battery: {len(ALL_TESTS)}")
out()
out(f"{'Label':<45} {'p':>10}  {'q (BH)':>10}  {'sig':>5}")
out("-"*80)
for t in sorted(ALL_TESTS, key=lambda x: x["p"]):
    q = qval(t["label"])
    out(f"  {t['label']:<43} {fmtp(t['p']):>10}  {fmtp(q):>10}  {sig(q):>5}")

# ---- Grand summary ----
section("SUMMARY: TOPOLOGY EFFECT SIZES ACROSS ALL CONDITIONS")
out(f"{'Condition':<37} {'N':>5}  {'η² NASR':>8} {'p':>8} {'q':>8}  {'η² NCC':>8} {'p':>8} {'q':>8}")
out("-"*100)

conditions = [
    ("Phase 2 / Minitaur (paper)",    156, 0.906, float("nan"), float("nan"), 0.285, float("nan"), float("nan")),
]
# Add computed conditions
cond_data = [
    ("No-news / Minitaur / Q28",      runs_nonews,  "c5_nonews_nasr_anova",  "c5_nonews_ncc_anova"),
    ("Q28 baseline / Minitaur",        runs_baseline,"c2_baseline_nasr_anova","c2_baseline_ncc_anova"),
    ("Q25 / Minitaur",                 [r for r in runs_question if r["question"]==25], "c4_q25_nasr_anova","c4_q25_ncc_anova"),
    ("Q28 / Minitaur (question runs)", [r for r in runs_question if r["question"]==28], "c4_q28_nasr_anova","c4_q28_ncc_anova"),
    ("Q29 / Minitaur",                 [r for r in runs_question if r["question"]==29], "c4_q29_nasr_anova","c4_q29_ncc_anova"),
    ("Model: Gemma-3-4B",              [r for r in runs_model if r["model"]=="gemma"],  "c1_gemma_nasr_anova","c1_gemma_ncc_anova"),
    ("Model: Llama-3.1-8B",            [r for r in runs_model if r["model"]=="llama3.1"],"c1_llama3.1_nasr_anova","c1_llama3.1_ncc_anova"),
    ("Model: Qwen2.5-7B",              [r for r in runs_model if r["model"]=="qwen"],   "c1_qwen_nasr_anova","c1_qwen_ncc_anova"),
]

for label, run_list, nasr_tid, ncc_tid in cond_data:
    n = len(run_list)
    nasr_t = next((t for t in ALL_TESTS if t["label"]==nasr_tid), None)
    ncc_t  = next((t for t in ALL_TESTS if t["label"]==ncc_tid), None)
    eta2_nasr = nasr_t.get("eta2",float("nan")) if nasr_t else float("nan")
    p_nasr    = nasr_t["p"] if nasr_t else float("nan")
    q_nasr    = qval(nasr_tid)
    eta2_ncc  = ncc_t.get("eta2",float("nan")) if ncc_t else float("nan")
    p_ncc     = ncc_t["p"] if ncc_t else float("nan")
    q_ncc     = qval(ncc_tid)
    out(f"  {label:<37} {n:>5}  {fmt(eta2_nasr):>8} {fmtp(p_nasr):>8} {fmtp(q_nasr):>8}  {fmt(eta2_ncc):>8} {fmtp(p_ncc):>8} {fmtp(q_ncc):>8}  {sig(q_nasr)}/{sig(q_ncc)}")

out()
out("Phase 2 p-values from paper (not re-tested here; listed for reference only).")
out("All other p/q values are from new experiments. BH-FDR applied across all registered tests.")

out()
out("="*80)
out("ANALYSIS COMPLETE")
out("="*80)

# Write output
with open("reports/reviewer_response_results.md", "w") as fh:
    fh.write("# Reviewer Response Analysis Results\n\n")
    fh.write("_Generated by `reports/reviewer_response_analysis.py`_\n\n")
    fh.write("\n".join(LINES))
print("\n[Written to reports/reviewer_response_results.md]")
