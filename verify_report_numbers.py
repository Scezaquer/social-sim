#!/usr/bin/env python3
"""
Reproduces all numbers quoted in analysis_report.md.
Run from the social-sim directory:
    ~/ENV-test/bin/python verify_report_numbers.py
"""

import collections
import glob
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def msd(vals: List) -> Tuple[Optional[float], Optional[float], int]:
    """Return (mean, stdev, n) ignoring None values."""
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def welch_t(a: List, b: List) -> Tuple[Optional[float], Optional[float]]:
    """Welch two-sample t-test; returns (t, p) using normal approximation for p."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None, None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def pearson_r(xs: List, ys: List) -> Tuple[Optional[float], Optional[float], int]:
    """Pearson r with two-tailed p; returns (r, p, n)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, None, len(pairs)
    x_, y_ = zip(*pairs)
    mx, my = statistics.mean(x_), statistics.mean(y_)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x_, y_))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x_))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y_))
    if sx == 0 or sy == 0:
        return 0.0, 1.0, len(pairs)
    r = num / (sx * sy)
    n = len(pairs)
    t_val = r * math.sqrt(n - 2) / math.sqrt(max(1e-12, 1 - r ** 2))
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_val) / math.sqrt(2))))
    return r, p_val, n


def sig_stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "(ns)"


def cohens_d(a: List, b: List) -> Optional[float]:
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    pooled_sd = math.sqrt((statistics.variance(a) * (len(a) - 1) + statistics.variance(b) * (len(b) - 1))
                          / (len(a) + len(b) - 2))
    return (ma - mb) / pooled_sd if pooled_sd > 0 else None


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

section("LOADING DATA")

files = sorted(glob.glob("visualizer_randomized_*.json"))
runs: List[Dict[str, Any]] = []

STANDARD_AGENTS = {64, 256, 1024, 4096}

for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
    except Exception:
        continue

    rp = d.get("run_parameters", {})
    bm_metrics = d.get("behavioral_metrics", {})
    bert = d.get("bert_real_vs_llm_classifier", {})
    herd = bm_metrics.get("herd_effect_metrics", {})
    echo = bm_metrics.get("echo_chamber_metrics", {})

    base_model = rp.get("base_model", "")
    model_short = (
        "minitaur" if "Minitaur" in base_model
        else ("llama3.1" if "Llama-3.1-8B" in base_model else "other")
    )

    # Extract per-transition opinion-shift for NASR
    nasr_vals = []
    for tr in herd.get("transitions", []):
        if isinstance(tr, dict):
            v = tr.get("neighbor_alignment_shift_rate")
            if v is not None:
                try:
                    nasr_vals.append(float(v))
                except (TypeError, ValueError):
                    pass

    num_agents = rp.get("num_agents")

    runs.append({
        "file": f,
        "question": rp.get("question_number"),
        "base_model": base_model,
        "model_short": model_short,
        "num_agents": num_agents,
        "num_agents_standard": num_agents in STANDARD_AGENTS if num_agents else None,
        "num_news_agents": rp.get("num_news_agents"),
        "graph_type": rp.get("graph_type"),
        "homophily": rp.get("homophily"),
        "add_survey_to_context": rp.get("add_survey_to_context"),
        "proportions_option": rp.get("proportions_option"),
        # Herd metrics
        "initial_consensus": herd.get("initial_consensus"),
        "final_consensus": herd.get("final_consensus"),
        "net_consensus_change": herd.get("net_consensus_change"),
        "mean_opinion_shift_rate": herd.get("mean_opinion_shift_rate"),
        "mean_majority_follow_rate": herd.get("mean_current_majority_follow_rate"),
        "mean_consensus_gain": herd.get("mean_consensus_gain"),
        # Echo
        "echo_assortativity": echo.get("network_assortativity"),
        "echo_local_agreement": echo.get("mean_local_agreement"),
        "echo_cross_cutting": echo.get("cross_cutting_edge_fraction"),
        "echo_same_option_exposure": echo.get("mean_same_option_exposure_share"),
        "echo_exposure_diversity": echo.get("mean_exposure_diversity"),
        # BERT
        "bert_accuracy": bert.get("accuracy"),
        "bert_n": bert.get("n"),
        # Misc
        "num_threads": len(d.get("threads", [])),
        "nasr_mean": statistics.mean(nasr_vals) if nasr_vals else None,
    })

print(f"Total runs loaded: {len(runs)}")


# ---------------------------------------------------------------------------
# Section 1: Dataset overview
# ---------------------------------------------------------------------------

section("SECTION 1 — DATASET OVERVIEW")

sub("Counts by question")
for q, n in sorted(collections.Counter(r["question"] for r in runs).items()):
    print(f"  Q{q}: n={n}")

sub("Counts by base_model")
for m, n in sorted(collections.Counter(r["base_model"] for r in runs).items()):
    print(f"  {m}: n={n}")

sub("Counts by num_agents")
for na, n in sorted(collections.Counter(r["num_agents"] for r in runs).items()):
    tag = "" if na in STANDARD_AGENTS else "  <-- NON-STANDARD"
    print(f"  {na}: n={n}{tag}")

sub("Non-standard num_agents runs (not in {64, 256, 1024, 4096})")
ns_runs = [r for r in runs if r["num_agents"] not in STANDARD_AGENTS]
print(f"  Total non-standard runs: {len(ns_runs)}")
model_ctr = collections.Counter(r["model_short"] for r in ns_runs)
prop_ctr = collections.Counter(r["proportions_option"] for r in ns_runs)
print(f"  By model: {dict(model_ctr)}")
print(f"  By proportions_option: {dict(prop_ctr)}")

sub("Counts by graph_type")
for v, n in sorted(collections.Counter(r["graph_type"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by homophily")
for v, n in sorted(collections.Counter(r["homophily"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by add_survey_to_context")
for v, n in sorted(collections.Counter(r["add_survey_to_context"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by proportions_option")
for v, n in sorted(collections.Counter(r["proportions_option"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by num_news_agents")
for v, n in sorted(collections.Counter(r["num_news_agents"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Thread count by num_agents (report: ~780, sd≈13)")
for na in sorted(set(r["num_agents"] for r in runs if r["num_agents"] is not None)):
    v = [r["num_threads"] for r in runs if r["num_agents"] == na]
    m, s, n = msd(v)
    print(f"  num_agents={na}: mean={m:.1f} sd={s:.1f} n={n}")


# ---------------------------------------------------------------------------
# Section 3.1: Consensus by model and question
# ---------------------------------------------------------------------------

section("SECTION 3.1 — CONSENSUS DECREASES: MODEL COMPARISON")

sub("Consensus trajectory by question × model")
for q in [25, 28, 29]:
    for m_s, label in [("minitaur", "Minitaur"), ("llama3.1", "Llama-3.1-8B")]:
        rr = [r for r in runs if r["question"] == q and r["model_short"] == m_s]
        init_m, _, _ = msd([r["initial_consensus"] for r in rr])
        delta_m, delta_s, n = msd([r["net_consensus_change"] for r in rr])
        if n > 0:
            print(f"  Q{q} {label}: n={n}, init={init_m:.3f}, "
                  f"Δconsensus={delta_m:.4f}±{delta_s:.4f}")

sub("Overall net_consensus_change by model (report: Minitaur −0.119, Llama +0.004)")
for m_s, label in [("minitaur", "Minitaur"), ("llama3.1", "Llama-3.1-8B")]:
    v = [r["net_consensus_change"] for r in runs if r["model_short"] == m_s]
    m, s, n = msd(v)
    pos = sum(1 for x in v if x is not None and x > 0)
    neg = sum(1 for x in v if x is not None and x < 0)
    print(f"  {label}: mean={m:.4f} sd={s:.4f} n={n}  "
          f"(increased={pos}/{n} = {100*pos//n}%, decreased={neg}/{n} = {100*neg//n}%)")

sub("Welch t-test: Minitaur vs Llama (report: t=−3.47, p=0.0005)")
a = [r["net_consensus_change"] for r in runs if r["model_short"] == "minitaur"]
b = [r["net_consensus_change"] for r in runs if r["model_short"] == "llama3.1"]
t, p = welch_t(a, b)
d_val = cohens_d(a, b)
print(f"  t={t:.3f}, p={p:.4f} {sig_stars(p)}, Cohen's d={d_val:.3f}")

sub("Consensus change direction by model")
for m_s, label in [("minitaur", "Minitaur"), ("llama3.1", "Llama-3.1-8B")]:
    rr = [r for r in runs if r["model_short"] == m_s]
    pos = sum(1 for r in rr if r["net_consensus_change"] is not None and r["net_consensus_change"] > 0)
    neg = sum(1 for r in rr if r["net_consensus_change"] is not None and r["net_consensus_change"] < 0)
    print(f"  {label}: n={len(rr)}, increased={pos} ({100*pos//len(rr)}%), "
          f"decreased={neg} ({100*neg//len(rr)}%)")


# ---------------------------------------------------------------------------
# Section 3.2: Agent count vs opinion shift rate
# ---------------------------------------------------------------------------

section("SECTION 3.2 — AGENT COUNT AND OPINION VOLATILITY")

sub("Opinion shift rate by num_agents for Minitaur")
for na in sorted(set(r["num_agents"] for r in runs if r["num_agents"] is not None)):
    v = [r["mean_opinion_shift_rate"] for r in runs
         if r["num_agents"] == na and r["model_short"] == "minitaur"]
    m, s, n = msd(v)
    if m is not None:
        print(f"  num_agents={na}: mean={m:.4f} sd={s:.4f} n={n}")

sub("Opinion shift rate by num_agents (all models)")
for na in sorted(set(r["num_agents"] for r in runs if r["num_agents"] is not None)):
    v = [r["mean_opinion_shift_rate"] for r in runs if r["num_agents"] == na]
    m, s, n = msd(v)
    if m is not None:
        print(f"  num_agents={na}: mean={m:.4f} sd={s:.4f} n={n}")

sub("Welch t-test: 64 vs 4096 agents (Minitaur only) [report: ratio 3.5×, p<0.001]")
small = [r["mean_opinion_shift_rate"] for r in runs
         if r["num_agents"] == 64 and r["model_short"] == "minitaur"]
large = [r["mean_opinion_shift_rate"] for r in runs
         if r["num_agents"] == 4096 and r["model_short"] == "minitaur"]
t, p = welch_t(small, large)
m_s, _, n_s = msd(small)
m_l, _, n_l = msd(large)
print(f"  64-agents: mean={m_s:.4f} n={n_s}")
print(f"  4096-agents: mean={m_l:.4f} n={n_l}")
print(f"  Ratio: {m_s/m_l:.2f}×, t={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Pearson correlation: num_agents vs opinion_shift_rate [report: r=−0.52, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["mean_opinion_shift_rate"] for r in runs]
)
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")

sub("Pearson correlation: num_agents vs net_consensus_change [report: r=0.03, ns]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["net_consensus_change"] for r in runs]
)
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")


# ---------------------------------------------------------------------------
# Section 3.3: Survey context — herding and BERT detectability
# ---------------------------------------------------------------------------

section("SECTION 3.3 — SURVEY CONTEXT: HERDING AND BERT DETECTABILITY")

sub("Majority follow rate by add_survey_to_context [report: 0.517 vs 0.472, p<0.001]")
for ctx in [True, False]:
    v = [r["mean_majority_follow_rate"] for r in runs
         if r["add_survey_to_context"] == ctx]
    m, s, n = msd(v)
    above = sum(1 for x in v if x is not None and x > 0.5)
    print(f"  ctx={ctx}: mean={m:.4f} sd={s:.4f} n={n}, frac>0.5: {above}/{n}")
t, p = welch_t(
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == False],
)
d_val = cohens_d(
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == False],
)
print(f"  Welch t={t:.3f}, p={p:.2e} {sig_stars(p)}, Cohen's d={d_val:.3f}")

sub("BERT accuracy by add_survey_to_context [report: 0.986 vs 0.952, +3.3pp, p<10^-15]")
for ctx in [True, False]:
    v = [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == ctx]
    m, s, n = msd(v)
    print(f"  ctx={ctx}: mean={m:.4f} sd={s:.4f} n={n}")
ctx_true_acc = [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == True]
ctx_false_acc = [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == False]
t, p = welch_t(ctx_true_acc, ctx_false_acc)
d_val = cohens_d(ctx_true_acc, ctx_false_acc)
m_t, _, _ = msd(ctx_true_acc)
m_f, _, _ = msd(ctx_false_acc)
print(f"  Difference: {m_t - m_f:.4f} ({(m_t - m_f)*100:.2f} pp)")
print(f"  Welch t={t:.3f}, p={p:.2e} {sig_stars(p)}, Cohen's d={d_val:.3f}")

sub("BERT accuracy by proportions × survey_context")
for p_opt in ["uniform", "blueprint", "average", "distribution"]:
    for ctx in [True, False]:
        v = [r["bert_accuracy"] for r in runs
             if r["proportions_option"] == p_opt and r["add_survey_to_context"] == ctx]
        m, s, n = msd(v)
        if m is not None and n > 0:
            print(f"  {p_opt}×ctx={ctx}: mean={m:.4f}±{s:.4f} n={n}")

sub("Overall BERT accuracy [report: mean=97.0% sd=2.7%, range [88.7%, 100%]]")
v = [r["bert_accuracy"] for r in runs if r["bert_accuracy"] is not None]
m, s, n = msd(v)
print(f"  mean={m:.4f} sd={s:.4f} n={n}, range=[{min(v):.4f}, {max(v):.4f}]")

sub("BERT accuracy by question")
for q in [25, 28, 29]:
    v = [r["bert_accuracy"] for r in runs if r["question"] == q and r["bert_accuracy"] is not None]
    m, s, n = msd(v)
    print(f"  Q{q}: mean={m:.4f} sd={s:.4f} n={n}")

sub("BERT accuracy by model")
for m_s, label in [("minitaur", "Minitaur"), ("llama3.1", "Llama-3.1-8B")]:
    v = [r["bert_accuracy"] for r in runs if r["model_short"] == m_s]
    mu, s, n = msd(v)
    print(f"  {label}: mean={mu:.4f} sd={s:.4f} n={n}")
t, p = welch_t(
    [r["bert_accuracy"] for r in runs if r["model_short"] == "minitaur"],
    [r["bert_accuracy"] for r in runs if r["model_short"] == "llama3.1"],
)
print(f"  Welch t (Minitaur vs Llama): t={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("bert_n (eval thread count) by add_survey_to_context")
for ctx in [True, False]:
    v = [r["bert_n"] for r in runs if r["add_survey_to_context"] == ctx and r["bert_n"] is not None]
    m, s, n = msd(v)
    print(f"  ctx={ctx}: mean bert_n={m:.1f} n={n}")

sub("Correlation: bert_n vs bert_accuracy (confound check) [report: r=−0.547]")
r_val, p_val, n = pearson_r(
    [r["bert_n"] for r in runs],
    [r["bert_accuracy"] for r in runs],
)
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")


# ---------------------------------------------------------------------------
# Section 3.4: Proportions option → consensus
# ---------------------------------------------------------------------------

section("SECTION 3.4 — PROPORTIONS OPTION AND CONSENSUS")

sub("Consensus stats by proportions_option")
for p_opt in ["uniform", "blueprint", "average", "distribution"]:
    rr = [r for r in runs if r["proportions_option"] == p_opt]
    init_m, _, _ = msd([r["initial_consensus"] for r in rr])
    delta_m, delta_s, n = msd([r["net_consensus_change"] for r in rr])
    shift_m, _, _ = msd([r["mean_opinion_shift_rate"] for r in rr])
    print(f"  {p_opt}: n={n}, init={init_m:.4f}, "
          f"Δconsensus={delta_m:.4f}±{delta_s:.4f}, shift_rate={shift_m:.4f}")

sub("Welch t-test: average vs uniform [report: t=−3.56, p=0.0004]")
avg = [r["net_consensus_change"] for r in runs if r["proportions_option"] == "average"]
uni = [r["net_consensus_change"] for r in runs if r["proportions_option"] == "uniform"]
t, p = welch_t(avg, uni)
print(f"  t={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Consensus change by proportions × model")
for p_opt in ["uniform", "blueprint", "average", "distribution"]:
    for m_s, label in [("minitaur", "Minitaur"), ("llama3.1", "Llama-3.1-8B")]:
        v = [r["net_consensus_change"] for r in runs
             if r["proportions_option"] == p_opt and r["model_short"] == m_s]
        m, s, n = msd(v)
        if n > 0 and m is not None:
            print(f"  {p_opt}×{label}: mean={m:.4f} sd={s:.4f} n={n}")


# ---------------------------------------------------------------------------
# Section 3.5: Echo chamber formation
# ---------------------------------------------------------------------------

section("SECTION 3.5 — ECHO CHAMBER FORMATION")

sub("Overall echo chamber metric distributions")
for key, label in [
    ("echo_assortativity", "Network assortativity"),
    ("echo_local_agreement", "Local agreement"),
    ("echo_cross_cutting", "Cross-cutting edges"),
    ("echo_same_option_exposure", "Same-option exposure"),
    ("echo_exposure_diversity", "Exposure diversity"),
]:
    v = [r[key] for r in runs if r[key] is not None]
    m, s, n = msd(v)
    print(f"  {label}: mean={m:.4f} sd={s:.4f} n={n}, range=[{min(v):.4f}, {max(v):.4f}]")

sub("Network assortativity by homophily [report: 0.011 vs −0.005, p=0.014]")
for h in [True, False]:
    v = [r["echo_assortativity"] for r in runs
         if r["homophily"] == h and r["echo_assortativity"] is not None]
    m, s, n = msd(v)
    print(f"  homophily={h}: mean={m:.4f} sd={s:.4f} n={n}")
t, p = welch_t(
    [r["echo_assortativity"] for r in runs if r["homophily"] == True],
    [r["echo_assortativity"] for r in runs if r["homophily"] == False],
)
print(f"  Welch t={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Network assortativity by graph_type [report: 0.008 vs −0.002, ns]")
for gt in ["random", "powerlaw_cluster"]:
    v = [r["echo_assortativity"] for r in runs
         if r["graph_type"] == gt and r["echo_assortativity"] is not None]
    m, s, n = msd(v)
    print(f"  {gt}: mean={m:.4f} sd={s:.4f} n={n}")
t, p = welch_t(
    [r["echo_assortativity"] for r in runs if r["graph_type"] == "random"],
    [r["echo_assortativity"] for r in runs if r["graph_type"] == "powerlaw_cluster"],
)
print(f"  Welch t={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Local agreement by graph_type [report: 0.700 vs 0.651, p=0.042]")
for gt in ["random", "powerlaw_cluster"]:
    v = [r["echo_local_agreement"] for r in runs
         if r["graph_type"] == gt and r["echo_local_agreement"] is not None]
    m, s, n = msd(v)
    print(f"  {gt}: mean={m:.4f} sd={s:.4f} n={n}")
t, p = welch_t(
    [r["echo_local_agreement"] for r in runs if r["graph_type"] == "powerlaw_cluster"],
    [r["echo_local_agreement"] for r in runs if r["graph_type"] == "random"],
)
print(f"  Welch t (powerlaw vs random)={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Local agreement by homophily [report: 0.689 vs 0.657, p=0.19 ns]")
for h in [True, False]:
    v = [r["echo_local_agreement"] for r in runs
         if r["homophily"] == h and r["echo_local_agreement"] is not None]
    m, s, n = msd(v)
    print(f"  homophily={h}: mean={m:.4f} sd={s:.4f} n={n}")
t, p = welch_t(
    [r["echo_local_agreement"] for r in runs if r["homophily"] == True],
    [r["echo_local_agreement"] for r in runs if r["homophily"] == False],
)
print(f"  Welch t={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Cross-cutting edges by graph_type [report: 0.349 vs 0.298, p=0.034]")
for gt in ["random", "powerlaw_cluster"]:
    v = [r["echo_cross_cutting"] for r in runs
         if r["graph_type"] == gt and r["echo_cross_cutting"] is not None]
    m, s, n = msd(v)
    print(f"  {gt}: mean={m:.4f} sd={s:.4f} n={n}")
t, p = welch_t(
    [r["echo_cross_cutting"] for r in runs if r["graph_type"] == "random"],
    [r["echo_cross_cutting"] for r in runs if r["graph_type"] == "powerlaw_cluster"],
)
print(f"  Welch t (random vs powerlaw)={t:.3f}, p={p:.4f} {sig_stars(p)}")

sub("Correlation: same-option exposure vs final consensus [report: r=0.65, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["echo_same_option_exposure"] for r in runs],
    [r["net_consensus_change"] for r in runs],
)
print(f"  same_option_exposure vs net_consensus_change: r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")

sub("Correlation: local agreement vs net_consensus_change [report: r=0.61, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["echo_local_agreement"] for r in runs],
    [r["net_consensus_change"] for r in runs],
)
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")

sub("Same-option exposure by graph_type × homophily")
for h in [True, False]:
    for gt in ["random", "powerlaw_cluster"]:
        v = [r["echo_same_option_exposure"] for r in runs
             if r["homophily"] == h and r["graph_type"] == gt
             and r["echo_same_option_exposure"] is not None]
        m, s, n = msd(v)
        if m is not None:
            print(f"  homophily={h}, graph={gt}: mean={m:.4f} sd={s:.4f} n={n}")


# ---------------------------------------------------------------------------
# Section 3.6: News agents
# ---------------------------------------------------------------------------

section("SECTION 3.6 — NEWS AGENTS (NULL RESULT)")

sub("Key metrics by num_news_agents")
for metric, label in [
    ("net_consensus_change", "Δconsensus"),
    ("mean_opinion_shift_rate", "opinion_shift_rate"),
    ("bert_accuracy", "BERT accuracy"),
]:
    for nn in [0, 1]:
        v = [r[metric] for r in runs if r["num_news_agents"] == nn]
        m, s, n = msd(v)
        if m is not None:
            print(f"  {label} news={nn}: mean={m:.4f} sd={s:.4f} n={n}")
    t, p = welch_t(
        [r[metric] for r in runs if r["num_news_agents"] == 0],
        [r[metric] for r in runs if r["num_news_agents"] == 1],
    )
    print(f"  Welch t ({label})={t:.3f}, p={p:.4f} {sig_stars(p)}\n")


# ---------------------------------------------------------------------------
# Section 3.7: Per-question breakdown
# ---------------------------------------------------------------------------

section("SECTION 3.7 — PER-QUESTION BREAKDOWN")

sub("Consensus change direction by question")
for q in [25, 28, 29]:
    rr = [r for r in runs if r["question"] == q]
    pos = sum(1 for r in rr if r["net_consensus_change"] is not None and r["net_consensus_change"] > 0)
    neg = sum(1 for r in rr if r["net_consensus_change"] is not None and r["net_consensus_change"] < 0)
    v = [r["net_consensus_change"] for r in rr]
    m, s, n = msd(v)
    med = statistics.median([x for x in v if x is not None])
    print(f"  Q{q}: n={n}, mean={m:.4f} median={med:.4f}, "
          f"increased={pos} ({100*pos//n}%), decreased={neg} ({100*neg//n}%)")


# ---------------------------------------------------------------------------
# Additional checks: other correlations cited in summary table
# ---------------------------------------------------------------------------

section("ADDITIONAL CHECKS — SUMMARY TABLE NUMBERS")

sub("Cohen's d: Minitaur vs Llama (consensus change) [report: ≈0.93]")
d_val = cohens_d(
    [r["net_consensus_change"] for r in runs if r["model_short"] == "minitaur"],
    [r["net_consensus_change"] for r in runs if r["model_short"] == "llama3.1"],
)
print(f"  Cohen's d = {d_val:.3f}")

sub("Cohen's d: survey_context → majority follow rate [report: ≈0.64]")
d_val = cohens_d(
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == False],
)
print(f"  Cohen's d = {d_val:.3f}")

sub("Cohen's d: survey_context → BERT accuracy [report: ≈1.54]")
d_val = cohens_d(
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == True],
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == False],
)
print(f"  Cohen's d = {d_val:.3f}")

sub("Pearson: num_agents vs majority_follow_rate [report: r=−0.25, p=0.001]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["mean_majority_follow_rate"] for r in runs],
)
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")

sub("Pearson: opinion_shift_rate vs net_consensus_change [report: r=−0.36, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["mean_opinion_shift_rate"] for r in runs],
    [r["net_consensus_change"] for r in runs],
)
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig_stars(p_val)}, n={n}")

sub("NASR availability and stats")
nasr_runs = [r for r in runs if r["nasr_mean"] is not None]
print(f"  Runs with NASR populated: {len(nasr_runs)}/{len(runs)}")
m, s, n = msd([r["nasr_mean"] for r in nasr_runs])
print(f"  Overall NASR: mean={m:.4f} sd={s:.4f} n={n}")
for na in sorted(set(r["num_agents"] for r in nasr_runs if r["num_agents"] is not None)):
    v = [r["nasr_mean"] for r in nasr_runs if r["num_agents"] == na]
    mu, s_, n_ = msd(v)
    if mu is not None:
        print(f"  num_agents={na}: mean={mu:.4f} sd={s_:.4f} n={n_}")

print("\n" + "=" * 70)
print("  DONE — all report numbers reproduced.")
print("=" * 70)
