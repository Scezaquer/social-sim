#!/usr/bin/env python3
"""
Reproduces every number quoted in analysis_report.md (updated version).
Run from the social-sim directory:
    ~/ENV-test/bin/python verify_report_numbers.py

Excludes runs with num_agents not in {64, 256, 1024, 4096}.
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

STANDARD_AGENTS = {64, 256, 1024, 4096}
MODEL_NAMES = {
    "Minitaur": "minitaur",
    "Llama-3.1-8B": "llama3.1",
    "Qwen2.5-7B": "qwen",
    "gemma-3-4b": "gemma",
}


def norm_model(raw: str) -> str:
    for key, name in MODEL_NAMES.items():
        if key in raw:
            return name
    return f"unknown:{raw[:60]}"


def msd(vals: List) -> Tuple[Optional[float], Optional[float], int]:
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def welch_t(a: List, b: List) -> Tuple[Optional[float], Optional[float]]:
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None, None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def pearson_r(xs: List, ys: List) -> Tuple[Optional[float], Optional[float], int]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, None, len(pairs)
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


def cohens_d(a: List, b: List) -> Optional[float]:
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    pooled = math.sqrt(
        (statistics.variance(a) * (len(a) - 1) + statistics.variance(b) * (len(b) - 1))
        / (len(a) + len(b) - 2)
    )
    return (ma - mb) / pooled if pooled > 0 else None


def sig(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "(ns)"


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

section("LOADING DATA  (excluding non-standard num_agents)")

files = sorted(glob.glob("visualizer_randomized_*.json"))
runs: List[Dict[str, Any]] = []
skipped_nonstandard = 0

for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
    except Exception:
        continue
    rp = d.get("run_parameters", {})
    na = rp.get("num_agents")
    if na not in STANDARD_AGENTS:
        skipped_nonstandard += 1
        continue

    bm = d.get("behavioral_metrics", {})
    herd = bm.get("herd_effect_metrics", {})
    echo = bm.get("echo_chamber_metrics", {})
    bert = d.get("bert_real_vs_llm_classifier", {})
    transitions = herd.get("transitions", [])

    # Absolute change count per step (for normalization check)
    abs_changes = [
        tr.get("changed_users")
        for tr in transitions
        if isinstance(tr, dict) and tr.get("changed_users") is not None
    ]

    # NASR from transitions (per-step, averaged)
    nasr_vals = [
        tr.get("neighbor_alignment_shift_rate")
        for tr in transitions
        if isinstance(tr, dict) and tr.get("neighbor_alignment_shift_rate") is not None
    ]
    osr_trans_vals = [
        tr.get("opinion_shift_rate")
        for tr in transitions
        if isinstance(tr, dict) and tr.get("opinion_shift_rate") is not None
    ]

    # Echo time series via by_survey
    by_survey = echo.get("by_survey", [])

    runs.append({
        "file": f,
        "model": norm_model(rp.get("base_model", "")),
        "num_agents": na,
        "question": rp.get("question_number"),
        "graph_type": rp.get("graph_type"),
        "homophily": rp.get("homophily"),
        "add_survey_to_context": rp.get("add_survey_to_context"),
        "proportions_option": rp.get("proportions_option"),
        "num_news_agents": rp.get("num_news_agents"),
        # Herd
        "initial_consensus": herd.get("initial_consensus"),
        "final_consensus": herd.get("final_consensus"),
        "net_consensus_change": herd.get("net_consensus_change"),
        "mean_opinion_shift_rate": herd.get("mean_opinion_shift_rate"),
        "mean_majority_follow_rate": herd.get("mean_current_majority_follow_rate"),
        "mean_consensus_gain": herd.get("mean_consensus_gain"),
        # NASR
        "mean_nasr": statistics.mean(nasr_vals) if nasr_vals else None,
        "nasr_n": len(nasr_vals),
        "mean_osr_from_transitions": statistics.mean(osr_trans_vals) if osr_trans_vals else None,
        # Echo (final snapshot — for cross-sectional analysis)
        "echo_assortativity": echo.get("network_assortativity"),
        "echo_local_agreement": echo.get("mean_local_agreement"),
        "echo_cross_cutting": echo.get("cross_cutting_edge_fraction"),
        "echo_same_option_exposure": echo.get("mean_same_option_exposure_share"),
        # Echo time series
        "by_survey": by_survey,
        # BERT
        "bert_accuracy": bert.get("accuracy"),
        "bert_n": bert.get("n"),
        # Misc
        "num_threads": len(d.get("threads", [])),
        "abs_shift_mean": statistics.mean(abs_changes) if abs_changes else None,
        "transitions": transitions,
    })

print(f"Total eligible runs loaded : {len(runs)}")
print(f"Skipped (non-standard na)  : {skipped_nonstandard}")

unknowns = [r for r in runs if r["model"].startswith("unknown")]
if unknowns:
    print(f"WARNING: {len(unknowns)} runs with unrecognized model names:")
    for r in unknowns[:5]:
        print(f"  {r['file']}")


# ---------------------------------------------------------------------------
# Section 1: Dataset overview
# ---------------------------------------------------------------------------

section("SECTION 1 — DATASET OVERVIEW")

sub("Counts by model  [report: minitaur=131, llama3.1=78, qwen=105, gemma=40]")
for m, n in sorted(collections.Counter(r["model"] for r in runs).items()):
    print(f"  {m}: n={n}")

sub("Counts by num_agents  [report: 64→100, 256→102, 1024→88, 4096→64]")
for na, n in sorted(collections.Counter(r["num_agents"] for r in runs).items()):
    print(f"  {na}: n={n}")

sub("Counts by question  [report: 25→121, 28→117, 29→116]")
for q, n in sorted(collections.Counter(r["question"] for r in runs).items()):
    print(f"  Q{q}: n={n}")

sub("Counts by graph_type  [report: random=178, powerlaw=176]")
for v, n in sorted(collections.Counter(r["graph_type"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by homophily  [report: True=166, False=188]")
for v, n in sorted(collections.Counter(r["homophily"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by add_survey_to_context  [report: True=190, False=164]")
for v, n in sorted(collections.Counter(r["add_survey_to_context"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by proportions_option  [report: uniform=102, blueprint=96, average=76, distribution=80]")
for v, n in sorted(collections.Counter(r["proportions_option"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Counts by num_news_agents  [report: 0=175, 1=179]")
for v, n in sorted(collections.Counter(r["num_news_agents"] for r in runs).items()):
    print(f"  {v}: n={n}")

sub("Thread count by num_agents  [report: roughly constant ~780]")
for na in [64, 256, 1024, 4096]:
    v = [r["num_threads"] for r in runs if r["num_agents"] == na]
    m, s, n = msd(v)
    print(f"  num_agents={na}: mean={m:.1f} sd={s:.1f} n={n}")


# ---------------------------------------------------------------------------
# Section 2: Normalization check
# ---------------------------------------------------------------------------

section("SECTION 2 — NORMALIZATION CHECK")

sub("opinion_shift_rate is a fraction (changed/shared), already normalized")
# Verify by comparing stored rate vs computed ratio for first few runs with transitions
checked = 0
for r in runs:
    if checked >= 3:
        break
    trs = [t for t in r["transitions"] if isinstance(t, dict)
           and t.get("opinion_shift_rate") is not None
           and t.get("shared_users")]
    if not trs:
        continue
    stored = [t["opinion_shift_rate"] for t in trs[:3]]
    computed = [t["changed_users"] / t["shared_users"] for t in trs[:3]]
    match = all(abs(s - c) < 1e-9 for s, c in zip(stored, computed))
    print(f"  num_agents={r['num_agents']}: stored={[round(x,4) for x in stored]} "
          f"computed={[round(x,4) for x in computed]}  match={match}")
    checked += 1

sub("opinion_shift_rate (fraction) by num_agents  [report: 0.245, 0.264, 0.144, 0.112]")
for na in [64, 256, 1024, 4096]:
    v_rate = [r["mean_opinion_shift_rate"] for r in runs if r["num_agents"] == na]
    v_abs = [r["abs_shift_mean"] for r in runs if r["num_agents"] == na]
    m_r, s_r, n = msd(v_rate)
    m_a, s_a, _ = msd(v_abs)
    print(f"  na={na}: fraction={m_r:.4f}±{s_r:.4f}  abs_count={m_a:.1f}±{s_a:.1f}  n={n}")

sub("Correlation: num_agents vs fraction  [report: r=−0.48]  vs abs count  [report: r=+0.67]")
r_frac, p_frac, n_f = pearson_r(
    [r["num_agents"] for r in runs],
    [r["mean_opinion_shift_rate"] for r in runs])
r_abs, p_abs, n_a = pearson_r(
    [r["num_agents"] for r in runs],
    [r["abs_shift_mean"] for r in runs])
print(f"  fraction r={r_frac:.4f}, p={p_frac:.4f} {sig(p_frac)}, n={n_f}")
print(f"  abs_count r={r_abs:.4f}, p={p_abs:.4f} {sig(p_abs)}, n={n_a}")
print("  => fraction negative, abs positive — effect is real, not artifact")

sub("Within-model correlations: num_agents vs opinion_shift_rate  [report: all p<0.001]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    xs = [r["num_agents"] for r in runs if r["model"] == model]
    ys = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model]
    r_val, p_val, n = pearson_r(xs, ys)
    print(f"  {model}: r={r_val:.4f}, p={p_val:.4f} {sig(p_val)}, n={n}")


# ---------------------------------------------------------------------------
# Section 3.1: Model comparison — consensus
# ---------------------------------------------------------------------------

section("SECTION 3.1 — MODEL COMPARISON: CONSENSUS")

sub("Net consensus change by model  [report: minitaur=−0.108, qwen=−0.067, llama3.1=−0.021, gemma=+0.013]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    v = [r["net_consensus_change"] for r in runs if r["model"] == model]
    m, s, n = msd(v)
    pos = sum(1 for x in v if x is not None and x > 0)
    neg = sum(1 for x in v if x is not None and x < 0)
    print(f"  {model}: mean={m:.4f}±{s:.4f} n={n}  ↑{pos}({100*pos//n}%) ↓{neg}({100*neg//n}%)")

sub("Initial consensus by model")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    v = [r["initial_consensus"] for r in runs if r["model"] == model]
    m, s, n = msd(v)
    print(f"  {model}: mean={m:.4f}±{s:.4f} n={n}")

sub("Pairwise Welch t-tests on net_consensus_change  [report table in Section 3.1]")
pairs = [
    ("minitaur", "gemma",   "d=1.06, p<0.001"),
    ("minitaur", "llama3.1","d=0.60, p<0.001"),
    ("qwen",     "gemma",   "d=0.50, p=0.002"),
    ("minitaur", "qwen",    "d=0.30, p=0.030"),
    ("llama3.1", "qwen",    "ns"),
    ("llama3.1", "gemma",   "ns"),
]
for m1, m2, expected in pairs:
    a = [r["net_consensus_change"] for r in runs if r["model"] == m1]
    b = [r["net_consensus_change"] for r in runs if r["model"] == m2]
    t, p = welch_t(a, b)
    d = cohens_d(a, b)
    ma, _, na = msd(a)
    mb, _, nb = msd(b)
    print(f"  {m1}(n={na},{ma:.4f}) vs {m2}(n={nb},{mb:.4f}): "
          f"t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}  [expected: {expected}]")

sub("Consensus change by question × model  [report table in Section 3.1]")
for q in [25, 28, 29]:
    print(f"  Q{q}:")
    for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
        v = [r["net_consensus_change"] for r in runs if r["question"] == q and r["model"] == model]
        init_m, _, _ = msd([r["initial_consensus"] for r in runs if r["question"] == q and r["model"] == model])
        m, s, n = msd(v)
        if n > 0 and m is not None:
            pos = sum(1 for x in v if x is not None and x > 0)
            print(f"    {model}: n={n}, init={init_m:.3f}, Δ={m:.4f}±{s:.4f}, ↑{pos}({100*pos//n}%)")

sub("Unusual initial consensus values  [report: qwen Q28=0.993, gemma Q29=0.562, minitaur Q29=1.000]")
for q in [25, 28, 29]:
    for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
        v = [r["initial_consensus"] for r in runs if r["question"] == q and r["model"] == model]
        m, s, n = msd(v)
        if m is not None:
            print(f"  Q{q} {model}: init={m:.4f}±{s:.4f} n={n}")


# ---------------------------------------------------------------------------
# Section 3.2: Agent count vs opinion shift rate
# ---------------------------------------------------------------------------

section("SECTION 3.2 — AGENT COUNT vs OPINION SHIFT RATE")

sub("opinion_shift_rate by model × num_agents  [report table]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    row = []
    for na in [64, 256, 1024, 4096]:
        v = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model and r["num_agents"] == na]
        m, s, n = msd(v)
        row.append(f"n{na}={m:.3f}(n={n})" if m is not None else f"n{na}=N/A")
    print(f"  {model}: {', '.join(row)}")

sub("64 vs 1024 ratio by model  [report: ~1.63–1.98×, all p<0.001]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    s64 = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model and r["num_agents"] == 64]
    s1024 = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model and r["num_agents"] == 1024]
    ms, _, ns = msd(s64)
    ml, _, nl = msd(s1024)
    t, p = welch_t(s64, s1024)
    if ms and ml:
        print(f"  {model}: 64={ms:.3f}(n={ns}) vs 1024={ml:.3f}(n={nl}), "
              f"ratio={ms/ml:.2f}x, t={t:.3f}, p={p:.4f} {sig(p)}")

sub("64 vs 4096 for models that have 4096 runs")
for model in ["minitaur", "llama3.1", "qwen"]:
    s64 = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model and r["num_agents"] == 64]
    s4096 = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model and r["num_agents"] == 4096]
    ms, _, ns = msd(s64)
    ml, _, nl = msd(s4096)
    t, p = welch_t(s64, s4096)
    if ms and ml:
        print(f"  {model}: 64={ms:.3f}(n={ns}) vs 4096={ml:.3f}(n={nl}), "
              f"ratio={ms/ml:.2f}x, t={t:.3f}, p={p:.4f} {sig(p)}")

sub("Correlation: num_agents vs net_consensus_change  [report: r=0.03, ns]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["net_consensus_change"] for r in runs])
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig(p_val)}, n={n}")


# ---------------------------------------------------------------------------
# Section 3.3: Survey context
# ---------------------------------------------------------------------------

section("SECTION 3.3 — SURVEY CONTEXT")

sub("Majority follow rate by ctx  [report: 0.521 vs 0.492, d=0.46, p<0.001]")
for ctx in [True, False]:
    v = [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == ctx]
    m, s, n = msd(v)
    print(f"  ctx={ctx}: mean={m:.4f}±{s:.4f} n={n}")
t, p = welch_t(
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == False])
d = cohens_d(
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == False])
print(f"  Welch t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}")

sub("BERT accuracy by ctx  [report: 0.982 vs 0.939, d=1.49, p<10^-15]")
for ctx in [True, False]:
    v = [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == ctx and r["bert_accuracy"] is not None]
    m, s, n = msd(v)
    print(f"  ctx={ctx}: mean={m:.4f}±{s:.4f} n={n}")
t, p = welch_t(
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == True],
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == False])
d = cohens_d(
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == True],
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == False])
print(f"  Welch t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}")

sub("Survey context distribution within each model  [for confound check]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    ct = sum(1 for r in runs if r["model"] == model and r["add_survey_to_context"] == True)
    cf = sum(1 for r in runs if r["model"] == model and r["add_survey_to_context"] == False)
    print(f"  {model}: ctx=True:{ct}, ctx=False:{cf}")

sub("Ctx effect on consensus change WITHIN each model  [report table in 3.3]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    for ctx in [True, False]:
        v = [r["net_consensus_change"] for r in runs
             if r["model"] == model and r["add_survey_to_context"] == ctx]
        m, s, n = msd(v)
        if m is not None:
            print(f"  {model} ctx={ctx}: mean={m:.4f}±{s:.4f} n={n}")
    t, p = welch_t(
        [r["net_consensus_change"] for r in runs if r["model"] == model and r["add_survey_to_context"] == True],
        [r["net_consensus_change"] for r in runs if r["model"] == model and r["add_survey_to_context"] == False])
    print(f"  {model}: ctx effect t={t:.3f}, p={p:.4f} {sig(p)}")

sub("BERT accuracy by ctx within each model  [report: gemma near-zero ctx effect]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    for ctx in [True, False]:
        v = [r["bert_accuracy"] for r in runs
             if r["model"] == model and r["add_survey_to_context"] == ctx and r["bert_accuracy"] is not None]
        m, s, n = msd(v)
        if m is not None:
            print(f"  {model} ctx={ctx}: mean={m:.4f}±{s:.4f} n={n}")

sub("BERT ctx effect for non-gemma only  [report: 98.0% vs 93.4%]")
for ctx in [True, False]:
    v = [r["bert_accuracy"] for r in runs
         if r["model"] != "gemma" and r["add_survey_to_context"] == ctx and r["bert_accuracy"] is not None]
    m, s, n = msd(v)
    print(f"  non-gemma ctx={ctx}: mean={m:.4f}±{s:.4f} n={n}")


# ---------------------------------------------------------------------------
# Section 3.4: Gemma BERT accuracy
# ---------------------------------------------------------------------------

section("SECTION 3.4 — GEMMA BERT ACCURACY")

sub("BERT accuracy by model  [report: gemma=0.998, others ~0.954–0.959]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    v = [r["bert_accuracy"] for r in runs if r["model"] == model and r["bert_accuracy"] is not None]
    m, s, n = msd(v)
    if m is not None:
        print(f"  {model}: mean={m:.4f}±{s:.4f} n={n}, range=[{min(v):.4f},{max(v):.4f}]")

sub("Welch t-tests: gemma vs each other model  [report: all d~1.25–1.45, p<0.001]")
for model in ["minitaur", "llama3.1", "qwen"]:
    a = [r["bert_accuracy"] for r in runs if r["model"] == "gemma"]
    b = [r["bert_accuracy"] for r in runs if r["model"] == model]
    t, p = welch_t(a, b)
    d = cohens_d(a, b)
    print(f"  gemma vs {model}: t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}")

sub("bert_n by model  [report: gemma=145.7 vs others=147.4 — similar, ruling out sample-size explanation]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    v = [r["bert_n"] for r in runs if r["model"] == model and r["bert_n"] is not None]
    m, s, n = msd(v)
    if m is not None:
        print(f"  {model}: mean bert_n={m:.1f}±{s:.1f} n={n}")

sub("Overall BERT accuracy  [report: 96.2%, range 80.4%–100%]")
v = [r["bert_accuracy"] for r in runs if r["bert_accuracy"] is not None]
m, s, n = msd(v)
print(f"  mean={m:.4f}±{s:.4f} n={n}, range=[{min(v):.4f},{max(v):.4f}]")

sub("BERT accuracy by question")
for q in [25, 28, 29]:
    v = [r["bert_accuracy"] for r in runs if r["question"] == q and r["bert_accuracy"] is not None]
    m, s, n = msd(v)
    print(f"  Q{q}: mean={m:.4f}±{s:.4f} n={n}")


# ---------------------------------------------------------------------------
# Section 3.5: Proportions option
# ---------------------------------------------------------------------------

section("SECTION 3.5 — PROPORTIONS OPTION")

sub("Net consensus change by proportions_option")
for p_opt in ["uniform", "blueprint", "average", "distribution"]:
    v_c = [r["net_consensus_change"] for r in runs if r["proportions_option"] == p_opt]
    init_m, _, _ = msd([r["initial_consensus"] for r in runs if r["proportions_option"] == p_opt])
    m, s, n = msd(v_c)
    print(f"  {p_opt}: n={n}, init={init_m:.4f}, Δconsensus={m:.4f}±{s:.4f}")

sub("average vs uniform: no longer significant  [report: p=0.29 ns]")
t, p = welch_t(
    [r["net_consensus_change"] for r in runs if r["proportions_option"] == "average"],
    [r["net_consensus_change"] for r in runs if r["proportions_option"] == "uniform"])
print(f"  t={t:.3f}, p={p:.4f} {sig(p)}")

sub("Proportions distribution within each model  [for confound check]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    ctr = collections.Counter(r["proportions_option"] for r in runs if r["model"] == model)
    print(f"  {model}: {dict(ctr)}")

sub("Consensus change by proportions × model  [report table in 3.5]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    vals = []
    for p_opt in ["uniform", "blueprint", "average", "distribution"]:
        v = [r["net_consensus_change"] for r in runs
             if r["model"] == model and r["proportions_option"] == p_opt]
        m, s, n = msd(v)
        vals.append(f"{p_opt}={m:.3f}(n={n})" if m is not None else f"{p_opt}=N/A")
    print(f"  {model}: {', '.join(vals)}")


# ---------------------------------------------------------------------------
# Section 3.6: Echo chamber metrics
# ---------------------------------------------------------------------------

section("SECTION 3.6 — ECHO CHAMBER METRICS")

sub("Overall distributions")
for key, label in [
    ("echo_assortativity", "Assortativity  [report: mean=0.000±0.045]"),
    ("echo_local_agreement", "Local Agreement  [report: mean=0.690±0.156]"),
    ("echo_cross_cutting", "Cross-cutting Edges  [report: mean=0.311±0.156]"),
    ("echo_same_option_exposure", "Same-option Exposure  [report: mean=0.692±0.156]"),
]:
    v = [r[key] for r in runs if r[key] is not None]
    m, s, n = msd(v)
    print(f"  {label}: mean={m:.4f}±{s:.4f} n={n}, range=[{min(v):.4f},{max(v):.4f}]")

sub("Assortativity by homophily  [report: 0.006 vs −0.005, p=0.018*]")
for h in [True, False]:
    v = [r["echo_assortativity"] for r in runs if r["homophily"] == h and r["echo_assortativity"] is not None]
    m, s, n = msd(v)
    print(f"  homophily={h}: mean={m:.4f}±{s:.4f} n={n}")
t, p = welch_t(
    [r["echo_assortativity"] for r in runs if r["homophily"] == True],
    [r["echo_assortativity"] for r in runs if r["homophily"] == False])
print(f"  Welch t={t:.3f}, p={p:.4f} {sig(p)}")

sub("Local agreement by graph_type  [report: powerlaw=0.701 vs random=0.679, p=0.18 ns]")
for gt in ["random", "powerlaw_cluster"]:
    v = [r["echo_local_agreement"] for r in runs if r["graph_type"] == gt and r["echo_local_agreement"] is not None]
    m, s, n = msd(v)
    print(f"  {gt}: mean={m:.4f}±{s:.4f} n={n}")
t, p = welch_t(
    [r["echo_local_agreement"] for r in runs if r["graph_type"] == "powerlaw_cluster"],
    [r["echo_local_agreement"] for r in runs if r["graph_type"] == "random"])
print(f"  Welch t (powerlaw vs random)={t:.3f}, p={p:.4f} {sig(p)}")

sub("Cross-cutting edges by graph_type  [report: random=0.321 vs powerlaw=0.300, p=0.19 ns]")
for gt in ["random", "powerlaw_cluster"]:
    v = [r["echo_cross_cutting"] for r in runs if r["graph_type"] == gt and r["echo_cross_cutting"] is not None]
    m, s, n = msd(v)
    print(f"  {gt}: mean={m:.4f}±{s:.4f} n={n}")
t, p = welch_t(
    [r["echo_cross_cutting"] for r in runs if r["graph_type"] == "random"],
    [r["echo_cross_cutting"] for r in runs if r["graph_type"] == "powerlaw_cluster"])
print(f"  Welch t (random vs powerlaw)={t:.3f}, p={p:.4f} {sig(p)}")

sub("Correlation: same-option exposure vs net_consensus_change  [report: r=+0.54, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["echo_same_option_exposure"] for r in runs],
    [r["net_consensus_change"] for r in runs])
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig(p_val)}, n={n}")


# ---------------------------------------------------------------------------
# Section 3.7: News agents
# ---------------------------------------------------------------------------

section("SECTION 3.7 — NEWS AGENTS (NULL RESULT)")

sub("Key metrics by num_news_agents  [report: all ns, p>0.16]")
for metric, label in [
    ("net_consensus_change", "Δconsensus"),
    ("mean_opinion_shift_rate", "opinion_shift_rate"),
    ("bert_accuracy", "BERT accuracy"),
]:
    for nn in [0, 1]:
        v = [r[metric] for r in runs if r["num_news_agents"] == nn]
        m, s, n = msd(v)
        if m is not None:
            print(f"  {label} news={nn}: mean={m:.4f}±{s:.4f} n={n}")
    t, p = welch_t(
        [r[metric] for r in runs if r["num_news_agents"] == 0],
        [r[metric] for r in runs if r["num_news_agents"] == 1])
    print(f"  {label}: Welch t={t:.3f}, p={p:.4f} {sig(p)}\n")


# ---------------------------------------------------------------------------
# Summary table numbers
# ---------------------------------------------------------------------------

section("SUMMARY TABLE — ALL EFFECT SIZES")

sub("Cohen's d: Minitaur vs Gemma  [report: d=1.06]")
d = cohens_d(
    [r["net_consensus_change"] for r in runs if r["model"] == "minitaur"],
    [r["net_consensus_change"] for r in runs if r["model"] == "gemma"])
print(f"  d={d:.3f}")

sub("Cohen's d: Minitaur vs Llama3.1  [report: d=0.60]")
d = cohens_d(
    [r["net_consensus_change"] for r in runs if r["model"] == "minitaur"],
    [r["net_consensus_change"] for r in runs if r["model"] == "llama3.1"])
print(f"  d={d:.3f}")

sub("Cohen's d: Qwen vs Gemma  [report: d=0.50]")
d = cohens_d(
    [r["net_consensus_change"] for r in runs if r["model"] == "qwen"],
    [r["net_consensus_change"] for r in runs if r["model"] == "gemma"])
print(f"  d={d:.3f}")

sub("Cohen's d: survey_context → majority follow rate  [report: d=0.46]")
d = cohens_d(
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_majority_follow_rate"] for r in runs if r["add_survey_to_context"] == False])
print(f"  d={d:.3f}")

sub("Cohen's d: survey_context → BERT accuracy  [report: d=1.49]")
d = cohens_d(
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == True],
    [r["bert_accuracy"] for r in runs if r["add_survey_to_context"] == False])
print(f"  d={d:.3f}")

sub("r: num_agents vs opinion_shift_rate  [report: r=−0.48]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["mean_opinion_shift_rate"] for r in runs])
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig(p_val)}, n={n}")

sub("r: num_agents vs consensus change  [report: r=0.03, ns]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["net_consensus_change"] for r in runs])
print(f"  r={r_val:.4f}, p={p_val:.4f} {sig(p_val)}, n={n}")

sub("bert_n confound: ctx=True vs False eval thread count")
ctx_t = [r["bert_n"] for r in runs if r["add_survey_to_context"] == True and r["bert_n"] is not None]
ctx_f = [r["bert_n"] for r in runs if r["add_survey_to_context"] == False and r["bert_n"] is not None]
mt, _, nt = msd(ctx_t)
mf, _, nf = msd(ctx_f)
print(f"  ctx=True bert_n: mean={mt:.1f} n={nt}")
print(f"  ctx=False bert_n: mean={mf:.1f} n={nf}")
r_val, p_val, _ = pearson_r([r["bert_n"] for r in runs], [r["bert_accuracy"] for r in runs])
print(f"  bert_n vs bert_accuracy: r={r_val:.4f}, p={p_val:.4f}")

# ---------------------------------------------------------------------------
# Section 3.7: NASR (neighbor_alignment_shift_rate)
# ---------------------------------------------------------------------------

section("SECTION 3.7 — NEIGHBOR ALIGNMENT SHIFT RATE (NASR)")

sub("NASR coverage  [report: 354/354 runs populated]")
has_nasr = [r for r in runs if r["mean_nasr"] is not None]
print(f"  Runs with NASR: {len(has_nasr)}/{len(runs)}")
m, s, n = msd([r["mean_nasr"] for r in runs])
print(f"  Overall mean NASR: {m:.4f}±{s:.4f} n={n}")

sub("NASR vs OSR correlation  [report: r=0.924, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["mean_nasr"] for r in runs],
    [r["mean_osr_from_transitions"] for r in runs])
print(f"  r={r_val:.4f}, p={p_val:.6f} {sig(p_val)}, n={n}")

sub("NASR by model  [report: gemma=0.099, qwen=0.081, llama3.1=0.074, minitaur=0.060]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    v = [r["mean_nasr"] for r in runs if r["model"] == model and r["mean_nasr"] is not None]
    m, s, n = msd(v)
    print(f"  {model}: {m:.4f}±{s:.4f} n={n}")

sub("NASR/OSR ratio by model  [report: ~0.36–0.42]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    sub_runs = [r for r in runs if r["model"] == model
                and r["mean_nasr"] is not None
                and r["mean_osr_from_transitions"] is not None
                and r["mean_osr_from_transitions"] > 0]
    ratios = [r["mean_nasr"] / r["mean_osr_from_transitions"] for r in sub_runs]
    m, s, n = msd(ratios)
    print(f"  {model}: ratio={m:.3f}±{s:.3f} n={n}")

sub("NASR pairwise tests  [report: all sig except llama3.1 vs qwen]")
for m1, m2, expected in [
    ("gemma", "minitaur", "d=1.10, p<0.001"),
    ("gemma", "llama3.1", "d=0.71, p<0.001"),
    ("gemma", "qwen",     "d=0.49, p=0.004"),
    ("qwen",  "minitaur", "d=0.56, p<0.001"),
    ("llama3.1", "minitaur", "d=0.38, p=0.008"),
    ("qwen",  "llama3.1", "ns"),
]:
    a = [r["mean_nasr"] for r in runs if r["model"] == m1 and r["mean_nasr"] is not None]
    b = [r["mean_nasr"] for r in runs if r["model"] == m2 and r["mean_nasr"] is not None]
    t, p = welch_t(a, b)
    d = cohens_d(a, b)
    ma, _, na_ = msd(a)
    mb, _, nb = msd(b)
    print(f"  {m1}({ma:.4f},n={na_}) vs {m2}({mb:.4f},n={nb}): "
          f"t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}  [expected: {expected}]")

sub("NASR vs num_agents  [report: r=−0.42, p<0.001]")
r_val, p_val, n = pearson_r(
    [r["num_agents"] for r in runs],
    [r["mean_nasr"] for r in runs])
print(f"  r={r_val:.4f}, p={p_val:.6f} {sig(p_val)}, n={n}")
for na in [64, 256, 1024, 4096]:
    v = [r["mean_nasr"] for r in runs if r["num_agents"] == na and r["mean_nasr"] is not None]
    m, s, n = msd(v)
    print(f"  na={na}: {m:.4f}±{s:.4f} n={n}")

sub("Survey context → NASR (global)  [report: p=0.007, d=0.285]")
for ctx in [True, False]:
    v = [r["mean_nasr"] for r in runs if r["add_survey_to_context"] == ctx and r["mean_nasr"] is not None]
    m, s, n = msd(v)
    print(f"  ctx={ctx}: {m:.4f}±{s:.4f} n={n}")
t, p = welch_t(
    [r["mean_nasr"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_nasr"] for r in runs if r["add_survey_to_context"] == False])
d = cohens_d(
    [r["mean_nasr"] for r in runs if r["add_survey_to_context"] == True],
    [r["mean_nasr"] for r in runs if r["add_survey_to_context"] == False])
print(f"  t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}")

sub("Survey context → NASR within each model  [report: all ns within model]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    for ctx in [True, False]:
        v = [r["mean_nasr"] for r in runs
             if r["model"] == model and r["add_survey_to_context"] == ctx and r["mean_nasr"] is not None]
        m, s, n = msd(v)
        print(f"  {model} ctx={ctx}: {m:.4f}±{s:.4f} n={n}")
    t, p = welch_t(
        [r["mean_nasr"] for r in runs if r["model"] == model and r["add_survey_to_context"] == True],
        [r["mean_nasr"] for r in runs if r["model"] == model and r["add_survey_to_context"] == False])
    print(f"  {model}: t={t:.3f}, p={p:.4f} {sig(p)}")


# ---------------------------------------------------------------------------
# Section 3.6 time series: Echo chamber temporal dynamics
# ---------------------------------------------------------------------------

section("SECTION 3.6 TIME SERIES — ECHO CHAMBER DYNAMICS")

ECHO_KEYS = {
    "network_assortativity": "assortativity",
    "mean_local_agreement": "local_agreement",
    "cross_cutting_edge_fraction": "cross_cutting",
    "mean_same_option_exposure_share": "same_option_exposure",
}

sub("by_survey coverage  [report: 354/354 runs, 11 steps each]")
has_bs = [r for r in runs if r["by_survey"]]
print(f"  Runs with by_survey: {len(has_bs)}/{len(runs)}")
if has_bs:
    lengths = [len(r["by_survey"]) for r in has_bs]
    print(f"  by_survey lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mode={max(set(lengths), key=lengths.count)}")

sub("First vs last survey step for each metric  [report table]")
# Build per-run deltas for further tests
deltas: dict = {lbl: [] for lbl in ECHO_KEYS.values()}

for raw_key, label in ECHO_KEYS.items():
    first_vals, last_vals = [], []
    for r in runs:
        bs = r["by_survey"]
        if not bs:
            continue
        v0 = bs[0].get(raw_key)
        v1 = bs[-1].get(raw_key)
        if v0 is not None:
            first_vals.append(v0)
        if v1 is not None:
            last_vals.append(v1)
        if v0 is not None and v1 is not None:
            deltas[label].append({
                "delta": v1 - v0,
                "homophily": r["homophily"],
                "graph_type": r["graph_type"],
                "net_consensus_change": r["net_consensus_change"],
            })
    m0, s0, n0 = msd(first_vals)
    m1, s1, n1 = msd(last_vals)
    delta_vals = [x["delta"] for x in deltas[label]]
    md, sd, nd = msd(delta_vals)
    t_zero = md / (sd / math.sqrt(nd)) if nd > 1 and sd > 0 else None
    p_zero = 2 * (1 - 0.5 * (1 + math.erf(abs(t_zero) / math.sqrt(2)))) if t_zero else None
    print(f"  {label}:")
    print(f"    first={m0:.4f}±{s0:.4f} (n={n0}), last={m1:.4f}±{s1:.4f} (n={n1})")
    print(f"    Δ={md:.4f}±{sd:.4f} (n={nd}), p_vs_zero={p_zero:.4f} {sig(p_zero)}")

sub("Homophily → Δ assortativity  [report: True=−0.049, False=+0.002, p<0.001, d=−0.636]")
dh_t = [x["delta"] for x in deltas["assortativity"] if x["homophily"] == True]
dh_f = [x["delta"] for x in deltas["assortativity"] if x["homophily"] == False]
mht, sht, nht = msd(dh_t)
mhf, shf, nhf = msd(dh_f)
t, p = welch_t(dh_t, dh_f)
d = cohens_d(dh_t, dh_f)
print(f"  homophily=True: {mht:.4f}±{sht:.4f} n={nht}")
print(f"  homophily=False: {mhf:.4f}±{shf:.4f} n={nhf}")
print(f"  t={t:.3f}, p={p:.4f} {sig(p)}, d={d:.3f}")

sub("Initial assortativity by homophily  [report: True=0.055, False=−0.007]")
for h in [True, False]:
    vals = [r["by_survey"][0].get("network_assortativity")
            for r in runs if r["by_survey"] and r["homophily"] == h]
    m, s, n = msd([v for v in vals if v is not None])
    print(f"  homophily={h}: initial assortativity={m:.4f}±{s:.4f} n={n}")

sub("Graph type → echo deltas  [report: all ns, p>0.06]")
for label in ["assortativity", "local_agreement", "cross_cutting", "same_option_exposure"]:
    vals_r = [x["delta"] for x in deltas[label] if x["graph_type"] == "random"]
    vals_p = [x["delta"] for x in deltas[label] if x["graph_type"] == "powerlaw_cluster"]
    mr, _, nr = msd(vals_r)
    mp, _, np_ = msd(vals_p)
    t, p = welch_t(vals_r, vals_p)
    print(f"  {label}: random={mr:.4f}(n={nr}), powerlaw={mp:.4f}(n={np_}), p={p:.4f} {sig(p)}")

sub("Δ local_agreement vs net_consensus_change  [report: r=0.915, p<0.001]")
la_deltas = [x["delta"] for x in deltas["local_agreement"]]
la_ncc = [x["net_consensus_change"] for x in deltas["local_agreement"]]
r_val, p_val, n = pearson_r(la_deltas, la_ncc)
print(f"  r={r_val:.4f}, p={p_val:.6f} {sig(p_val)}, n={n}")

sub("Δ assortativity vs net_consensus_change  [report: r=−0.121, p=0.022*]")
as_deltas = [x["delta"] for x in deltas["assortativity"]]
as_ncc = [x["net_consensus_change"] for x in deltas["assortativity"]]
r_val, p_val, n = pearson_r(as_deltas, as_ncc)
print(f"  r={r_val:.4f}, p={p_val:.6f} {sig(p_val)}, n={n}")


# ---------------------------------------------------------------------------
# Section 3.9: Parameter interactions
# ---------------------------------------------------------------------------

section("SECTION 3.9 — PARAMETER INTERACTIONS")


def interaction_2x2(label_a, label_b, metric_label, a_fn, b_fn, metric_fn, filter_fn=None):
    """Compute 2x2 interaction contrast IC = mean_11 - mean_10 - mean_01 + mean_00."""
    if filter_fn is None:
        filter_fn = lambda r: True
    cells = {}
    for av in [False, True]:
        for bv in [False, True]:
            vals = [metric_fn(r) for r in runs
                    if filter_fn(r) and a_fn(r) == av and b_fn(r) == bv
                    and metric_fn(r) is not None]
            cells[(av, bv)] = vals
    means = {k: statistics.mean(v) if v else None for k, v in cells.items()}
    ns = {k: len(v) for k, v in cells.items()}
    if any(v is None for v in means.values()) or any(n < 2 for n in ns.values()):
        print(f"  [{metric_label}] {label_a} x {label_b}: insufficient data")
        return None
    ic = means[(True, True)] - means[(True, False)] - means[(False, True)] + means[(False, False)]
    eff_b_a0 = means[(False, True)] - means[(False, False)]
    eff_b_a1 = means[(True, True)] - means[(True, False)]
    print(f"  [{metric_label}] {label_a} x {label_b}:")
    print(f"    {label_a}=F/{label_b}=F={means[(False,False)]:.4f}(n={ns[(False,False)]})  "
          f"{label_a}=F/{label_b}=T={means[(False,True)]:.4f}(n={ns[(False,True)]})")
    print(f"    {label_a}=T/{label_b}=F={means[(True,False)]:.4f}(n={ns[(True,False)]})  "
          f"{label_a}=T/{label_b}=T={means[(True,True)]:.4f}(n={ns[(True,True)]})")
    print(f"    Effect of {label_b} given {label_a}=F: {eff_b_a0:+.4f}")
    print(f"    Effect of {label_b} given {label_a}=T: {eff_b_a1:+.4f}")
    print(f"    Interaction contrast (IC): {ic:+.4f}")
    return ic


def eta_sq_single(group_fn, metric_fn, label):
    vals_all = [metric_fn(r) for r in runs if metric_fn(r) is not None]
    if len(vals_all) < 5:
        return None
    grand = statistics.mean(vals_all)
    ss_total = sum((v - grand) ** 2 for v in vals_all)
    if ss_total == 0:
        return None
    groups = collections.defaultdict(list)
    for r in runs:
        m = metric_fn(r)
        g = group_fn(r)
        if m is not None and g is not None:
            groups[g].append(m)
    ss_between = sum(len(v) * (statistics.mean(v) - grand) ** 2 for v in groups.values() if v)
    eta2 = ss_between / ss_total
    print(f"    {label}: η²={eta2:.3f}")
    return eta2


# Compute per-run d_assort (for interaction tests)
for r in runs:
    bs = r["by_survey"]
    v0 = bs[0].get("network_assortativity") if bs else None
    v1 = bs[-1].get("network_assortativity") if bs else None
    r["d_assort"] = (v1 - v0) if v0 is not None and v1 is not None else None


sub("Variance decomposition (η²) — net_consensus_change  [report: model=0.075, ctx=0.053]")
for param, fn in [
    ("model",     lambda r: r["model"]),
    ("num_agents", lambda r: r["num_agents"]),
    ("ctx",        lambda r: r["add_survey_to_context"]),
    ("homophily",  lambda r: r["homophily"]),
    ("graph_type", lambda r: r["graph_type"]),
    ("prop_opt",   lambda r: r["proportions_option"]),
    ("num_news",   lambda r: r["num_news_agents"]),
    ("question",   lambda r: r["question"]),
]:
    eta_sq_single(fn, lambda r: r["net_consensus_change"], param)

sub("Variance decomposition (η²) — opinion_shift_rate  [report: num_agents=0.346, model=0.076]")
for param, fn in [
    ("model",     lambda r: r["model"]),
    ("num_agents", lambda r: r["num_agents"]),
    ("ctx",        lambda r: r["add_survey_to_context"]),
    ("homophily",  lambda r: r["homophily"]),
    ("graph_type", lambda r: r["graph_type"]),
    ("prop_opt",   lambda r: r["proportions_option"]),
    ("question",   lambda r: r["question"]),
]:
    eta_sq_single(fn, lambda r: r["mean_opinion_shift_rate"], param)

sub("Variance decomposition (η²) — BERT accuracy  [report: ctx=0.357, model=0.126]")
for param, fn in [
    ("model",     lambda r: r["model"]),
    ("ctx",        lambda r: r["add_survey_to_context"]),
    ("num_agents", lambda r: r["num_agents"]),
    ("homophily",  lambda r: r["homophily"]),
    ("question",   lambda r: r["question"]),
]:
    eta_sq_single(fn, lambda r: r["bert_accuracy"], param)

sub("Variance decomposition (η²) — Δ_assortativity  [report: homophily=0.092, num_agents=0.041]")
for param, fn in [
    ("homophily",  lambda r: r["homophily"]),
    ("model",      lambda r: r["model"]),
    ("graph_type", lambda r: r["graph_type"]),
    ("num_agents", lambda r: r["num_agents"]),
    ("ctx",        lambda r: r["add_survey_to_context"]),
]:
    eta_sq_single(fn, lambda r: r.get("d_assort"), param)

sub("2x2 interaction contrasts  [report: homophily×powerlaw IC=+0.038, ctx×news IC=−0.006]")
interaction_2x2("ctx", "homophily", "opinion_shift_rate",
                lambda r: r["add_survey_to_context"], lambda r: r["homophily"],
                lambda r: r["mean_opinion_shift_rate"])
interaction_2x2("ctx", "homophily", "net_consensus_change",
                lambda r: r["add_survey_to_context"], lambda r: r["homophily"],
                lambda r: r["net_consensus_change"])
interaction_2x2("ctx", "powerlaw", "net_consensus_change",
                lambda r: r["add_survey_to_context"], lambda r: r["graph_type"] == "powerlaw_cluster",
                lambda r: r["net_consensus_change"])
interaction_2x2("homophily", "powerlaw", "Δ_assortativity",
                lambda r: r["homophily"], lambda r: r["graph_type"] == "powerlaw_cluster",
                lambda r: r.get("d_assort"))
interaction_2x2("ctx", "news", "bert_accuracy",
                lambda r: r["add_survey_to_context"], lambda r: r["num_news_agents"] == 1,
                lambda r: r["bert_accuracy"])

sub("Model x ctx → net_consensus_change  [report: Llama/Qwen diff=+0.128***, Minitaur/Gemma ns]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    vt = [r["net_consensus_change"] for r in runs
          if r["model"] == model and r["add_survey_to_context"] == True]
    vf = [r["net_consensus_change"] for r in runs
          if r["model"] == model and r["add_survey_to_context"] == False]
    mt, _, nt = msd(vt)
    mf, _, nf = msd(vf)
    t, p = welch_t(vt, vf)
    if mt is not None and mf is not None:
        print(f"  {model}: ctx=T={mt:.4f}(n={nt}) ctx=F={mf:.4f}(n={nf}) diff={mt-mf:+.4f} "
              f"t={t:.3f}, p={p:.4f} {sig(p)}")

sub("ctx × num_agents → majority_follow_rate  [report: ns at 64; p<0.001 at 1024 and 4096]")
for na in [64, 256, 1024, 4096]:
    vt = [r["mean_majority_follow_rate"] for r in runs
          if r["num_agents"] == na and r["add_survey_to_context"] == True]
    vf = [r["mean_majority_follow_rate"] for r in runs
          if r["num_agents"] == na and r["add_survey_to_context"] == False]
    mt, _, nt = msd(vt)
    mf, _, nf = msd(vf)
    t, p = welch_t(vt, vf)
    if mt is not None and mf is not None:
        print(f"  na={na}: ctx=T={mt:.4f}(n={nt}) ctx=F={mf:.4f}(n={nf}) diff={mt-mf:+.4f} "
              f"p={p:.4f} {sig(p)}")

sub("num_agents × model → OSR: additive check  [report: all r negative, all p<0.001]")
for model in ["minitaur", "llama3.1", "qwen", "gemma"]:
    xs = [r["num_agents"] for r in runs if r["model"] == model]
    ys = [r["mean_opinion_shift_rate"] for r in runs if r["model"] == model]
    r_val, p_val, n = pearson_r(xs, ys)
    if r_val is not None:
        print(f"  {model}: r={r_val:.4f}, p={p_val:.4f} {sig(p_val)}, n={n}")

sub("homophily × graph_type → Δ_assortativity  [report: IC=+0.038, effect concentrated in homophilic]")
for h in [True, False]:
    for gt in ["random", "powerlaw_cluster"]:
        vals = [r["d_assort"] for r in runs
                if r["homophily"] == h and r["graph_type"] == gt and r.get("d_assort") is not None]
        m, s, n = msd(vals)
        if m is not None:
            print(f"  homophily={h}, {gt}: Δassort={m:.4f}±{s:.4f} n={n}")


print("\n" + "=" * 72)
print("  DONE")
print("=" * 72)
