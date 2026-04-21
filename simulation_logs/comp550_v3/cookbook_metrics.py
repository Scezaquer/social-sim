#!/usr/bin/env python3
"""
Cookbook Table 12 metrics analysis.
Computes all six behavioral metrics from
Silicon Society Cookbook Table 12, grouped by each independent variable.

Run from simulation_logs/comp550/.
"""

import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

# Helpers: entirely copied from reports/verify_report_numbers.py

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


# End of copied helpers



def load_all_runs():
    """Load all 720 JSONs."""
    files = [f for f in os.listdir(".") if f.startswith("visualizer_comp550_") and f.endswith(".json")]
    runs = []

    for f in files:
        with open(f) as fh:
            d = json.load(fh)

        rp = d["run_parameters"]
        bm = d.get("behavioral_metrics", {})
        herd = bm.get("herd_effect_metrics", {})
        echo = bm.get("echo_chamber_metrics", {})
        user_herd = bm.get("user_only_herd_effect_metrics", {})

        user_transitions = user_herd.get("transitions", []) if isinstance(user_herd, dict) else []

        def mean_transition_value(key: str) -> Optional[float]:
            vals = [t.get(key) for t in user_transitions if isinstance(t, dict)]
            vals = [x for x in vals if isinstance(x, (int, float))]
            return statistics.mean(vals) if vals else None

        runs.append({
            "question":          rp["question_number"],
            "strategy":          rp["adversarial_strategy"],
            "centralize":        rp["centralize_adversaries"],
            "proportion":        rp["proportion_adversarial_agents"],
            "num_agents":        rp["num_agents"],
            "osr":               herd.get("mean_opinion_shift_rate"), # Can also look at osr between transitions.
            "ncc":               herd.get("net_consensus_change"), 
            "initial_consensus": herd.get("initial_consensus"),
            "final_consensus":   herd.get("final_consensus"),
            "mfr":               herd.get("mean_current_majority_follow_rate"), # Can also look at mfr between transitions.
            "nasr":              herd.get("mean_neighbor_alignment_shift_rate"), # Can also look at nasr between transitions.
            "assortativity":     echo.get("network_assortativity"), # Can also look at assortativity for the surveys.
            "u_osr":             user_herd.get("mean_opinion_shift_rate") if isinstance(user_herd, dict) else None,
            "u_ncc":             user_herd.get("net_consensus_change") if isinstance(user_herd, dict) else None,
            "u_initial_consensus": user_herd.get("initial_consensus") if isinstance(user_herd, dict) else None,
            "u_final_consensus": user_herd.get("final_consensus") if isinstance(user_herd, dict) else None,
            "u_mfr":             user_herd.get("mean_current_majority_follow_rate") if isinstance(user_herd, dict) else None,
            "u_nasr":            user_herd.get("mean_neighbor_alignment_shift_rate") if isinstance(user_herd, dict) else None,
            "u_initial_diversity": user_herd.get("initial_diversity") if isinstance(user_herd, dict) else None,
            "u_final_diversity": user_herd.get("final_diversity") if isinstance(user_herd, dict) else None,
            "u_mean_consensus_gain": user_herd.get("mean_consensus_gain") if isinstance(user_herd, dict) else None,
            "u_mean_changed_users": mean_transition_value("changed_users"),
            "u_mean_shared_users": mean_transition_value("shared_users"),
        })

    return runs


def print_table(rows, headers):
    """Print rows as tab-separated values with a header line."""
    print("\t".join(str(h) for h in headers))
    for row in rows:
        print("\t".join(str(cell) for cell in row))


def split_by_level(runs, metric_key, iv_key, levels):
    """Split metric values into separate lists, one per IV level.

    Example: split_by_level(runs, "osr", "question", [13, 23, 35])
    returns {13: [list of 240 osr values], 23: [...], 35: [...]}
    """
    groups = {}
    for lev in levels:
        groups[lev] = []
    for r in runs:
        lev = r[iv_key]
        groups[lev].append(r[metric_key])
    return groups


def eta_squared(group_lists):
    """Compute eta squared (ss_between / ss_total)."""
    vals_all = []
    for g in group_lists:
        for x in g:
            if x is not None:
                vals_all.append(x)
    if len(vals_all) < 5:
        return None
    grand = statistics.mean(vals_all)
    ss_total = sum((v - grand) ** 2 for v in vals_all)
    if ss_total == 0:
        return None
    ss_between = sum(len(v) * (statistics.mean(v) - grand) ** 2 for v in group_lists if v)
    return ss_between / ss_total


def main():
    print("Loading JSON files...")
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs.\n")

    metrics = [ # metric_key, metric_label
        # Existing (all-users) metrics
        ("osr", "Opinion Shift Rate (OSR)"),
        ("ncc", "Net Consensus Change (NCC)"),
        ("initial_consensus", "Initial Consensus"),
        ("final_consensus", "Final Consensus"),
        ("mfr", "Majority Follow Rate (MFR)"),
        ("nasr", "Neighbor Alignment Shift Rate (NASR)"),
        ("assortativity", "Network Assortativity"),

        # Additional user-only herd-effect metrics
        ("u_osr", "User-only Opinion Shift Rate (OSR)"),
        ("u_ncc", "User-only Net Consensus Change (NCC)"),
        ("u_initial_consensus", "User-only Initial Consensus"),
        ("u_final_consensus", "User-only Final Consensus"),
        ("u_mfr", "User-only Majority Follow Rate (MFR)"),
        ("u_nasr", "User-only Neighbor Alignment Shift Rate (NASR)"),
        ("u_mean_consensus_gain", "User-only Mean Consensus Gain"),
        ("u_initial_diversity", "User-only Initial Diversity"),
        ("u_final_diversity", "User-only Final Diversity"),
        ("u_mean_changed_users", "User-only Mean Changed Users per Transition"),
        ("u_mean_shared_users", "User-only Mean Shared Users per Transition"),
    ]

    ivs = [ # iv_key, iv_label, levels
        ("question", "Question", [25, 28, 29]),
        ("strategy", "Adversarial Strategy", ["false_information", "red_teaming"]),
        ("centralize", "Centralize Adversaries", [False, True]),
        ("proportion", "Adversarial Proportion", [0.0, 0.0625, 0.125, 0.25]),
        ("num_agents", "Number of Agents", [64, 128, 256]),
    ]

    numeric_ivs = {"proportion", "num_agents"}

    # Per-metric analysis
    for metric_key, metric_label in metrics:
        section(metric_label)

        for iv_key, iv_label, levels in ivs:
            sub(f"Grouped by {iv_label}")

            groups = split_by_level(runs, metric_key, iv_key, levels)

            # Group means table
            rows = []
            for lev in levels:
                m, s, n = msd(groups[lev])
                rows.append([
                    str(lev),
                    str(n),
                    f"{m:.6f}" if m is not None else "N/A",
                    f"{s:.6f}" if s is not None else "N/A",
                ])
            print_table(rows, ["Level", "n", "Mean", "SD"])

            # Pearson r for numeric IVs
            if iv_key in numeric_ivs:
                xs = [r[iv_key] for r in runs]
                ys = [r[metric_key] for r in runs]
                r_val, p_val, n = pearson_r(xs, ys)
                if r_val is not None:
                    print(f"\n  Pearson r = {r_val:.4f}, p = {p_val:.6f} {sig(p_val)}, n = {n}")

            # eta squared
            group_lists = [groups[lev] for lev in levels]
            eta2 = eta_squared(group_lists)
            if eta2 is not None:
                print(f"\n  eta squared = {eta2:.6f}")

            if len(levels) == 2:
                a = groups[levels[0]]
                b = groups[levels[1]]
                t, p = welch_t(a, b)
                d = cohens_d(a, b)
                if t is not None:
                    print(f"\n  Welch t = {t:.4f}, p = {p:.6f} {sig(p)}")
                if d is not None:
                    print(f"  Cohen's d = {d:.4f}")


if __name__ == "__main__":
    main()