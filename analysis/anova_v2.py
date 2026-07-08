#!/usr/bin/env python3
"""Corrected variance-attribution analysis for the V2 design-space sweep.

Implements the statistical protocol promised to reviewers (R3-W3, R4-W3):
1. Type-III ANOVA with sum-to-zero contrasts over all main effects and all
   two-way interactions, reporting PARTIAL eta^2 (not single-factor eta^2 on
   unbalanced data, which let `model` absorb correlated choices in V1).
2. Mixed-effects model per (metric, factor): factor fixed effects with the
   SLURM batch (job_id) as a random intercept.
3. Benjamini-Hochberg FDR q-values within each declared test family
   (family = one outcome metric).
4. Run-level bootstrap percentile CIs for every partial eta^2.

Input: the tidy CSV from analysis/build_dataset.py.
Output: <outdir>/anova_v2.json + <outdir>/anova_v2.md (ranked tables).

Usage:
    python analysis/anova_v2.py --dataset v2_runs.csv --outdir analysis_out \
        [--experiments e1_core e2_base_vs_lora] [--bootstrap 1000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("This script requires statsmodels and pandas (pip install statsmodels pandas).", file=sys.stderr)
    raise

# Design factors treated as categorical. `model_family` is the base model
# (e.g. Qwen2.5-7B-Instruct) WITHOUT its fine-tuning status, so it is not
# aliased with `lora_finetuned`, which is only identifiable when E2 runs are
# pooled in alongside E1.
FACTORS = [
    "model_family",
    "lora_finetuned",
    "proportions_option",
    "question_number",
    "num_agents",
    "graph_type",
    "homophily",
    "add_survey_to_context",
    "num_news_agents",
    "activity_exponent",
]

METRICS = [
    "net_consensus_change",
    "mean_opinion_shift_rate",
    "mean_current_majority_follow_rate",
    "mean_neighbor_alignment_shift_rate",
    "delta_assortativity",
    "mean_local_agreement",
    "cross_cutting_edge_fraction",
    "order_consistency_rate",
    "bert_accuracy",
]


def usable_factors(df: pd.DataFrame) -> list[str]:
    """Factors with >1 level in this dataset (drops e.g. lora_finetuned on E1-only)."""
    out = []
    for f in FACTORS:
        if f in df.columns and df[f].nunique(dropna=True) > 1:
            out.append(f)
    return out


def build_formula(metric: str, factors: list[str], interactions: bool) -> str:
    terms = [f"C({f}, Sum)" for f in factors]
    if interactions:
        terms += [
            f"C({a}, Sum):C({b}, Sum)"
            for a, b in itertools.combinations(factors, 2)
        ]
    return f"{metric} ~ " + " + ".join(terms)


def type3_partial_eta2(df: pd.DataFrame, metric: str, factors: list[str], interactions: bool) -> pd.DataFrame:
    model = smf.ols(build_formula(metric, factors, interactions), data=df).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anova_full = sm.stats.anova_lm(model, typ=3)

    # Residual sum of squares for partial eta^2 denominator
    ss_resid = anova_full.loc["Residual", "sum_sq"]

    # Total sum of squares (exclude the Intercept row if present)
    intercept_ss = float(anova_full.loc["Intercept", "sum_sq"]) if "Intercept" in anova_full.index else 0.0
    ss_total = float(anova_full["sum_sq"].sum() - intercept_ss)

    # Drop intercept and residual rows for per-term reporting
    table = anova_full.drop(index=[i for i in ("Intercept", "Residual") if i in anova_full.index], errors="ignore")
    table["partial_eta2"] = table["sum_sq"] / (table["sum_sq"] + ss_resid)
    table["eta2"] = table["sum_sq"] / ss_total
    return table


def bootstrap_eta2_ci(df, metric, factors, interactions, n_boot, rng, alpha=0.05):
    """Run-level bootstrap percentile CI for each term's partial eta^2."""
    stats: dict[str, list[float]] = {}
    for _ in range(n_boot):
        sample = df.sample(n=len(df), replace=True, random_state=rng.integers(0, 2**31 - 1))
        try:
            table = type3_partial_eta2(sample, metric, factors, interactions)
        except Exception:
            continue
        for term, value in table["partial_eta2"].items():
            stats.setdefault(term, []).append(float(value))
    cis = {}
    for term, values in stats.items():
        if len(values) >= max(50, n_boot // 4):
            cis[term] = [
                float(np.percentile(values, 100 * alpha / 2)),
                float(np.percentile(values, 100 * (1 - alpha / 2))),
            ]
    return cis


def mixed_effects_check(df: pd.DataFrame, metric: str, factors: list[str]) -> dict:
    """Refits main effects with job batch as a random intercept; reports Wald p
    per factor so batch-correlated noise cannot masquerade as a design effect."""
    out = {}
    data = df.dropna(subset=[metric] + factors + ["job_id"]).copy()
    if data["job_id"].nunique() < 2:
        return {"status": "single batch, mixed model skipped"}
    formula = f"{metric} ~ " + " + ".join(f"C({f}, Sum)" for f in factors)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, data, groups=data["job_id"]).fit(reml=True, method="lbfgs")
        for f in factors:
            terms = [t for t in model.pvalues.index if t.startswith(f"C({f}, Sum)")]
            if terms:
                out[f] = {"min_p": float(min(model.pvalues[t] for t in terms))}
        out["group_var"] = float(model.cov_re.iloc[0, 0]) if model.cov_re.size else None
    except Exception as e:
        out["status"] = f"mixed model failed: {e}"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="v2_runs.csv")
    parser.add_argument("--outdir", default="analysis_out")
    parser.add_argument("--experiments", nargs="*", default=["e1_core", "e2_base_vs_lora"],
                        help="Experiments to pool (default: E1 + E2 so lora_finetuned is crossed with model).")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR level for BH correction.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    full = pd.read_csv(args.dataset)
    full = full[full["stimulus_mode"].fillna("normal") == "normal"]
    if args.experiments:
        full = full[full["experiment"].isin(args.experiments)]
    if "run_complete" in full.columns:
        n_incomplete = int((~full["run_complete"].astype(bool)).sum())
        if n_incomplete:
            print(f"Dropping {n_incomplete} incomplete runs.")
        full = full[full["run_complete"].astype(bool)]

    # Two passes, because proportions_option='none' perfectly aliases
    # lora_finetuned=False when E2 is pooled in (rank-deficient otherwise):
    #  - 'e1_only': proportions identifiable, LoRA status constant;
    #  - 'pooled':  LoRA status crossed with model (the V1 confound fix),
    #               proportions dropped.
    passes = []
    e1 = full[full["experiment"] == "e1_core"]
    if len(e1):
        passes.append(("e1_only", e1, ["lora_finetuned"]))
    if full["lora_finetuned"].nunique() > 1:
        passes.append(("pooled", full, ["proportions_option"]))

    rng = np.random.default_rng(args.seed)
    for pass_name, df, excluded in passes:
        run_pass(pass_name, df, excluded, args, rng)


def run_pass(pass_name, df, excluded_factors, args, rng):
    print(f"\n######## Pass: {pass_name} ({len(df)} runs) ########")
    factors = [f for f in usable_factors(df) if f not in excluded_factors]
    print(f"Active factors: {factors}")

    results = {"pass": pass_name, "n_runs": int(len(df)), "factors": factors, "metrics": {}}

    for metric in METRICS:
        if metric not in df.columns or df[metric].notna().sum() < 50:
            print(f"Skipping {metric}: insufficient data")
            continue
        data = df.dropna(subset=[metric] + factors).copy()
        print(f"\n=== {metric} (n={len(data)}) ===")

        try:
            table = type3_partial_eta2(data, metric, factors, interactions=True)
        except Exception as e:
            print(f"  full model failed ({e}); falling back to main effects only")
            table = type3_partial_eta2(data, metric, factors, interactions=False)

        # BH-FDR within the family = all terms tested for this metric.
        pvals = table["PR(>F)"].values
        reject, qvals, _, _ = multipletests(pvals, alpha=args.alpha, method="fdr_bh")
        table["q_value"] = qvals
        table["significant_fdr"] = reject

        cis = bootstrap_eta2_ci(data, metric, factors, True, args.bootstrap, rng) if args.bootstrap else {}
        mixed = mixed_effects_check(data, metric, factors)

        entry = {"n": int(len(data)), "terms": {}, "mixed_effects": mixed}
        for term, row in table.sort_values("partial_eta2", ascending=False).iterrows():
            entry["terms"][str(term)] = {
                "partial_eta2": float(row["partial_eta2"]),
                "eta2": float(row["eta2"]),
                "partial_eta2_ci95": cis.get(str(term)),
                "F": float(row["F"]),
                "df": float(row["df"]),
                "p": float(row["PR(>F)"]),
                "q_bh": float(row["q_value"]),
                "significant_fdr": bool(row["significant_fdr"]),
            }
        results["metrics"][metric] = entry

        top = table.sort_values("partial_eta2", ascending=False).head(5)
        for term, row in top.iterrows():
            print(
                f"  {term:55s} pEta2={row['partial_eta2']:.3f} eta2={row['eta2']:.3f} q={row['q_value']:.2g}"
            )

    json_path = os.path.join(args.outdir, f"anova_v2_{pass_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(args.outdir, f"anova_v2_{pass_name}.md")
    with open(md_path, "w") as f:
        f.write(f"# V2 Type-III ANOVA — pass `{pass_name}` (eta^2, partial eta^2, BH-FDR, bootstrap CIs)\n\n")
        f.write(f"Runs: {results['n_runs']}; factors: {', '.join(factors)}\n\n")
        for metric, entry in results["metrics"].items():
            f.write(f"## {metric} (n={entry['n']})\n\n")
            f.write("| Term | eta^2 | partial eta^2 | 95% CI | F | p | q (BH) | sig. |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for term, t in entry["terms"].items():
                ci = t["partial_eta2_ci95"]
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
                f.write(
                    f"| {term} | {t['eta2']:.3f} | {t['partial_eta2']:.3f} | {ci_str} | {t['F']:.2f} "
                    f"| {t['p']:.2g} | {t['q_bh']:.2g} | {'yes' if t['significant_fdr'] else ''} |\n"
                )
            f.write("\n")
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
