"""Analysis of the cross-simulator replication (OASIS / Concordia / SiliSocS).

Computes, per run, the paper's headline survey-response metrics with the same
definitions as src/simulation_components/metrics.py (OSR, MFR, NASR, NCC), the
dual-order consistency rate, margin-stratified flips, and the response-shuffle
null for NASR (same procedure as reports/response_shuffle_null.py). Then tests
each paper claim per simulator.

Usage:
    python crosssim/analyze.py --root "$CROSSSIM_OUT" --out crosssim/results
"""
from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["OSR", "MFR", "NASR", "NCC"]
SIM_LABEL = {"oasis": "OASIS", "concordia": "Concordia", "silisocs": "SiliSocS"}
FAM_LABEL = {"qwen": "Qwen2.5-7B", "minitaur": "Llama-3.1-Minitaur-8B"}
GRAPH_LABEL = {"random": "Erd\\H{o}s--R\\'enyi", "barabasi_albert": "Barab\\'asi--Albert", "cycle": "cycle"}


# ---------------------------------------------------------------- metrics
def neighbor_majority(i, prev, adj):
    opts = [prev[j] for j in adj[i] if j in prev]
    if not opts:
        return None
    ranked = Counter(opts).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def herd_metrics(snaps: list[dict[int, str]], adj: dict[int, set[int]]) -> dict:
    """snaps: list of {agent_idx: choice} in survey order. Mirrors compute_herd_effect_metrics."""
    osr, mfr, nasr = [], [], []
    for prev, curr in zip(snaps[:-1], snaps[1:]):
        shared = set(prev) & set(curr)
        if not shared:
            continue
        changed = [a for a in shared if prev[a] != curr[a]]
        cc = Counter(curr[a] for a in shared)
        cmaj = max(cc.items(), key=lambda kv: kv[1])[0]
        moved_maj = sum(1 for a in changed if curr[a] == cmaj)
        to_nb = 0
        prev_shared = {a: prev[a] for a in shared}
        for a in shared:
            nb = neighbor_majority(a, prev_shared, adj)
            if nb is not None and prev[a] != curr[a] and curr[a] == nb:
                to_nb += 1
        n = len(shared)
        osr.append(len(changed) / n)
        mfr.append(moved_maj / len(changed) if changed else 0.0)
        nasr.append(to_nb / n)
    if not osr:
        return {k: float("nan") for k in METRICS}

    def cons(s):
        c = Counter(s.values())
        return max(c.values()) / max(1, sum(c.values())) if c else 0.0

    return {"OSR": float(np.mean(osr)), "MFR": float(np.mean(mfr)), "NASR": float(np.mean(nasr)),
            "NCC": cons(snaps[-1]) - cons(snaps[0])}


def shuffled(snaps, n_agents, rng):
    perm = list(range(n_agents))
    rng.shuffle(perm)
    return [{perm[a]: c for a, c in s.items()} for s in snaps]


def null_test(snaps, adj, n_agents, n_perm, seed):
    real = herd_metrics(snaps, adj)["NASR"]
    rng = random.Random(seed)
    null = [herd_metrics(shuffled(snaps, n_agents, rng), adj)["NASR"] for _ in range(n_perm)]
    null = [v for v in null if not math.isnan(v)]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1)) if len(null) > 1 else 0.0
    ge = sum(v >= real for v in null)
    le = sum(v <= real for v in null)
    return {"NASR_null_mean": mu, "NASR_null_sd": sd,
            "NASR_z": (real - mu) / sd if sd > 0 else float("nan"),
            "NASR_p_perm": min(1.0, 2 * min(ge, le) / (len(null) + 1))}


# ---------------------------------------------------------------- loading
def load_runs(root: Path, n_perm: int):
    rows, flips = [], []
    for rdir in sorted((root / "runs").iterdir()):
        sf, cf_ = rdir / "surveys.json", rdir / "run_config.json"
        if not sf.exists():
            continue
        cfg = json.loads(cf_.read_text())
        sv = json.loads(sf.read_text())
        meta = json.loads((rdir / "run_meta.json").read_text()) if (rdir / "run_meta.json").exists() else {}
        n = cfg["num_agents"]
        adj = defaultdict(set)
        for u, v in cfg["edges"]:
            adj[u].add(v)
            adj[v].add(u)
        surveys = sorted(sv["surveys"], key=lambda s: s["round"])
        snaps = [{int(a): c for a, c in s["results"].items()} for s in surveys]
        if len(snaps) < 2:
            continue
        m = herd_metrics(snaps, adj)
        # order consistency and margins (all survey rounds)
        cons, margins = [], []
        for s in surveys:
            for a, d in s["detail"].items():
                if "error" in d:
                    continue
                cons.append(d["order_consistent"])
                margins.append(abs(d["margin"]))
        # margin-stratified flips: earlier answer's margin vs whether it flips next survey
        for s0, s1 in zip(surveys[:-1], surveys[1:]):
            for a, d0 in s0["detail"].items():
                d1 = s1["detail"].get(a)
                if not d1 or "error" in d0 or "error" in d1:
                    continue
                flips.append({"simulator": cfg["simulator"], "family": cfg["model_family"],
                              "finetuned": cfg["finetuned"], "stimulus": cfg["stimulus"],
                              "margin": abs(d0["margin"]), "flip": d0["choice"] != d1["choice"]})
        row = {"run_id": cfg["run_id"], "simulator": cfg["simulator"], "family": cfg["model_family"],
               "finetuned": cfg["finetuned"], "stimulus": cfg["stimulus"], "graph": cfg["graph"],
               "question": cfg["question_number"], "seed": cfg["seed"], **m,
               "order_consistency": float(np.mean(cons)) if cons else float("nan"),
               "mean_abs_margin": float(np.mean(margins)) if margins else float("nan"),
               "n_surveys": len(snaps), "survey_errors": sv.get("n_errors", 0),
               "turns": meta.get("n_turns", float("nan")), "failed_turns": meta.get("n_failed_turns", float("nan")),
               "posts": meta.get("n_posts", float("nan")), "wall_s": meta.get("wall_seconds", float("nan"))}
        if cfg["stimulus"] == "normal":
            row.update(null_test(snaps, adj, n, n_perm, seed=zlib.crc32(cfg["run_id"].encode())))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(flips)


# ---------------------------------------------------------------- stats
def partial_eta2(df: pd.DataFrame, metric: str, formula_rhs: str) -> dict:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = df.dropna(subset=[metric]).copy()
    d["y"] = d[metric]
    model = smf.ols(f"y ~ {formula_rhs}", data=d).fit()
    tab = anova_lm(model, typ=3)
    ss_res = tab.loc["Residual", "sum_sq"]
    out = {}
    for term in tab.index:
        if term in ("Intercept", "Residual"):
            continue
        ss = tab.loc[term, "sum_sq"]
        out[term] = {"eta2p": float(ss / (ss + ss_res)), "p": float(tab.loc[term, "PR(>F)"])}
    return out


def bootstrap_delta(a: np.ndarray, b: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def perm_eta2(values: np.ndarray, groups: np.ndarray, n_perm=2000, seed=0):
    def eta2(v, g):
        grand = v.mean()
        ss_t = ((v - grand) ** 2).sum()
        ss_b = sum(((v[g == k].mean() - grand) ** 2) * (g == k).sum() for k in np.unique(g))
        return ss_b / ss_t if ss_t > 0 else 0.0
    real = eta2(values, groups)
    rng = np.random.default_rng(seed)
    ge = sum(eta2(values, rng.permutation(groups)) >= real for _ in range(n_perm))
    return float(real), float((ge + 1) / (n_perm + 1))


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    return f"{p:.2g}" if p >= 1e-3 else "$<10^{-3}$"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="crosssim/results")
    ap.add_argument("--n_perm", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs, flips = load_runs(Path(args.root), args.n_perm)
    if runs.empty:
        raise SystemExit("no completed runs found")
    runs.to_csv(out / "runs.csv", index=False)
    flips.to_csv(out / "flips.csv", index=False)
    summary: dict = {"n_runs": int(len(runs)),
                     "runs_by_sim": runs.groupby("simulator").size().to_dict()}
    md = [f"# Cross-simulator replication results\n\nCompleted runs: {len(runs)} "
          f"({', '.join(f'{SIM_LABEL.get(k, k)}: {v}' for k, v in summary['runs_by_sim'].items())})\n"]

    # Health
    health = runs.groupby(["simulator", "family", "finetuned"]).agg(
        runs=("run_id", "size"), failed_turn_rate=("failed_turns", "sum"), turns=("turns", "sum"),
        survey_errors=("survey_errors", "sum")).reset_index()
    health["failed_turn_rate"] = health["failed_turn_rate"] / health["turns"].replace(0, np.nan)
    md.append("## Health (failed turns / survey errors)\n\n" + health.to_markdown(index=False) + "\n")

    sims = [s for s in ["oasis", "concordia", "silisocs"] if s in set(runs.simulator)]
    normal = runs[runs.stimulus == "normal"].copy()
    normal["ft"] = normal.finetuned.map({True: "ft", False: "base"})

    # ---- Claim 1: fine-tuning effect, model-gated direction
    md.append("## Claim 1: fine-tuning on social-media data is a dominant, model-gated driver\n")
    c1 = {}
    tex_rows = []
    for sim in sims:
        d = normal[normal.simulator == sim]
        c1[sim] = {}
        for metric in ["OSR", "MFR", "NASR"]:
            try:
                eta = partial_eta2(d, metric, "C(ft, Sum) * C(family, Sum) + C(graph, Sum) + C(question, Sum)")
            except Exception as exc:  # noqa: BLE001
                eta = {"error": str(exc)}
            deltas = {}
            for fam in sorted(set(d.family)):
                a = d[(d.family == fam) & (d.ft == "ft")][metric].dropna().to_numpy()
                b = d[(d.family == fam) & (d.ft == "base")][metric].dropna().to_numpy()
                deltas[fam] = bootstrap_delta(a, b)
            c1[sim][metric] = {"eta": eta, "delta": deltas}
            ft_term = next((v for k, v in eta.items() if k.startswith("C(ft") and ":" not in k), {}) if "error" not in eta else {}
            int_term = next((v for k, v in eta.items() if k.startswith("C(ft") and ":" in k), {}) if "error" not in eta else {}
            fam_term = next((v for k, v in eta.items() if k.startswith("C(family") and ":" not in k), {}) if "error" not in eta else {}
            graph_term = next((v for k, v in eta.items() if k.startswith("C(graph")), {}) if "error" not in eta else {}
            dq = deltas.get("qwen", (float("nan"),) * 3)
            dl = deltas.get("minitaur", (float("nan"),) * 3)
            tex_rows.append((SIM_LABEL[sim], metric, ft_term.get("eta2p"), ft_term.get("p"), int_term.get("eta2p"),
                             fam_term.get("eta2p"), graph_term.get("eta2p"), dl, dq))
    md.append("| Simulator | Metric | ft η²p | p | ft×model η²p | model η²p | graph η²p | Δ Minitaur [CI] | Δ Qwen [CI] |\n|---|---|---|---|---|---|---|---|---|")
    for r in tex_rows:
        md.append(f"| {r[0]} | {r[1]} | {fmt(r[2])} | {fmt_p(r[3])} | {fmt(r[4])} | {fmt(r[5])} | {fmt(r[6])} | "
                  f"{fmt(r[7][0])} [{fmt(r[7][1])}, {fmt(r[7][2])}] | {fmt(r[8][0])} [{fmt(r[8][1])}, {fmt(r[8][2])}] |")
    md.append("")
    summary["claim1_finetuning"] = c1
    # pooled across simulators
    try:
        pooled = {m: partial_eta2(normal, m, "C(ft, Sum) * C(family, Sum) * C(simulator, Sum) + C(graph, Sum) + C(question, Sum)")
                  for m in ["OSR", "MFR", "NASR"]}
        summary["claim1_pooled"] = pooled
        md.append("Pooled across simulators (Type-III partial η², sum contrasts):\n")
        md.append("| Metric | term | η²p | p |\n|---|---|---|---|")
        for m, terms in pooled.items():
            for t, v in sorted(terms.items(), key=lambda kv: -kv[1]["eta2p"])[:6]:
                md.append(f"| {m} | {t} | {fmt(v['eta2p'])} | {fmt_p(v['p'])} |")
        md.append("")
    except Exception as exc:  # noqa: BLE001
        md.append(f"(pooled ANOVA failed: {exc})\n")

    tex = ["% Auto-generated by crosssim/analyze.py",
           "\\begin{tabular}{llrrrrr}", "\\toprule",
           "Simulator & Metric & ft $\\eta^2_p$ & ft$\\times$model $\\eta^2_p$ & graph $\\eta^2_p$ & $\\Delta$ Minitaur & $\\Delta$ Qwen \\\\",
           "\\midrule"]
    for r in tex_rows:
        tex.append(f"{r[0]} & {r[1]} & {fmt(r[2], 2)} & {fmt(r[4], 2)} & {fmt(r[6], 2)} & "
                   f"${r[7][0]:+.3f}$ & ${r[8][0]:+.3f}$ \\\\" if not math.isnan(r[7][0]) and not math.isnan(r[8][0]) else
                   f"{r[0]} & {r[1]} & {fmt(r[2], 2)} & {fmt(r[4], 2)} & {fmt(r[6], 2)} & {fmt(r[7][0])} & {fmt(r[8][0])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_finetuning.tex").write_text("\n".join(tex))

    # ---- Claim 2: scrambled-stimulus noise floor
    md.append("## Claim 2: much of the raw shift rate is context-perturbation noise (scrambled floor)\n")
    er = runs[runs.graph == "random"]
    c2 = []
    for sim in sims:
        for fam in sorted(set(er.family)):
            for ft in [True, False]:
                d = er[(er.simulator == sim) & (er.family == fam) & (er.finetuned == ft)]
                a = d[d.stimulus == "normal"]["OSR"].dropna().to_numpy()
                b = d[d.stimulus == "scrambled"]["OSR"].dropna().to_numpy()
                delta, lo, hi = bootstrap_delta(a, b)
                c2.append({"simulator": sim, "family": fam, "finetuned": ft,
                           "OSR_normal": float(a.mean()) if len(a) else float("nan"),
                           "OSR_scrambled": float(b.mean()) if len(b) else float("nan"),
                           "excess": delta, "ci_lo": lo, "ci_hi": hi,
                           "floor_share": float(b.mean() / a.mean()) if len(a) and len(b) and a.mean() > 0 else float("nan")})
    c2 = pd.DataFrame(c2)
    summary["claim2_scrambled"] = c2.to_dict(orient="records")
    md.append(c2.to_markdown(index=False, floatfmt=".3f") + "\n")
    tex = ["% Auto-generated by crosssim/analyze.py", "\\begin{tabular}{lllrrrr}", "\\toprule",
           "Simulator & Model & FT & OSR normal & OSR scrambled & excess & floor share \\\\", "\\midrule"]
    for _, r in c2.iterrows():
        tex.append(f"{SIM_LABEL[r.simulator]} & {FAM_LABEL[r.family]} & {'on' if r.finetuned else 'off'} & "
                   f"{fmt(r.OSR_normal)} & {fmt(r.OSR_scrambled)} & {fmt(r.excess)} & {fmt(r.floor_share, 2)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_scrambled.tex").write_text("\n".join(tex))

    # ---- Claim 3: order (prompt) sensitivity
    md.append("## Claim 3: answers are prompt-sensitive (dual-order consistency)\n")
    oc = runs.groupby(["simulator", "family", "finetuned"])["order_consistency"].agg(["mean", "std", "size"]).reset_index()
    summary["claim3_order"] = oc.to_dict(orient="records")
    md.append(oc.to_markdown(index=False, floatfmt=".3f") + "\n")
    try:
        oce = {sim: partial_eta2(runs[runs.simulator == sim].assign(ft=lambda x: x.finetuned.astype(str)),
                                 "order_consistency", "C(ft, Sum) * C(family, Sum) + C(question, Sum) + C(stimulus, Sum)")
               for sim in sims}
        summary["claim3_eta"] = oce
        md.append("Order-consistency partial η² per simulator:\n")
        for sim, terms in oce.items():
            md.append(f"- {SIM_LABEL[sim]}: " + ", ".join(f"{k}={fmt(v['eta2p'])}" for k, v in terms.items()))
        md.append("")
    except Exception as exc:  # noqa: BLE001
        md.append(f"(order-consistency ANOVA failed: {exc})\n")

    # ---- Claim 4: margin-stratified flips
    md.append("## Claim 4: flip probability falls with answer confidence (margin)\n")
    c4 = []
    if not flips.empty:
        for sim in sims:
            f = flips[(flips.simulator == sim) & (flips.stimulus == "normal")].copy()
            if len(f) < 50:
                continue
            f["bin"] = pd.qcut(f["margin"].rank(method="first"), 5, labels=False)
            rates = f.groupby("bin")["flip"].mean().to_list()
            rho = float(pd.Series(f["margin"]).corr(f["flip"].astype(float), method="spearman"))
            c4.append({"simulator": sim, "flip_rate_by_margin_quintile": rates, "spearman": rho,
                       "monotone_decreasing": all(x >= y for x, y in zip(rates[:-1], rates[1:]))})
            md.append(f"- {SIM_LABEL[sim]}: quintile flip rates " + ", ".join(f"{x:.3f}" for x in rates) +
                      f" (Spearman ρ = {rho:.3f})")
    summary["claim4_margin"] = c4
    md.append("")

    # ---- Claim 5: topology effect on NASR is mechanical (response-shuffle null)
    md.append("## Claim 5: topology effect on NASR is mechanical (response-shuffle null)\n")
    c5 = []
    tex = ["% Auto-generated by crosssim/analyze.py", "\\begin{tabular}{llrrrr}", "\\toprule",
           "Simulator & Topology & NASR real & NASR null & mean $z$ & \\% $p<.05$ \\\\", "\\midrule"]
    for sim in sims:
        d = normal[normal.simulator == sim].dropna(subset=["NASR", "NASR_null_mean"])
        if d.empty:
            continue
        eta_real, p_real = perm_eta2(d["NASR"].to_numpy(), d["graph"].to_numpy())
        eta_null, p_null = perm_eta2(d["NASR_null_mean"].to_numpy(), d["graph"].to_numpy())
        for g in ["cycle", "random", "barabasi_albert"]:
            dg = d[d.graph == g]
            if dg.empty:
                continue
            row = {"simulator": sim, "graph": g, "n": len(dg), "NASR_real": dg.NASR.mean(),
                   "NASR_null": dg.NASR_null_mean.mean(), "mean_z": dg.NASR_z.mean(),
                   "pct_sig": float((dg.NASR_p_perm < 0.05).mean()),
                   "graph_eta2_real": eta_real, "graph_eta2_real_p": p_real,
                   "graph_eta2_null": eta_null, "graph_eta2_null_p": p_null}
            c5.append(row)
            tex.append(f"{SIM_LABEL[sim]} & {GRAPH_LABEL[g]} & {fmt(row['NASR_real'])} & {fmt(row['NASR_null'])} & "
                       f"${row['mean_z']:+.2f}$ & {100 * row['pct_sig']:.0f} \\\\")
        tex.append("\\midrule")
    if tex[-1] == "\\midrule":
        tex.pop()
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_topology_null.tex").write_text("\n".join(tex))
    c5 = pd.DataFrame(c5)
    summary["claim5_topology_null"] = c5.to_dict(orient="records")
    if not c5.empty:
        md.append(c5.to_markdown(index=False, floatfmt=".3f") + "\n")

    # ---- compact one-row-per-simulator table for the paper (with V2 reference row)
    tex = ["% Auto-generated by crosssim/analyze.py: one row per simulator, normal stimulus unless noted",
           "\\begin{tabular}{lrrrrrrr}", "\\toprule",
           " & \\multicolumn{3}{c}{Fine-tuning (OSR)} & Scrambled & Order & Flip rate & NASR \\\\",
           "\\cmidrule(lr){2-4}",
           "Simulator & $\\eta^2_p$ & $\\Delta$ Minitaur & $\\Delta$ Qwen & floor share & consist. & low$\\to$high margin & real$-$null (\\% sig.) \\\\",
           "\\midrule",
           "Ours (V2, $N{=}256$) & 0.75 & $+$ & $-$ & -- & 0.37--0.87 & 0.32$\\to$0.06 & $-0.001$ (6.5) \\\\",
           "\\midrule"]
    for sim in sims:
        osr = c1.get(sim, {}).get("OSR", {})
        eta = osr.get("eta", {})
        ft_eta = next((v["eta2p"] for k, v in eta.items() if k.startswith("C(ft") and ":" not in k), float("nan")) \
            if isinstance(eta, dict) and "error" not in eta else float("nan")
        dl = osr.get("delta", {}).get("minitaur", (float("nan"),))[0]
        dq = osr.get("delta", {}).get("qwen", (float("nan"),))[0]
        fs = c2[(c2.simulator == sim) & (c2.finetuned)]  # floor share for fine-tuned agents (base agents barely move)
        floor = float((fs.OSR_scrambled.sum() / fs.OSR_normal.sum())) if len(fs) and fs.OSR_normal.sum() > 0 else float("nan")
        ocs = runs[runs.simulator == sim]["order_consistency"]
        oc_txt = f"{ocs.groupby([runs.family, runs.finetuned]).mean().min():.2f}--{ocs.groupby([runs.family, runs.finetuned]).mean().max():.2f}" if len(ocs) else "--"
        m4 = next((r for r in c4 if r["simulator"] == sim), None)
        flip_txt = f"{m4['flip_rate_by_margin_quintile'][0]:.2f}$\\to${m4['flip_rate_by_margin_quintile'][-1]:.2f}" if m4 else "--"
        dn = normal[(normal.simulator == sim)].dropna(subset=["NASR", "NASR_null_mean"])
        nasr_txt = (f"${(dn.NASR - dn.NASR_null_mean).mean():+.3f}$ ({100 * (dn.NASR_p_perm < 0.05).mean():.1f})"
                    if len(dn) else "--")
        sgn = lambda x: "--" if math.isnan(x) else f"${x:+.3f}$"
        tex.append(f"{SIM_LABEL[sim]} & {fmt(ft_eta, 2)} & {sgn(dl)} & {sgn(dq)} & {fmt(floor, 2)} & {oc_txt} & {flip_txt} & {nasr_txt} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_xsim_summary.tex").write_text("\n".join(tex))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
