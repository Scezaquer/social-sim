#!/usr/bin/env python3
"""Deterministic, balanced experimental designs for the Cookbook V2 experiments.

Every V2 simulation run is one row of a committed design CSV; SLURM array index i
runs row i. This replaces V1's on-node $RANDOM sampling, which produced
unbalanced cells and irreproducible configurations (reviewer-blocking issue:
single-factor eta^2 on unbalanced data).

Balance guarantees:
- exact marginal balance for every factor;
- exact model x factor pairwise balance (factors are shuffled independently
  *within* model strata);
- all randomization seeded (DESIGN_SEED), so re-running this script reproduces
  the committed CSVs byte-for-byte.

Run seeds are derived from run_id, so every simulation is reproducible from its
design row alone.

Usage:
    python generate_design.py [--outdir designs]
"""

from __future__ import annotations

import argparse
import csv
import os
import random

DESIGN_SEED = 20260612

QUESTIONS = {
    25: "genetic_enhancements_tweets.json",
    28: "ai_copyright_tweets.json",
    29: "environmental_protection_tweets.json",
}

LORA_PROFILES = ["minitaur_loras", "llama3.1_loras", "qwen_loras", "gemma_loras"]
BASE_PROFILES = ["minitaur_base", "llama3.1_base", "qwen_base", "gemma_base"]

COLUMNS = [
    "run_id",
    "experiment",
    "seed",
    "model_profile",
    "proportions_option",
    "question_number",
    "tweet_file",
    "num_agents",
    "graph_model",
    "homophily",
    "survey_ctx",
    "num_news_agents",
    "activity_exponent",
    "stimulus_mode",
    "survey_order_mode",
    "max_steps",
    "survey_interval",
]


def balanced_column(levels, n, rng):
    """Each level appears exactly n/len(levels) times, in seeded random order."""
    if n % len(levels) != 0:
        raise ValueError(f"{n} runs not divisible by {len(levels)} levels ({levels})")
    column = list(levels) * (n // len(levels))
    rng.shuffle(column)
    return column


def stratified_design(strata_levels, strata_size, factors, rng):
    """One stratum per level of the primary factor (model); every other factor is
    exactly balanced within each stratum, guaranteeing primary x factor balance."""
    rows = []
    for stratum in strata_levels:
        columns = {name: balanced_column(levels, strata_size, rng) for name, levels in factors.items()}
        for i in range(strata_size):
            row = {"model_profile": stratum}
            for name in factors:
                row[name] = columns[name][i]
            rows.append(row)
    rng.shuffle(rows)
    return rows


def finalize(rows, experiment, id_offset, defaults):
    out = []
    for i, row in enumerate(rows):
        run_id = id_offset + i
        full = dict(defaults)
        full.update(row)
        full["run_id"] = run_id
        full["experiment"] = experiment
        full["seed"] = run_id
        full["tweet_file"] = QUESTIONS[int(full["question_number"])]
        out.append({c: full[c] for c in COLUMNS})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):4d} runs -> {path}")


def e1_core_sweep(rng):
    """E1: balanced core sweep, 576 runs.

    model(4) x proportions(4) x question(3) x N(3) x graph(2) x homophily(2)
    x ctx(2) x news(3) x activity(3); 144 runs per model stratum.
    """
    factors = {
        "proportions_option": ["uniform", "blueprint", "average", "distribution"],
        "question_number": [25, 28, 29],
        "num_agents": [64, 256, 1024],
        "graph_model": ["random", "powerlaw_cluster"],
        "homophily": ["on", "off"],
        "survey_ctx": ["on", "off"],
        "num_news_agents": [0, 1, 4],
        "activity_exponent": [0.0, 0.5, 1.0],
    }
    rows = stratified_design(LORA_PROFILES, 144, factors, rng)
    defaults = {
        "stimulus_mode": "normal",
        "survey_order_mode": "both",
        "max_steps": 2500,
        "survey_interval": 250,
    }
    return finalize(rows, "e1_core", 10000, defaults)


def e2_base_vs_lora(rng):
    """E2: no-LoRA arm for all four base models, 144 runs at N=256.

    Crossed with E1's LoRA arm, this breaks the V1 nesting of fine-tuning
    status under model identity (only Qwen had a base-only condition).
    """
    factors = {
        "question_number": [25, 28, 29],
        "graph_model": ["random", "powerlaw_cluster"],
        "homophily": ["on", "off"],
        "survey_ctx": ["on", "off"],
        "num_news_agents": [0, 1, 4],
        "activity_exponent": [0.0, 0.5, 1.0],
    }
    rows = stratified_design(BASE_PROFILES, 36, factors, rng)
    defaults = {
        "proportions_option": "none",
        "num_agents": 256,
        "stimulus_mode": "normal",
        "survey_order_mode": "both",
        "max_steps": 2500,
        "survey_interval": 250,
    }
    return finalize(rows, "e2_base_vs_lora", 20000, defaults)


def e3_topology(rng):
    """E3: controlled topology study, 336 runs.

    Full factorial: topology(7) x model(4) x question(3) x 4 replicates, all
    other axes fixed; topologies density-matched to mean degree ~16 in the
    simulator (cycle and fully_connected are intentional structural extremes).
    """
    topologies = [
        "random",
        "powerlaw_cluster",
        "barabasi_albert",
        "stochastic_block",
        "forest_fire",
        "fully_connected",
        "cycle",
    ]
    rows = []
    for topo in topologies:
        for model in LORA_PROFILES:
            for q in QUESTIONS:
                for _ in range(4):
                    rows.append({
                        "model_profile": model,
                        "graph_model": topo,
                        "question_number": q,
                    })
    rng.shuffle(rows)
    defaults = {
        "proportions_option": "blueprint",
        "num_agents": 256,
        "homophily": "off",
        "survey_ctx": "off",
        "num_news_agents": 0,
        "activity_exponent": 0.5,
        "stimulus_mode": "normal",
        "survey_order_mode": "both",
        "max_steps": 2500,
        "survey_interval": 250,
    }
    return finalize(rows, "e3_topology", 30000, defaults)


def e4_noise_floor(rng):
    """E4: measurement-validity baselines, 96 runs.

    stimulus(none/scrambled) x ctx(2) x model(4) x question(3) x 2 replicates.
    'none' quantifies pure self-feedback drift (expected exactly 0 when
    ctx=off: deterministic argmax choice on an unchanged context - reported as
    a sanity check). 'scrambled' is the context-perturbation noise floor that
    real-simulation shift rates must exceed to be called socially driven.
    Matched 'normal' comparators are the E1 cells at N=256.
    """
    rows = []
    for stim in ["none", "scrambled"]:
        for ctx in ["on", "off"]:
            for model in LORA_PROFILES:
                for q in QUESTIONS:
                    for _ in range(2):
                        rows.append({
                            "model_profile": model,
                            "stimulus_mode": stim,
                            "survey_ctx": ctx,
                            "question_number": q,
                        })
    rng.shuffle(rows)
    defaults = {
        "proportions_option": "blueprint",
        "num_agents": 256,
        "graph_model": "powerlaw_cluster",
        "homophily": "off",
        "num_news_agents": 0,
        "activity_exponent": 0.5,
        "survey_order_mode": "both",
        "max_steps": 2500,
        "survey_interval": 250,
    }
    return finalize(rows, "e4_noise_floor", 40000, defaults)


def e5_scale_cadence(rng):
    """E5: scale with per-agent-matched survey cadence, 72 runs.

    V1's OSR-vs-scale trend was an artifact of a fixed survey interval (fewer
    actions per agent between surveys at larger N). Here interval = N and
    max_steps = 8N, so expected actions per agent between surveys is constant
    across scales. If OSR still falls with N, scale matters; if it flattens,
    V1's trend was pure cadence.
    """
    rows = []
    for n in [64, 256, 1024]:
        for model in ["minitaur_loras", "qwen_loras"]:
            for q in QUESTIONS:
                for _ in range(4):
                    rows.append({
                        "model_profile": model,
                        "num_agents": n,
                        "question_number": q,
                        "max_steps": 8 * n,
                        "survey_interval": n,
                    })
    rng.shuffle(rows)
    defaults = {
        "proportions_option": "blueprint",
        "graph_model": "powerlaw_cluster",
        "homophily": "off",
        "survey_ctx": "off",
        "num_news_agents": 0,
        "activity_exponent": 0.5,
        "stimulus_mode": "normal",
        "survey_order_mode": "both",
    }
    return finalize(rows, "e5_scale_cadence", 50000, defaults)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "designs"))
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = random.Random(DESIGN_SEED)
    write_csv(os.path.join(args.outdir, "e1_core_sweep.csv"), e1_core_sweep(rng))
    write_csv(os.path.join(args.outdir, "e2_base_vs_lora.csv"), e2_base_vs_lora(rng))
    write_csv(os.path.join(args.outdir, "e3_topology.csv"), e3_topology(rng))
    write_csv(os.path.join(args.outdir, "e4_noise_floor.csv"), e4_noise_floor(rng))
    write_csv(os.path.join(args.outdir, "e5_scale_cadence.csv"), e5_scale_cadence(rng))


if __name__ == "__main__":
    main()
