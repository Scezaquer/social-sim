"""Cross-simulator design generator (V3 resubmission, reviewer weakness 1).

Builds a deterministic design: one SLURM array task ("job") per
(simulator, model family, question, seed); each job runs an ordered list of
simulation runs against one vLLM server. Everything that must be identical
across simulators for the same seed (personas, names, LoRA assignment, graphs,
activity schedule, topic seed, scrambled corpus) is generated here, once.

Usage (from repo root, on the cluster login node, in the tools venv):
    python crosssim/prep.py personas   # writes crosssim/data/personas.json
    python crosssim/prep.py design --out "$CROSSSIM_OUT"   # writes jobs.csv + run configs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("CROSSSIM_REPO", str(HERE.parent))).resolve()
DATA = HERE / "data"

SIMULATORS = ["oasis", "concordia", "silisocs"]
FAMILIES = ["qwen", "llama3.1"]
QUESTIONS = [28, 29]
SEEDS = [1, 2]
NUM_LORAS = 25

N_AGENTS = 64
ROUNDS = 20
ACTIVE_FRAC = 0.5
SURVEY_EVERY = 4
FEED_SIZE = 5
MAX_TOKENS = 160
TEMPERATURE = 0.7
TARGET_MEAN_DEGREE = 16  # same density target as the paper's simulator

# Topic seeds: neutral phrasing of the survey question's issue (no answer cue).
TOPIC_SEEDS = {
    25: "People are debating whether it is acceptable to use genetic engineering to enhance human intelligence.",
    28: "People are debating whether it should be legal to use copyrighted material to train artificial intelligence models.",
    29: "People are debating whether governments should prioritize economic growth or environmental protection.",
}

# Unrelated-topic corpus for the scrambled-stimulus baseline.
SCRAMBLED_FILES = [
    "news_tweets/drone_tweets.json",
    "news_tweets/trump_tweets.json",
    "news_tweets/poilievre_tweets.json",
    "news_tweets/maduro_tweets.json",
    "news_tweets/maduro_tweets2.json",
]

# Order matters: most important runs first, so a job that runs out of time
# still yields the core contrasts.  (graph, finetuned, stimulus)
RUN_ORDER = [
    ("random", True, "normal"),
    ("random", False, "normal"),
    ("random", True, "scrambled"),
    ("random", False, "scrambled"),
    ("barabasi_albert", True, "normal"),
    ("barabasi_albert", False, "normal"),
    ("cycle", True, "normal"),
    ("cycle", False, "normal"),
]

FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn",
         "Drew", "Robin", "Charlie", "Emerson", "Hayden", "Kendall", "Logan", "Parker", "Reese", "Rowan",
         "Sage", "Skyler", "Blake", "Cameron", "Dakota", "Ellis", "Finley", "Harper", "Jesse", "Kai",
         "Lee", "Marley", "Noel", "Oakley", "Peyton", "River", "Shay", "Tatum", "Val", "Winter"]
LAST = ["Smith", "Garcia", "Nguyen", "Brown", "Martin", "Lee", "Walker", "Hall", "Young", "King",
        "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Nelson", "Carter", "Mitchell",
        "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart"]


def build_graph(kind: str, n: int, seed: int) -> nx.Graph:
    """Same construction (and density target) as src/main.py::_build_graph."""
    if kind == "random":
        p = min(1.0, TARGET_MEAN_DEGREE / max(1, n - 1))
        return nx.erdos_renyi_graph(n, p, seed=seed)
    if kind == "barabasi_albert":
        m = min(TARGET_MEAN_DEGREE // 2, max(1, n - 1))
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if kind == "cycle":
        return nx.cycle_graph(n)
    raise ValueError(kind)


def load_scrambled_posts() -> list[dict]:
    posts = []
    for rel in SCRAMBLED_FILES:
        for item in json.loads((REPO / rel).read_text(encoding="utf-8")):
            text = (item.get("message") or "").strip()
            if text:
                posts.append({"author": item.get("author", "news"), "text": text})
    return posts


def persona_to_text(raw: str) -> str:
    try:
        d = json.loads(raw)
    except Exception:
        return str(raw).strip()
    if isinstance(d, dict):
        return "; ".join(f"{k}: {v}" for k, v in d.items() if v not in (None, "", []))
    return str(d)


def make_personas(n: int = 3000, seed: int = 0) -> list[str]:
    """Personas from Tianyi-Lab/Personas (the dataset the paper's base arm uses)."""
    from datasets import load_dataset  # tools venv only
    ds = load_dataset("Tianyi-Lab/Personas")["train"]
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    return [persona_to_text(ds[i]["meta_persona"]) for i in idx]


def load_questions(qnum: int) -> tuple[str, list[str], str]:
    data = json.loads((REPO / "divisive_questions_probabilities.json").read_text(encoding="utf-8"))
    entry = data[qnum]
    question = entry["question"]
    options = list(next(iter(entry["distributions"].values())).keys())
    flipped = json.loads((REPO / "flipped_questions.json").read_text(encoding="utf-8"))[str(qnum)]
    assert flipped["question"] == question, "flipped_questions.json mismatch"
    return question, options, flipped["flipped"]


def population(seed: int, personas_all: list[str], N_AGENTS: int = N_AGENTS, ROUNDS: int = ROUNDS) -> dict:
    rng = random.Random(10_000 + seed)
    names, used = [], set()
    while len(names) < N_AGENTS:
        nm = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if nm not in used:
            used.add(nm)
            names.append(nm)
    personas = rng.sample(personas_all, N_AGENTS)
    lora_ids = [rng.randrange(NUM_LORAS) for _ in range(N_AGENTS)]  # "uniform" proportions
    sched_rng = random.Random(20_000 + seed)
    k = max(1, int(round(ACTIVE_FRAC * N_AGENTS)))
    schedule = [sorted(sched_rng.sample(range(N_AGENTS), k)) for _ in range(ROUNDS)]
    return {"names": names, "personas": personas, "lora_ids": lora_ids, "schedule": schedule}


def cmd_personas(args):
    DATA.mkdir(parents=True, exist_ok=True)
    try:
        personas = make_personas()
        src = "Tianyi-Lab/Personas"
    except Exception as exc:  # deterministic fallback, flagged in the file
        print(f"WARNING: could not load Tianyi-Lab/Personas ({exc}); using synthetic personas", file=sys.stderr)
        rng = random.Random(0)
        jobs = ["teacher", "nurse", "software developer", "retired engineer", "student", "farmer",
                "small-business owner", "lawyer", "artist", "electrician", "accountant", "journalist"]
        places = ["a large city", "a suburb", "a small town", "a rural area"]
        personas = [f"AGE: {rng.randint(18, 80)}; OCCUPATION: {rng.choice(jobs)}; LIVES IN: {rng.choice(places)}"
                    for _ in range(3000)]
        src = "synthetic-fallback"
    out = DATA / "personas.json"
    out.write_text(json.dumps({"source": src, "personas": personas}, indent=0), encoding="utf-8")
    print(f"wrote {len(personas)} personas ({src}) -> {out}")


def cmd_design(args):
    out = Path(args.out).resolve()
    pfile = DATA / "personas.json"
    if not pfile.exists():
        sys.exit("run `prep.py personas` first")
    personas_all = json.loads(pfile.read_text(encoding="utf-8"))["personas"]
    scrambled = load_scrambled_posts()
    N_AGENTS, ROUNDS = args.n_agents, args.rounds
    survey_rounds = list(range(0, ROUNDS + 1, args.survey_every))
    if survey_rounds[-1] != ROUNDS:
        survey_rounds.append(ROUNDS)
    run_order = RUN_ORDER[:3] if args.smoke else RUN_ORDER

    sims = args.simulators.split(",")
    fams = args.families.split(",")
    questions = [int(q) for q in args.questions.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    for sim in sims:
        for fam in fams:
            for q in questions:
                for seed in seeds:
                    pop = population(seed, personas_all, N_AGENTS, ROUNDS)
                    question, options, flipped = load_questions(q)
                    run_ids = []
                    for graph, ft, stim in run_order:
                        run_id = f"{sim}_{fam}_q{q}_s{seed}_{graph}_{'ft' if ft else 'base'}_{stim}"
                        g = build_graph(graph, N_AGENTS, seed)
                        cfg = {
                            "run_id": run_id, "simulator": sim, "model_family": fam, "finetuned": ft,
                            "base_url": "__BASE_URL__",
                            "agent_models": [f"lora{k}" for k in pop["lora_ids"]] if ft else ["base"] * N_AGENTS,
                            "agent_names": pop["names"], "personas": pop["personas"],
                            "edges": [[int(u), int(v)] for u, v in g.edges()],
                            "graph": graph, "graph_mean_degree": 2 * g.number_of_edges() / N_AGENTS,
                            "num_agents": N_AGENTS, "rounds": ROUNDS, "active_schedule": pop["schedule"],
                            "survey_rounds": survey_rounds, "stimulus": stim,
                            "topic_seed": TOPIC_SEEDS[q], "scrambled_posts": scrambled,
                            "feed_size": FEED_SIZE, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
                            "seed": seed, "question_number": q, "question": question,
                            "flipped_question": flipped, "options": options,
                            "out_dir": str(out / "runs" / run_id),
                        }
                        rdir = out / "runs" / run_id
                        rdir.mkdir(parents=True, exist_ok=True)
                        (rdir / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
                        run_ids.append(run_id)
                    jobs.append({"job_index": len(jobs), "simulator": sim, "family": fam,
                                 "question": q, "seed": seed, "runs": ";".join(run_ids)})
    with open(out / "jobs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(jobs)
    print(f"wrote {len(jobs)} jobs / {sum(len(j['runs'].split(';')) for j in jobs)} runs -> {out}")
    print(f"sbatch --array=0-{len(jobs) - 1} --export=ALL,DESIGN={out} crosssim/slurm/job.sh")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("personas")
    d = sub.add_parser("design")
    d.add_argument("--out", required=True)
    d.add_argument("--simulators", default=",".join(SIMULATORS))
    d.add_argument("--families", default=",".join(FAMILIES))
    d.add_argument("--questions", default=",".join(map(str, QUESTIONS)))
    d.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    d.add_argument("--n_agents", type=int, default=N_AGENTS)
    d.add_argument("--rounds", type=int, default=ROUNDS)
    d.add_argument("--survey_every", type=int, default=SURVEY_EVERY)
    d.add_argument("--smoke", action="store_true", help="only the first 3 runs per job (ER ft/base normal, ER ft scrambled)")
    args = ap.parse_args()
    {"personas": cmd_personas, "design": cmd_design}[args.cmd](args)


if __name__ == "__main__":
    main()
