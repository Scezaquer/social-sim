# Cross-simulator replication (V3, reviewer weakness 1)

Re-tests the paper's main claims in three independently built simulators, OASIS
(camel-oasis 0.2.5), Concordia (gdm-concordia 2.4.0) and SiliSocS (silisocs 0.4.0),
with the paper's own base models and BluePrint LoRAs served by vLLM.

Each simulator keeps its own agent architecture, prompting, memory and action
mechanics. Shared across simulators (per seed): personas (Tianyi-Lab/Personas),
LoRA assignment (uniform over the 25 archetypes), follow graph (density-matched as in
`src/main.py`), activity schedule, a neutral topic seed, the scrambled corpus, and an
out-of-band dual-order log-prob survey identical in form to the paper's
(`survey.py`). See `RUNNER_SPEC.md` for the runner contract.

## Design (`prep.py`)

24 jobs = 3 simulators x 2 models (Qwen2.5-7B-Instruct, Llama-3.1-Minitaur-8B) x 2 questions
(Q28 AI-copyright, Q29 growth-vs-environment) x 2 seeds. Each job runs 8 simulations,
most important first:

| # | graph | fine-tuning | stimulus |
|---|---|---|---|
| 1-2 | Erdős–Rényi | on / off | normal |
| 3-4 | Erdős–Rényi | on / off | scrambled |
| 5-6 | Barabási–Albert | on / off | normal |
| 7-8 | cycle | on / off | normal |

N = 64 agents, 20 rounds, 50% of agents active per round, surveys at rounds 0,4,...,20
(6 snapshots), both option orders. 192 runs total.

Claims tested (`analyze.py`):
1. Fine-tuning is the dominant driver of OSR/NASR (partial η²) and its direction is model-gated.
2. Much of the raw shift rate survives a scrambled-stimulus baseline (noise floor).
3. Survey answers are order-sensitive (dual-order consistency), model/fine-tuning dependent.
4. Flip probability falls with the earlier answer's log-prob margin.
5. Topology differences in NASR are reproduced by the response-shuffle null (mechanical).

## Running on Mila

```bash
cd ~/social-sim && git pull
bash crosssim/slurm/setup_mila.sh                 # login node, once (~20 min)

# 1) smoke test: 6 tiny jobs (3 sims x 2 models), ~20-30 min including model load
sbatch --array=0-5 --export=ALL,DESIGN=$SCRATCH/crosssim_smoke crosssim/slurm/job.sh
#    check logs/xsim_*.out: "vLLM up", "[check] tool call: [...]", runner exit=0, "[survey] ... 0 errors"

# 2) full experiment: 24 GPUs
sbatch --array=0-23 --export=ALL,DESIGN=$SCRATCH/crosssim_out crosssim/slurm/job.sh
bash crosssim/slurm/status.sh                     # progress

# 3) results (works on partial results too)
bash crosssim/slurm/collect.sh                    # -> crosssim/results/report.md + crosssim_results.tgz
```

Re-submitting the same array resumes: finished runs (`DONE` marker) are skipped.

### If something fails

* `unknown family` / `HF_MODEL: unbound variable`: the design was generated before a code change.
  Regenerate both designs (finished runs keep their `DONE` markers and are skipped):
  ```bash
  source crosssim/slurm/env.sh
  $ENVS/tools/bin/python crosssim/prep.py design --out $CROSSSIM_SMOKE --smoke --n_agents 16 --rounds 2 --survey_every 1 --seeds 1 --questions 29
  $ENVS/tools/bin/python crosssim/prep.py design --out $CROSSSIM_OUT
  ```

* vLLM does not start: `job.sh` retries with `VLLM_ATTENTION_BACKEND=TRITON_ATTN` and then
  `FLEX_ATTENTION` (both with `--enforce-eager`); logs in `$DESIGN/server_logs/`. Another
  version: `VLLM_VERSION=0.10.2 bash crosssim/slurm/setup_mila.sh` (re-creates nothing else).
* SiliSocS tool calls fail: resubmit SiliSocS jobs with
  `--export=ALL,DESIGN=...,RUNNER_ARGS="--tool-mode none"` (SiliSocS's text-action fallback).
* OASIS needs tool calls (no fallback): check the `[check] tool call` line in the job log.

## Local tests (no GPU)

`tests/integration.sh` runs prep -> all three runners (stub servers) -> survey -> analysis.
