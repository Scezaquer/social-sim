# Cross-simulator runner contract (V3 resubmission)

Goal: test whether the paper's main claims (fine-tuning dominance with model-gated sign,
scrambled-stimulus noise floor, prompt/order sensitivity, margin-stratified flips, mechanical
topology effect under a response-shuffle null) hold in three independently built simulators:
OASIS, Concordia, SiliSocS. Each simulator keeps its OWN agent architecture, prompting, memory
and action mechanics. What is shared across simulators: the population (personas, model/LoRA
assignment), the follow graph, the topic seed, the scrambled corpus, the schedule
(rounds / activity / survey rounds), and the out-of-band survey instrument.

## Invocation

    python crosssim/run_<sim>.py --config <run_dir>/run_config.json

`run_config.json` (written by `crosssim/prep.py`) contains:

| key | type | meaning |
|---|---|---|
| run_id | str | unique id |
| simulator | "oasis" \| "concordia" \| "silisocs" | |
| model_family | "llama3.1" \| "qwen" | |
| finetuned | bool | BluePrint LoRA arm or base arm |
| base_url | str | vLLM OpenAI-compatible URL, e.g. `http://127.0.0.1:8000/v1` |
| agent_models | list[str], len N | served model name per agent (`base` or `lora{k}`) |
| agent_names | list[str], len N | display names (unique) |
| personas | list[str], len N | persona text per agent |
| edges | list[[u,v]] | UNDIRECTED edges over agent indices 0..N-1 -> mutual follows |
| num_agents | int | N |
| rounds | int | R simulation rounds (round index r = 1..R) |
| active_schedule | list[list[int]], len R | agent indices that act in round r (index r-1). Precomputed so all sims share it |
| survey_rounds | list[int] | e.g. [0,4,8,12,16,20]; context must be dumped AFTER round r completes (round 0 = before any interaction) |
| stimulus | "normal" \| "scrambled" | |
| topic_seed | str | neutral trending-topic text shown to every agent once, before round 1, in BOTH conditions |
| scrambled_posts | list[{"author": str, "text": str}] | unrelated-topic corpus for the scrambled condition |
| feed_size | int | max posts an agent sees per round (default 5) |
| max_tokens | int | generation cap per call (default 160) |
| temperature | float | sampling temperature for actions (default 0.7) |
| seed | int | |
| out_dir | str | write all outputs here |

## Semantics

* Round 0: every agent has observed `topic_seed` (and nothing else).
* Rounds 1..R: the agents listed in `active_schedule[r-1]` take one turn each: observe their
  feed, then act (post / reply / like / do nothing, as the simulator natively allows; at
  minimum "write a post").
* `normal`: the feed is the simulator's native feed restricted as far as practical to posts by
  agents the actor follows (graph neighbours), newest first, at most `feed_size`.
* `scrambled`: everything the agent would have observed from OTHER agents is replaced by
  `feed_size` posts sampled (seeded) from `scrambled_posts`. Agents still act and their own
  posts are still recorded, but there is no coupling between agents. The topic seed is still
  shown at round 0.
* No survey is ever written into agent memory / context (out-of-band surveys only).

## Required outputs in out_dir

1. `contexts.jsonl` — one line per (survey round, agent):
   `{"round": r, "agent": i, "model": agent_models[i], "context": "<text>"}`
   `context` = the simulator-native text of what this agent currently has in its
   working context (persona excluded; recent observations + own recent actions/memories),
   newest LAST, at most ~6000 characters (truncate from the start). Round 0 must exist for all
   agents. This text is what `crosssim/survey.py` conditions the survey on.
2. `actions.jsonl` — one line per agent turn: `{"round": r, "agent": i, "type": "...", "text": "...", "ok": bool}`.
3. `run_meta.json` — `{"run_id", "simulator", "wall_seconds", "n_turns", "n_failed_turns",
   "n_posts", "notes": [...]}`. A failed turn (exception, no tool call, invalid action)
   must be counted, not crash the run.
4. Exit code 0 on success. The run must not hang forever on a dead server: use request
   timeouts (<=180 s) and bounded retries.

The survey (`crosssim/survey.py`) is run by the job script after the runner exits, against the
same vLLM server.
