"""SiliSocS runner for the cross-simulator study (see crosssim/RUNNER_SPEC.md).

Requires: silisocs==0.4.0 (Python >= 3.11), openai<2, pyyaml.
    pip install "silisocs==0.4.0"

Design: SiliSocS keeps its own architecture -- twitter_like SQLite backend + component
game master, NativeAgent (persona + recent observations, one tool-calling chat request
per turn, single_action turn policy), follower_chronological feed (timeline_posts =
feed_size) over a static, predefined mutual-follow graph (follow/unfollow/mute
disabled). This runner writes a Hydra scenario under out_dir/silisocs_raw/conf, runs
`python -m silisocs.runtime.runner` in a subprocess, and post-processes the outputs.
Custom pieces live in silisocs_hooks.py (participation = active_schedule, scrambled
feed replacement, context dumps, per-turn log).

    python crosssim/run_silisocs.py --config <run_dir>/run_config.json [--tool-mode none]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import openai
import yaml

HERE = Path(__file__).resolve().parent
OBS_HISTORY = 10  # NativeAgent recent-observation window (feeds + own action results)
ACTIONS = ["create_tweet", "reply_to_tweet", "like_tweet", "do_nothing"]
OK_LABELS = {"post": "create_tweet", "reply": "reply_to_tweet", "like": "like_tweet",
             "do_nothing": "do_nothing", "repost": "repost_tweet"}


def wait_for_server(base_url, deadline_s=300):
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=10, max_retries=0)
    t0 = time.time()
    while True:
        try:
            return [m.id for m in client.models.list().data]
        except Exception:  # pylint: disable=broad-except
            if time.time() - t0 > deadline_s:
                raise
            time.sleep(5)


def build_scenario(cfg, conf: Path, tool_mode: str, scripted: bool):
    n = int(cfg["num_agents"])
    names, models_ = cfg["agent_names"], cfg["agent_models"]
    adj = {names[i]: [] for i in range(n)}
    for u, v in cfg["edges"]:
        u, v = int(u), int(v)
        if u != v:
            adj[names[u]].append(names[v])
            adj[names[v]].append(names[u])
    adj = {k: sorted(set(v)) for k, v in adj.items()}
    records = [{"name": names[i], "persona": cfg["personas"][i], "model": models_[i]}
               for i in range(n)]
    world = {
        "scenario_name": "crosssim", "num_agents": n, "num_steps": int(cfg["rounds"]),
        "seed": int(cfg.get("seed", 0)), "run_name": str(cfg["run_id"]),
        "jobname_format": "${run_name}", "output_rootname": "",
        # Required by SiliSocS validation; NOT shown to agents (shared memories and
        # world_context are emptied below).
        "setting": {"name": "Social media platform", "background": ["A social media platform."]},
        "event": {"name": "crosssim", "context": "Users share posts on a social media platform."},
        # Round 0: the topic seed is delivered (memory + recent observations) to every
        # agent before step 0 runs; nothing else is pre-seeded.
        "interventions": [{"at_step": 0, "actions": [{
            "kind": "broadcast_observation", "agents": [],
            "text": "Trending on the platform: " + cfg["topic_seed"]}]}],
    }
    agents = {
        "shared_memories": [], "initial_observations": [],
        "persona_pipeline": {
            "defaults": {"params": {"world_context": ""}, "shared_memories": []},
            "classes": {"user": {
                "count": n, "class_path": "silisocs_hooks.CrossSimAgent", "sim_role_name": "user",
                "data": {"source": "inline", "records": records},
                "params": {"observation_history": OBS_HISTORY, "memory_history": OBS_HISTORY},
                "field_map": {"name": "name", "context": "persona", "model": "model"}}}}}
    env = {"gm": {
        "backend": {"type": "twitter_like", "enabled_actions": ACTIONS, "excluded_actions": None},
        "components": {
            "initialize": {"built_in": "social_media", "params": {"graph": {
                "network_type": "predefined", "predefined_graph": adj,
                "fully_connected_targets": [], "base_followership_probability": 0.0}}},
            "next_acting": {"built_in": "all_agents"},
            # SiliSocS requires resolver and sim.tool_calling.mode to agree.
            "resolve": {"built_in": "tool_calling" if tool_mode != "none" else "parsed_action"},
            "observe": {"built_in": "timeline_every_turn", "params": {
                "timeline_mode": "follower_chronological", "recsys_type": None,
                "timeline_posts": int(cfg.get("feed_size", 5))}}}}}
    llm = ({"provider": "scripted"} if scripted else {
        "provider": "openai_compatible", "name": models_[0], "api_base": cfg["base_url"],
        "api_key": "EMPTY", "temperature": float(cfg.get("temperature", 0.7)),
        # extra_kwargs is merged LAST into every chat.completions request (text and
        # tool calls), so it caps generation everywhere and overrides the temperature
        # SiliSocS hard-codes (0.5) for tool-calling requests.
        "extra_kwargs": {"max_tokens": int(cfg.get("max_tokens", 160)),
                         "temperature": float(cfg.get("temperature", 0.7))}})
    sim = {
        "llm": llm, "max_concurrent_actions": 64,
        "tool_calling": {"mode": tool_mode},
        "memory": {"built_in": None, "class_path": "silisocs_hooks.RecentOnlyMemory", "params": {}},
        "initialization": {"agents": {"built_in": "none"},
                           "simulation": {"built_in": "none"}},
        "checkpoint": {"every_n_steps": None, "explicit_steps": [], "auto_resume": False},
        "engine": {
            "executor": "threads",
            "loop": {"built_in": None, "class_path": "silisocs_hooks.HookedLoop"},
            "turn_policy": {"built_in": "single_action"},
            "participation": {"built_in": None,
                              "class_path": "silisocs_hooks.ScheduleParticipation",
                              "params": {}}}}
    evalc = {"probes": {"deployment": {"enabled": False}, "probes": {}}}
    (conf / "world").mkdir(parents=True, exist_ok=True)
    (conf / "world" / "default.yaml").write_text(
        "# @package _global_\n" + yaml.safe_dump(world, sort_keys=False))
    for name, d in [("agents", agents), ("env", env), ("sim", sim), ("eval", evalc)]:
        (conf / f"{name}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tool-mode", choices=["single", "none"], default="single",
                    help="single = SiliSocS tool calling (tool_choice=required); "
                         "none = SiliSocS text-parsing fallback")
    ap.add_argument("--scripted", action="store_true",
                    help="SiliSocS scripted provider (no server; smoke test only)")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    t_start = time.time()
    n = int(cfg["num_agents"])
    assert len(cfg["agent_names"]) == len(cfg["personas"]) == len(cfg["agent_models"]) == n
    assert len(set(cfg["agent_names"])) == n, "agent_names must be unique"
    out_dir = Path(cfg["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("contexts.jsonl", "turns.jsonl", "actions.jsonl"):
        (out_dir / f).unlink(missing_ok=True)
    raw = out_dir / "silisocs_raw"
    shutil.rmtree(raw, ignore_errors=True)  # fresh run (auto_resume is off too)
    conf = raw / "conf"
    notes = [f"silisocs==0.4.0 twitter_like, tool_mode={args.tool_mode}",
             "feed: follower_chronological over predefined mutual follows, "
             f"timeline_posts=feed_size, read live at turn time (same-round posts visible)",
             f"NativeAgent observation_history={OBS_HISTORY}, memory section disabled "
             "(RecentOnlyMemory; native window memory re-rendered the same observations)",
             f"actions: {ACTIONS}; reply/like ids must be real post ids"]
    if args.scripted:
        notes.append("SCRIPTED provider (no LLM)")
    else:
        served = wait_for_server(cfg["base_url"])
        missing = sorted(set(cfg["agent_models"]) - set(served))
        if missing:
            notes.append(f"WARNING: models not listed by server: {missing}")
    build_scenario(cfg, conf, args.tool_mode, args.scripted)
    # Hooks read this copy (absolute out_dir; the SiliSocS subprocess runs in raw/).
    hook_cfg = raw / "run_config.resolved.json"
    hook_cfg.write_text(json.dumps({**cfg, "out_dir": str(out_dir)}))

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(HERE)] + [p for p in [env.get("PYTHONPATH")] if p]),
        "CROSSSIM_RUN_CONFIG": str(hook_cfg),
        # SiliSocS defaults to 50 retries with 5-30 s backoff (looks like a hang).
        "SIM_LLM_MAX_RETRIES": env.get("SIM_LLM_MAX_RETRIES", "4"),
        "SIM_LLM_BACKOFF_BASE_SECONDS": env.get("SIM_LLM_BACKOFF_BASE_SECONDS", "2"),
        "SIM_LLM_BACKOFF_MAX_SECONDS": env.get("SIM_LLM_BACKOFF_MAX_SECONDS", "10"),
        "HYDRA_FULL_ERROR": "1",
    })
    with open(raw / "silisocs_stdout.log", "w") as log:
        rc = subprocess.call([sys.executable, "-m", "silisocs.runtime.runner",
                              "--config-path", str(conf)], cwd=raw, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        notes.append(f"ERROR: silisocs exited with code {rc}; see silisocs_raw/silisocs_stdout.log")

    # ---- post-process ----------------------------------------------------- #
    run_dirs = sorted((d for d in raw.glob("outputs/crosssim/*/crosssim_*") if d.is_dir()),
                      key=os.path.getmtime)
    events = read_jsonl(run_dirs[-1] / "action_events.jsonl") if run_dirs else []
    by_turn = {}  # (episode, name) -> committed backend events of that turn
    for e in events:
        if e.get("label") in OK_LABELS:
            by_turn.setdefault((int(e.get("episode", -1)), e.get("source_user")), []).append(e)
    turns = read_jsonl(out_dir / "turns.jsonl")
    stats = {"turns": 0, "failed": 0, "posts": 0}
    with open(out_dir / "actions.jsonl", "w") as f:
        for t in sorted(turns, key=lambda t: (t["round"], t["agent"])):
            evs = by_turn.get((t["round"] - 1, t["name"]), [])
            ok = bool(evs) and t["error"] is None
            typ = OK_LABELS[evs[0]["label"]] if evs else (t["type"] or "error")
            text = (evs[0].get("data", {}).get("post_text") if evs else None) or t["text"]
            rec = {"round": t["round"], "agent": t["agent"], "type": typ, "text": text, "ok": ok}
            if not ok:
                rec["error"] = t["error"] or ("not committed: " + t["resolved"])[:300]
            stats["turns"] += 1
            stats["failed"] += int(not ok)
            stats["posts"] += int(ok and typ in ("create_tweet", "reply_to_tweet"))
            f.write(json.dumps(rec) + "\n")
    expected = sum(len(set(s)) for s in cfg["active_schedule"][: int(cfg["rounds"])])
    if stats["turns"] < expected:  # turns that never reached run_agent_step
        notes.append(f"{expected - stats['turns']} scheduled turns missing; counted as failed")
        stats["failed"] += expected - stats["turns"]
        stats["turns"] = expected
    ctx_rounds = {r["round"] for r in read_jsonl(out_dir / "contexts.jsonl")}
    if 0 not in ctx_rounds:
        notes.append("ERROR: no round-0 contexts written")
    meta = {"run_id": cfg["run_id"], "simulator": "silisocs",
            "wall_seconds": round(time.time() - t_start, 2),
            "n_turns": stats["turns"], "n_failed_turns": stats["failed"],
            "n_posts": stats["posts"], "notes": notes}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return 0 if rc == 0 and 0 in ctx_rounds else 1


if __name__ == "__main__":
    sys.exit(main())
