"""SiliSocS (silisocs==0.4.0) extension classes for crosssim/run_silisocs.py.

Imported by the generated SiliSocS scenario through `class_path` entries, so this
module must be importable in the SiliSocS subprocess (run_silisocs.py puts its own
directory on PYTHONPATH). All shared state comes from the run_config.json named by
$CROSSSIM_RUN_CONFIG.

* ScheduleParticipation -- sim.engine.participation: agents in active_schedule[step]
  (SiliSocS step_index is 0-based; round r = step_index + 1).
* CrossSimAgent         -- NativeAgent subclass; in the scrambled condition it
  replaces every feed observation with posts sampled from scrambled_posts.
* HookedLoop            -- the fixed_steps loop, plus context dumps (round 0 and
  survey rounds) and a per-turn log (turns.jsonl) used to build actions.jsonl.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from typing import Any

from silisocs.agents.memory import WindowMemory
from silisocs.agents.native import NativeAgent
from silisocs.simulation_engines.policies.loops import FixedStepsLoopStrategy
from silisocs.simulation_engines.policies.participation import ParticipationPolicy

CONTEXT_CHARS = 6000
TIMELINE_PREFIXES = ("STARTING SOCIAL MEDIA SESSION", "## Timeline")

_CFG: dict[str, Any] | None = None
_LOCK = threading.Lock()
# The round being run; set by HookedLoop at each step boundary (single-threaded).
CURRENT_ROUND = {"r": 0}


def cfg() -> dict[str, Any]:
    global _CFG
    if _CFG is None:
        with open(os.environ["CROSSSIM_RUN_CONFIG"], encoding="utf-8") as f:
            _CFG = json.load(f)
        _CFG["_name_to_idx"] = {n: i for i, n in enumerate(_CFG["agent_names"])}
    return _CFG


def _out(name: str) -> str:
    return os.path.join(cfg()["out_dir"], name)


def _append_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
    with _LOCK, open(_out(name), "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------- #
class RecentOnlyMemory(WindowMemory):
    """Window memory whose prompt 'Memory' section is empty.

    NativeAgent records every observation in BOTH its recent-observation list and its
    memory, and the default window memory renders the last 10 of them again, so the
    prompt carried each observation twice. (render_count=0 does not work in 0.4.0:
    `list[-0:]` renders everything.)
    """

    def render(self, *, query: str | None = None) -> str:
        del query
        return ""


# --------------------------------------------------------------------------- #
class ScheduleParticipation(ParticipationPolicy):
    """Participation = the precomputed active_schedule (shared by all simulators)."""

    name = "crosssim_schedule"

    def __init__(self, **_ignored: object) -> None:
        pass

    def participating_agents(self, *, agent_names, step_index, seed):
        del seed
        c = cfg()
        schedule = c["active_schedule"]
        if not 0 <= int(step_index) < len(schedule):
            return []
        present = set(agent_names)
        wanted = [c["agent_names"][int(i)] for i in dict.fromkeys(schedule[int(step_index)])]
        return [n for n in wanted if n in present]


# --------------------------------------------------------------------------- #
def format_scrambled_feed(agent_idx: int, rnd: int) -> str:
    c = cfg()
    pool = c.get("scrambled_posts") or []
    k = min(int(c.get("feed_size", 5)), len(pool))
    rng = random.Random(int(c.get("seed", 0)) * 100003 + rnd * 1009 + agent_idx)
    picks = rng.sample(range(len(pool)), k)
    body = "".join(
        f"\n\nUser: {pool[j]['author']}\n"
        f"Content: {pool[j]['text']}\n"
        f"Tweet ID: ext-{j}\n"  # deliberately unresolvable: no coupling via ids
        f"Likes: 0, Reposts: 0, Replies: 0\n"
        for j in picks
    )
    return "Your timeline:\n" + body


class CrossSimAgent(NativeAgent):
    """NativeAgent; scrambled runs swap feed observations for the scrambled corpus."""

    def observe(self, observation: str) -> None:
        text = str(observation or "").strip()
        if text and cfg().get("stimulus") == "scrambled" and text.startswith(TIMELINE_PREFIXES):
            idx = cfg()["_name_to_idx"].get(self.name, 0)
            observation = format_scrambled_feed(idx, CURRENT_ROUND["r"])
        super().observe(observation)

    def crosssim_context(self) -> str:
        """Agent-native working context minus Instructions/Persona/World/Goal/Style."""
        query = self._observations[-1] if self._observations else None
        sections = [
            ("Recent observations", "\n".join(self._observations[-self._observation_history:])),
            ("Memory", self._memory.render(query=query)),
        ]
        text = "\n\n".join(f"{k}:\n{v.strip()}" for k, v in sections if v.strip())
        return text[-CONTEXT_CHARS:]


# --------------------------------------------------------------------------- #
def _dump_contexts(rnd: int, agents: list[Any]) -> None:
    c = cfg()
    rows = []
    for a in agents:
        i = c["_name_to_idx"][a.name]
        getter = getattr(a, "crosssim_context", None)
        text = getter() if callable(getter) else str(a._context())[-CONTEXT_CHARS:]
        rows.append({"round": rnd, "agent": i, "model": c["agent_models"][i], "context": text})
    rows.sort(key=lambda r: r["agent"])
    _append_jsonl("contexts.jsonl", rows)


def _describe(raw: Any) -> tuple[str, str]:
    calls = list(getattr(raw, "tool_calls", ()) or ())
    if calls:
        args = dict(calls[0].arguments or {})
        return calls[0].name, str(args.get("status") or args.get("content") or "")
    return "text", str(getattr(raw, "text", "") or raw or "")


class HookedLoop(FixedStepsLoopStrategy):
    """fixed_steps + crosssim dumps. Survey/round-0 dumps never touch agent memory."""

    name = "crosssim_hooked_fixed_steps"

    def run(self, *, engine, game_masters, agents, **kw):
        c = cfg()
        survey = {int(r) for r in c.get("survey_rounds", [])}
        orig_step, orig_turn = engine.run_step, engine.run_agent_step

        def run_agent_step(**kwargs):
            agent = kwargs.get("agent")
            rec = {"round": CURRENT_ROUND["r"], "agent": c["_name_to_idx"].get(agent.name, -1),
                   "name": agent.name, "type": "", "text": "", "resolved": "", "error": None}
            t0 = time.time()
            try:
                res = orig_turn(**kwargs)
                rec["type"], rec["text"] = _describe(res.raw_action)
                rec["resolved"] = str(res.resolved_result)[:500]
                return res
            except Exception as exc:  # recorded, then re-raised to SiliSocS isolation
                rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
                raise
            finally:
                rec["seconds"] = round(time.time() - t0, 3)
                _append_jsonl("turns.jsonl", [rec])

        def run_step(*, step_index, game_masters, agents, verbose):
            if step_index == 0:
                # Interventions (the topic-seed broadcast) already fired; nobody has acted.
                _dump_contexts(0, agents)
            CURRENT_ROUND["r"] = int(step_index) + 1
            res = orig_step(step_index=step_index, game_masters=game_masters,
                            agents=agents, verbose=verbose)
            if CURRENT_ROUND["r"] in survey:
                _dump_contexts(CURRENT_ROUND["r"], agents)
            return res

        engine.run_step, engine.run_agent_step = run_step, run_agent_step
        return super().run(engine=engine, game_masters=game_masters, agents=agents, **kw)
