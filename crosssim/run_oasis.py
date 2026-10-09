"""OASIS runner for the cross-simulator study (see crosssim/RUNNER_SPEC.md).

Requires: Python 3.10/3.11, `pip install camel-oasis==0.2.5 "mcp<2" networkx`.

Design: native OASIS twitter-like platform (one SocialAgent per config agent,
ids 0..N-1, each with its own camel VLLM backend = agent_models[i]). Agents act
ONLY through native tool calls (OASIS has no text parsing); tool_choice
"required" is forwarded through camel's model_config_dict. Actions:
create_post / create_comment / like_post / do_nothing. Follow graph = mutual
FOLLOW ManualActions at setup. Recsys "random" with refresh_rec_post_count=1;
the followee part of Platform.refresh is patched to return the NEWEST followee
posts. Topic seed = user-role memory message before round 1 (not a post).
Scrambled: SocialEnvironment.get_posts_env is patched per agent and round to
show feed_size seeded corpus posts in OASIS's native template.
"""
import argparse
import asyncio
import json
import os
import random
import sqlite3
import sys
import time
import traceback

TIMEOUT_S = 120.0       # per HTTP request
HTTP_RETRIES = 1        # openai-client retries per request
AGENT_ATTEMPTS = 2      # camel ChatAgent retry_attempts around the model call
CONTEXT_TOKENS = 4500   # agent memory budget (max_model_len 8192)
CONTEXT_CHARS = 6000
FAKE_POST_ID0 = 100000
FAKE_USER_ID0 = 200000
ACTION_TYPES = ("create_post", "create_comment", "like_post", "do_nothing")


def load_cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        return json.load(f)


def main():
    cfg = load_cfg()
    out_dir = os.path.abspath(cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)  # oasis/camel create ./log at import time -> keep it per run
    db_path = os.path.join(out_dir, "oasis.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    os.environ["OASIS_DB_PATH"] = db_path
    t0 = time.time()
    notes = []
    try:
        stats = asyncio.run(run(cfg, db_path, notes))
    except Exception:
        traceback.print_exc()
        notes.append("FATAL: " + traceback.format_exc()[-2000:])
        stats = {"n_turns": 0, "n_failed_turns": 0, "n_posts": 0}
        write_meta(cfg, out_dir, t0, stats, notes)
        sys.exit(1)
    write_meta(cfg, out_dir, t0, stats, notes)


def write_meta(cfg, out_dir, t0, stats, notes):
    meta = {"run_id": cfg["run_id"], "simulator": "oasis",
            "wall_seconds": round(time.time() - t0, 2), **stats, "notes": notes}
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def build_classes():
    """Imports deferred until after chdir(out_dir)."""
    from camel.utils import BaseTokenCounter
    from oasis import Platform

    class ApproxCounter(BaseTokenCounter):
        """~3 chars/token (conservative for Llama/Qwen); avoids tiktoken's
        runtime download of o200k_base."""

        def count_tokens_from_messages(self, messages):
            return sum(len(str(m)) for m in messages) // 3 + 1

        def encode(self, text):
            return list(range(len(text) // 3 + 1))

        def decode(self, token_ids):
            return ""

    class NewestFollowPlatform(Platform):
        """Platform.refresh (oasis 0.2.5, platform.py:258) with one change:
        followee posts are ordered newest first (post_id DESC) instead of by
        num_likes; result posts are returned newest first."""

        async def refresh(self, agent_id: int):
            from oasis.social_platform.typing import ActionType
            current_time = self.sandbox_clock.get_time_step()
            try:
                user_id = agent_id
                self.pl_utils._execute_db_command(
                    "SELECT post_id FROM rec WHERE user_id = ?", (user_id, ))
                post_ids = [row[0] for row in self.db_cursor.fetchall()]
                selected = post_ids
                if len(selected) >= self.refresh_rec_post_count:
                    selected = random.sample(selected,
                                             self.refresh_rec_post_count)
                self.pl_utils._execute_db_command(
                    "SELECT post.post_id FROM post "
                    "JOIN follow ON post.user_id = follow.followee_id "
                    "WHERE follow.follower_id = ? "
                    "ORDER BY post.post_id DESC LIMIT ?",
                    (user_id, self.following_post_count))
                following = [row[0] for row in self.db_cursor.fetchall()]
                selected = list(dict.fromkeys(following + selected))
                ph = ", ".join("?" for _ in selected)
                self.pl_utils._execute_db_command(
                    "SELECT post_id, user_id, original_post_id, content, "
                    "quote_content, created_at, num_likes, num_dislikes, "
                    f"num_shares FROM post WHERE post_id IN ({ph}) "
                    "ORDER BY post_id DESC", selected)
                results = self.db_cursor.fetchall()
                if not results:
                    return {"success": False, "message": "No posts found."}
                posts = self.pl_utils._add_comments_to_posts(results)
                self.pl_utils._record_trace(user_id, ActionType.REFRESH.value,
                                            {"posts": posts}, current_time)
                return {"success": True, "posts": posts}
            except Exception as e:
                return {"success": False, "error": str(e)}

    return ApproxCounter, NewestFollowPlatform


def _boilerplate():
    """Constant OASIS observation wrapper text (agent.py:127-131,
    agent_environment.py:40-52) stripped from context dumps so that more
    history fits in CONTEXT_CHARS. The feed text itself is kept verbatim."""
    from oasis.social_agent.agent_environment import SocialEnvironment as SE
    prefix = ("Please perform social media actions after observing the "
              "platform environments. Notice that don't limit your "
              "actions for example to just like the posts. "
              "Here is your social media environment: ")
    groups = SE.groups_env_template.substitute(
        all_groups="{}", joined_groups="[]", messages="{}") + "\n"
    suffix = SE.env_template.template.split("$posts_env")[1]
    return [prefix, groups, suffix]


def flatten_context(agent):
    """Agent's working context (camel memory as sent to the LLM), system
    message excluded, newest last, last CONTEXT_CHARS chars."""
    strip = _boilerplate()
    msgs, _ = agent.memory.get_context()
    lines = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(str(c.get("text", c)) if isinstance(c, dict)
                               else str(c) for c in content)
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                lines.append(f"ASSISTANT (action): {fn.get('name')}"
                             f"({fn.get('arguments')})")
            if content:
                lines.append(f"ASSISTANT: {content}")
        elif role == "tool":
            lines.append(f"TOOL RESULT: {content}")
        elif content:
            if role == "user":
                for b in strip:
                    content = content.replace(b, "")
                content = content.strip()
            lines.append(f"{str(role).upper()}: {content}")
    text = "\n".join(lines)
    return text[-CONTEXT_CHARS:]


def tool_text(name, args):
    if name in ("create_post", "create_comment"):
        return str(args.get("content", ""))
    if name == "like_post":
        return str(args.get("post_id", ""))
    return ""


async def run(cfg, db_path, notes):
    from camel.memories import ChatHistoryMemory, ScoreBasedContextCreator
    from camel.messages import BaseMessage
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType, OpenAIBackendRole
    import oasis
    from oasis import (ActionType, AgentGraph, LLMAction, ManualAction,
                       SocialAgent, UserInfo)
    from oasis.social_platform.channel import Channel

    ApproxCounter, NewestFollowPlatform = build_classes()
    n = int(cfg["num_agents"])
    feed_size = int(cfg.get("feed_size", 5))
    scrambled = cfg["stimulus"] == "scrambled"
    seed = int(cfg["seed"])
    random.seed(seed)  # platform's random rec sampling
    out_dir = os.getcwd()

    backends = {}

    def backend(name):
        if name not in backends:
            backends[name] = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM, model_type=name,
                url=cfg["base_url"], token_counter=ApproxCounter(),
                timeout=TIMEOUT_S, max_retries=HTTP_RETRIES,
                model_config_dict={"temperature": float(cfg.get("temperature", 0.7)),
                                   "max_tokens": int(cfg.get("max_tokens", 160)),
                                   "tool_choice": "required"})
        return backends[name]

    def fresh_memory(agent):
        agent.memory = ChatHistoryMemory(
            ScoreBasedContextCreator(agent.model_backend.token_counter,
                                     CONTEXT_TOKENS), agent_id=agent.agent_id)

    actions = [ActionType.CREATE_POST, ActionType.CREATE_COMMENT,
               ActionType.LIKE_POST, ActionType.DO_NOTHING]
    graph = AgentGraph()
    agents = []
    for i in range(n):
        name = cfg["agent_names"][i]
        persona = cfg["personas"][i]
        ui = UserInfo(user_name=name, name=name, description=persona,
                      profile={"other_info": {"user_profile": persona}},
                      recsys_type="twitter")
        a = SocialAgent(agent_id=i, user_info=ui, agent_graph=graph,
                        model=backend(cfg["agent_models"][i]),
                        available_actions=actions)
        a.retry_attempts = AGENT_ATTEMPTS
        graph.add_agent(a)
        agents.append(a)

    # Record each LLM turn's outcome (env.step discards return values).
    turn_results = {}

    def wrap(agent):
        orig = agent.perform_action_by_llm

        async def wrapped():
            try:
                res = await orig()
            except Exception as e:  # perform_action_by_llm already catches
                res = e
            turn_results[agent.social_agent_id] = res
            return res
        agent.perform_action_by_llm = wrapped
    for a in agents:
        wrap(a)

    platform = NewestFollowPlatform(
        db_path=db_path, channel=Channel(), recsys_type="random",
        refresh_rec_post_count=1,
        following_post_count=max(feed_size - 1, 0),
        max_rec_post_len=max(feed_size, 2))
    env = oasis.make(agent_graph=graph, platform=platform,
                     database_path=db_path)
    await env.reset()

    # Setup step (sandbox time 0): mutual follows from undirected edges.
    follows = {}
    for u, v in cfg["edges"]:
        for x, y in ((u, v), (v, u)):
            follows.setdefault(agents[x], []).append(
                ManualAction(ActionType.FOLLOW, {"followee_id": int(y)}))
            graph.add_edge(int(x), int(y))
    if follows:
        await env.step(follows)
    con = sqlite3.connect(db_path)
    n_follow = con.execute("SELECT COUNT(*) FROM follow").fetchone()[0]
    notes.append(f"follow rows={n_follow} (expected {2 * len(cfg['edges'])})")
    # ManualActions leave "Agent i performed follow ..." notes in memory:
    # start every agent from a clean memory, then add the topic seed.
    for a in agents:
        fresh_memory(a)
        a.update_memory(BaseMessage.make_user_message(
            role_name="User",
            content="Trending on the platform: " + cfg["topic_seed"]),
            OpenAIBackendRole.USER)

    fctx = open(os.path.join(out_dir, "contexts.jsonl"), "w")
    fact = open(os.path.join(out_dir, "actions.jsonl"), "w")

    def dump_contexts(r):
        for i, a in enumerate(agents):
            fctx.write(json.dumps({"round": r, "agent": i,
                                   "model": cfg["agent_models"][i],
                                   "context": flatten_context(a)}) + "\n")
        fctx.flush()

    survey_rounds = set(int(x) for x in cfg["survey_rounds"])
    if 0 in survey_rounds:
        dump_contexts(0)

    corpus = cfg.get("scrambled_posts") or []
    authors = {}
    for p in corpus:
        authors.setdefault(p.get("author", ""), FAKE_USER_ID0 + len(authors))

    def scrambled_posts_env(agent_idx, r, now):
        rng = random.Random(seed * 100003 + r * 1009 + agent_idx)
        picks = rng.sample(range(len(corpus)), min(feed_size, len(corpus)))
        posts = [{"post_id": FAKE_POST_ID0 + k, "user_id": authors[corpus[k].get("author", "")],
                  "content": corpus[k]["text"], "created_at": now,
                  "num_likes": 0, "num_dislikes": 0, "num_shares": 0,
                  "num_reports": 0, "comments": []} for k in picks]
        env_cls = type(agents[agent_idx].env)

        async def get_posts_env():
            if not posts:
                return "After refreshing, there are no existing posts."
            return env_cls.posts_env_template.substitute(
                posts=json.dumps(posts, indent=4))
        return get_posts_env

    n_turns = n_failed = 0
    if scrambled:
        notes.append("scrambled: SocialEnvironment.get_posts_env patched per "
                     "agent/round; fake post_ids >= %d" % FAKE_POST_ID0)
    for r in range(1, int(cfg["rounds"]) + 1):
        active = [int(i) for i in cfg["active_schedule"][r - 1]]
        now = env.platform.sandbox_clock.time_step
        if scrambled:
            for i in active:
                agents[i].env.get_posts_env = scrambled_posts_env(i, r, now)
        turn_results.clear()
        try:
            await env.step({agents[i]: LLMAction() for i in active})
        except Exception as e:
            notes.append(f"round {r}: env.step raised {e!r}")
        for i in active:
            n_turns += 1
            res = turn_results.get(i)
            recs = []
            err = None
            if isinstance(res, BaseException):
                err = f"exception: {res!r}"[:300]
            elif res is None:  # perform_action_by_llm returns None if no tool call
                err = "no tool call"
            else:
                recs = (getattr(res, "info", None) or {}).get("tool_calls") or []
                if not recs:
                    err = "no tool call"
            if err:
                n_failed += 1
                fact.write(json.dumps({"round": r, "agent": i, "type": "none",
                                       "text": "", "ok": False,
                                       "error": err}) + "\n")
                continue
            turn_ok = False
            for k, rec in enumerate(recs):
                name = rec.tool_name
                args = rec.args or {}
                result = rec.result
                ok = (name in ACTION_TYPES and isinstance(result, dict)
                      and bool(result.get("success")))
                turn_ok = turn_ok or ok
                row = {"round": r, "agent": i, "type": name,
                       "text": tool_text(name, args), "ok": ok}
                if name in ("create_comment", "like_post"):
                    row["post_id"] = args.get("post_id")
                if not ok:
                    row["error"] = str(result)[:300]
                if k > 0:
                    row["extra_call"] = True
                fact.write(json.dumps(row) + "\n")
            if not turn_ok:
                n_failed += 1
        fact.flush()
        if r in survey_rounds:
            dump_contexts(r)

    await env.close()
    fctx.close()
    fact.close()
    con = sqlite3.connect(db_path)
    n_posts = con.execute(
        "SELECT COUNT(*) FROM post WHERE original_post_id IS NULL").fetchone()[0]
    by_action = dict(con.execute(
        "SELECT action, COUNT(*) FROM trace GROUP BY action").fetchall())
    con.close()
    notes.append(f"trace action counts: {by_action}")
    notes.append(f"tool_choice=required forwarded via model_config_dict; "
                 f"memory budget {CONTEXT_TOKENS} approx tokens")
    return {"n_turns": n_turns, "n_failed_turns": n_failed, "n_posts": n_posts}


if __name__ == "__main__":
    main()
