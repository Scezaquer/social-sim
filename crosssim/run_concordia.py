"""Concordia runner for the cross-simulator study (see crosssim/RUNNER_SPEC.md).

Requires: gdm-concordia==2.4.0 (Python >= 3.12), numpy, networkx, openai.

Design: no game master. Each agent is a Concordia prefab entity
(`minimal` by default, 1 LLM call per act; `--prefab basic` = 4 calls per act)
with its own AssociativeMemoryBank and its own LanguageModel wrapper pointing
at the served model name agent_models[i] on the vLLM OpenAI-compatible server.
Every round, each active agent observe()s its feed and then act()s with a
free-text "write a post" ActionSpec. Its own post is observed back.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import sys
import threading
import time
import traceback

import networkx as nx
import numpy as np
import openai

from concordia.associative_memory import basic_associative_memory as amem
from concordia.language_model import language_model as lm
from concordia.prefabs.entity import basic as basic_prefab
from concordia.prefabs.entity import minimal as minimal_prefab
from concordia.typing import entity as entity_lib

TIMEOUT_S = 120.0
MAX_RETRIES = 2
HISTORY = 20
CONTEXT_CHARS = 6000
MAX_WORKERS = 32


class VLLMCompletionModel(lm.LanguageModel):
  """Concordia LanguageModel over vLLM /v1/completions (raw prompt)."""

  def __init__(self, model_name, base_url, max_tokens, temperature,
               default_stop=('\n',)):
    self.model_name = model_name
    self.max_tokens = int(max_tokens)
    self.temperature = float(temperature)
    self.stop = list(default_stop)
    self.client = openai.OpenAI(base_url=base_url, api_key='EMPTY',
                                timeout=TIMEOUT_S, max_retries=MAX_RETRIES)

  def sample_text(self, prompt, *, max_tokens=lm.DEFAULT_MAX_TOKENS,
                  terminators=lm.DEFAULT_TERMINATORS,
                  temperature=lm.DEFAULT_TEMPERATURE, top_p=lm.DEFAULT_TOP_P,
                  top_k=lm.DEFAULT_TOP_K, timeout=lm.DEFAULT_TIMEOUT_SECONDS,
                  seed=None):
    # Concordia passes temperature=1.0 and max_tokens up to 2200 by default;
    # the shared config values take precedence.
    del temperature, timeout
    r = self.client.completions.create(
        model=self.model_name, prompt=prompt,
        max_tokens=min(int(max_tokens), self.max_tokens),
        temperature=self.temperature, top_p=top_p, seed=seed,
        stop=list(terminators) or self.stop, extra_body={'top_k': top_k})
    return r.choices[0].text or ''

  def sample_choice(self, prompt, responses, *, seed=None):
    # Not used by this runner (no in-band surveys); logprob scoring of letters.
    r = self.client.completions.create(
        model=self.model_name, prompt=prompt, max_tokens=1, temperature=0.0,
        logprobs=20)
    top = r.choices[0].logprobs.top_logprobs[0]
    scores = {s: max([lp for t, lp in top.items() if t.strip() == s],
                     default=-math.inf) for s in responses}
    idx = max(range(len(responses)), key=lambda i: scores[responses[i]])
    return idx, responses[idx], {'logprobs': scores}


def build_agent(prefab, name, persona, model):
  bank = amem.AssociativeMemoryBank(sentence_embedder=lambda _: np.ones(8))
  if prefab == 'minimal':
    agent = minimal_prefab.Entity(params={
        'name': name,
        'randomize_choices': False,
        'custom_instructions': (
            f'You are {name}, a user of a social media platform.\n{persona}'),
    }).build(model=model, memory_bank=bank)
    agent.get_component('__observation__').set_state(
        {'history_length': HISTORY})
  else:  # basic: no custom-instructions param; persona goes in the goal slot
    agent = basic_prefab.Entity(params={
        'name': name,
        'goal': f'{name} is a user of a social media platform. {persona}',
        'randomize_choices': False,
        'observation_history_length': HISTORY,
    }).build(model=model, memory_bank=bank)
  return agent


def wait_for_server(base_url, deadline_s=300):
  client = openai.OpenAI(base_url=base_url, api_key='EMPTY', timeout=10,
                         max_retries=0)
  t0 = time.time()
  while True:
    try:
      return [m.id for m in client.models.list().data]
    except Exception:  # pylint: disable=broad-except
      if time.time() - t0 > deadline_s:
        raise
      time.sleep(5)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--config', required=True)
  ap.add_argument('--prefab', choices=['minimal', 'basic'], default='minimal')
  args = ap.parse_args()
  with open(args.config) as f:
    cfg = json.load(f)

  t_start = time.time()
  n = int(cfg['num_agents'])
  names, personas, models_ = cfg['agent_names'], cfg['personas'], cfg['agent_models']
  assert len(names) == len(personas) == len(models_) == n
  rounds = int(cfg['rounds'])
  schedule = cfg['active_schedule']
  survey_rounds = set(int(r) for r in cfg['survey_rounds'])
  stimulus = cfg['stimulus']
  feed_size = int(cfg.get('feed_size', 5))
  max_tokens = int(cfg.get('max_tokens', 160))
  temperature = float(cfg.get('temperature', 0.7))
  seed = int(cfg.get('seed', 0))
  scrambled = cfg.get('scrambled_posts', [])
  out_dir = cfg['out_dir']
  os.makedirs(out_dir, exist_ok=True)
  notes = [f'prefab={args.prefab}', f'observation_history_length={HISTORY}',
           'no game master: runner routes feeds and calls entity.observe/act',
           'normal feed built from posts of rounds < r (snapshot at round start)',
           'action space: free-text post only']

  served = wait_for_server(cfg['base_url'])
  missing = sorted(set(models_) - set(served))
  if missing:
    notes.append(f'WARNING: models not listed by server: {missing}')

  g = nx.Graph()
  g.add_nodes_from(range(n))
  g.add_edges_from((int(u), int(v)) for u, v in cfg['edges'])
  neighbours = {i: set(g.neighbors(i)) - {i} for i in range(n)}

  lms = [VLLMCompletionModel(models_[i], cfg['base_url'], max_tokens,
                             temperature) for i in range(n)]
  agents = [build_agent(args.prefab, names[i], personas[i], lms[i])
            for i in range(n)]
  post_spec = entity_lib.free_action_spec(
      call_to_action=('Write the next post {name} publishes on the platform '
                      '(one line, under 60 words).'))

  posts = []  # global post log: (round, agent, text), append order = time
  ctx_f = open(os.path.join(out_dir, 'contexts.jsonl'), 'w')
  act_f = open(os.path.join(out_dir, 'actions.jsonl'), 'w')
  lock = threading.Lock()
  stats = {'turns': 0, 'failed': 0, 'posts': 0}

  def dump_contexts(r):
    for i in range(n):
      mem = agents[i].get_component('__memory__').get_all_memories_as_text()
      text = '\n'.join(mem)[-CONTEXT_CHARS:]
      ctx_f.write(json.dumps({'round': r, 'agent': i, 'model': models_[i],
                              'context': text}) + '\n')
    ctx_f.flush()

  def feed_for(i, r, snapshot):
    if stimulus == 'scrambled':
      rng = random.Random(seed * 100003 + r * 1009 + i)
      k = min(feed_size, len(scrambled))
      return [(p['author'], p['text']) for p in rng.sample(scrambled, k)]
    out = []
    for (_, a, txt) in reversed(snapshot):
      if a != i and a in neighbours[i]:
        out.append((names[a], txt))
        if len(out) >= feed_size:
          break
    return out

  def turn(i, r, snapshot):
    rec = {'round': r, 'agent': i, 'type': 'post', 'text': '', 'ok': False}
    try:
      feed = feed_for(i, r, snapshot)
      if feed:
        obs = f'[round {r}] Your feed: ' + ' | '.join(
            f'{a}: {t}' for a, t in feed)
      else:
        obs = f'[round {r}] Your feed is empty.'
      agents[i].observe(obs)
      raw = agents[i].act(post_spec)
      text = raw.strip()
      if text.startswith(names[i]):
        text = text[len(names[i]):].strip()
      text = text.strip().strip('"').strip()
      if text:
        agents[i].observe(f'[round {r}] You posted: {text}')
        rec.update(text=text, ok=True)
    except Exception as e:  # pylint: disable=broad-except
      rec['error'] = f'{type(e).__name__}: {e}'[:500]
      traceback.print_exc(file=sys.stderr)
    return rec

  # Round 0: topic seed only.
  for a in agents:
    a.observe(f'[round 0] Trending on the platform: {cfg["topic_seed"]}')
  dump_contexts(0)

  round_secs = []
  with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    for r in range(1, rounds + 1):
      t0 = time.time()
      active = list(dict.fromkeys(int(i) for i in schedule[r - 1]))  # unique
      snapshot = list(posts)
      recs = list(ex.map(lambda i: turn(i, r, snapshot), active))
      for rec in recs:  # deterministic order (schedule order)
        stats['turns'] += 1
        if rec['ok']:
          stats['posts'] += 1
          posts.append((r, rec['agent'], rec['text']))
        else:
          stats['failed'] += 1
        act_f.write(json.dumps(rec) + '\n')
      act_f.flush()
      if r in survey_rounds:
        dump_contexts(r)
      round_secs.append(time.time() - t0)
      print(f'[concordia] round {r}/{rounds}: {len(active)} turns, '
            f'{round_secs[-1]:.1f}s', flush=True)

  ctx_f.close()
  act_f.close()
  if round_secs:
    notes.append(f'mean_round_seconds={sum(round_secs)/len(round_secs):.2f}')
  meta = {'run_id': cfg['run_id'], 'simulator': 'concordia',
          'wall_seconds': round(time.time() - t_start, 2),
          'n_turns': stats['turns'], 'n_failed_turns': stats['failed'],
          'n_posts': stats['posts'], 'notes': notes}
  with open(os.path.join(out_dir, 'run_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
  return 0


if __name__ == '__main__':
  sys.exit(main())
