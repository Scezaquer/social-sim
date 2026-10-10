"""Out-of-band dual-order survey for cross-simulator runs.

Mirrors the paper's survey instrument (src/simulation_components/entities.py):
the question is appended as a user turn after the agent's context, the chat
template is applied with a generation prompt, and each option is scored by the
summed log-probability of its full token sequence as a continuation.  Both the
canonical and the option-order-flipped phrasing are asked; we record both
choices, all option log-probs and margins.  Surveys never touch agent state.

Scoring uses the vLLM OpenAI-compatible /v1/completions endpoint with
prompt_logprobs on the token ids of prompt+option (so token boundaries are
exact and identical to the paper's HF computation).

Usage:
    python crosssim/survey.py --run_dir RUN_DIR --base_url URL --hf_model NAME
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import time
import urllib.request
from pathlib import Path

CHATML = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] "
          "+ '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")

CTX_TOKENS = 1500  # survey conditions on the most recent CTX_TOKENS tokens of agent context
SYSTEM = ("You are a user on a social media platform. Answer in character, consistently with the "
          "following persona: {persona}")
CTX_HEADER = "Here is what you have recently seen and done on the platform:\n"


class DummyTokenizer:
    """Test-only tokenizer: one token per whitespace-separated chunk (keeps trailing spaces)."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _tok(self, text):
        import re
        return re.findall(r"\S+\s*|\s+", text)

    def __call__(self, text, add_special_tokens=True):
        ids = [self.vocab.setdefault(t, len(self.vocab) + 10) for t in self._tok(text)]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, "") for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def load_tokenizer(hf_model: str, name: str | None = None):
    name = name or hf_model
    if hf_model == "dummy":
        return DummyTokenizer()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_model, local_files_only=True)
    # Same template rule as the paper's simulator (unsloth_model.py).
    if tok.chat_template is None or "Qwen" in name or "Minitaur" in name:
        tok.chat_template = CHATML
    return tok


def post_json(url: str, payload: dict, timeout: float = 180.0, retries: int = 4) -> dict:
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


class Surveyor:
    def __init__(self, tok, base_url: str):
        self.tok = tok
        self.url = base_url.rstrip("/") + "/completions"

    def ids(self, text):
        out = self.tok(text)
        return list(out["input_ids"])

    def truncate_ctx(self, text: str) -> str:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= CTX_TOKENS:
            return text
        return "..." + self.tok.decode(ids[-CTX_TOKENS:])

    def build_prompt(self, persona: str, context: str, question: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM.format(persona=persona)}]
        if context.strip():
            msgs.append({"role": "user", "content": CTX_HEADER + context})
        msgs.append({"role": "user", "content": question})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score_option(self, model: str, prompt: str, option: str) -> float:
        p_ids = self.ids(prompt)
        f_ids = self.ids(prompt + option)
        n_resp = max(1, len(f_ids) - len(p_ids))
        res = post_json(self.url, {"model": model, "prompt": f_ids, "max_tokens": 1, "temperature": 0.0,
                                   "prompt_logprobs": 1})
        plp = res["choices"][0].get("prompt_logprobs")
        if plp is None:
            raise RuntimeError("server returned no prompt_logprobs")
        total = 0.0
        for pos in range(len(f_ids) - n_resp, len(f_ids)):
            entry = plp[pos] or {}
            tid = str(f_ids[pos])
            if tid in entry:
                lp = entry[tid]
            elif len(entry) == 1:
                lp = next(iter(entry.values()))
            else:
                raise RuntimeError(f"token {tid} missing from prompt_logprobs at {pos}")
            total += lp["logprob"] if isinstance(lp, dict) else float(lp)
        return total

    def ask(self, model, persona, context, question, options):
        prompt = self.build_prompt(persona, context, question)
        lps = {o: self.score_option(model, prompt, o) for o in options}
        choice = max(options, key=lambda o: lps[o])
        others = [v for o, v in lps.items() if o != choice]
        margin = lps[choice] - max(others) if others else 0.0
        return choice, margin, lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--hf_model", required=True, help="HF id used for the tokenizer/chat template, or 'dummy'")
    ap.add_argument("--model_name", default=None, help="HF repo id (for the chat-template rule) if --hf_model is a path")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "contexts.jsonl").read_text().splitlines() if l.strip()]
    sv = Surveyor(load_tokenizer(args.hf_model, args.model_name), args.base_url)
    q, qf, options = cfg["question"], cfg["flipped_question"], cfg["options"]

    def work(row):
        i = row["agent"]
        model = cfg["agent_models"][i]
        ctx = sv.truncate_ctx(row.get("context") or "")
        try:
            c, m, lp = sv.ask(model, cfg["personas"][i], ctx, q, options)
            cflip, mflip, lpflip = sv.ask(model, cfg["personas"][i], ctx, qf, options)
            return row["round"], i, {"choice": c, "margin": m, "logprobs": lp, "choice_flipped": cflip,
                                     "margin_flipped": mflip, "logprobs_flipped": lpflip,
                                     "order_consistent": c == cflip}
        except Exception as exc:  # noqa: BLE001
            return row["round"], i, {"error": str(exc)}

    t0 = time.time()
    by_round: dict[int, dict] = {}
    n_err = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r, i, res in ex.map(work, rows):
            by_round.setdefault(r, {})[str(i)] = res
            n_err += "error" in res
    surveys = []
    for r in sorted(by_round):
        ok = {a: v for a, v in by_round[r].items() if "error" not in v}
        surveys.append({"round": r,
                        "results": {a: v["choice"] for a, v in ok.items()},
                        "results_flipped": {a: v["choice_flipped"] for a, v in ok.items()},
                        "detail": by_round[r]})
    out = {"run_id": cfg["run_id"], "question_number": cfg["question_number"], "options": options,
           "n_errors": n_err, "wall_seconds": time.time() - t0, "surveys": surveys}
    (run_dir / "surveys.json").write_text(json.dumps(out))
    print(f"[survey] {cfg['run_id']}: {len(rows)} agent-rounds, {n_err} errors, {time.time() - t0:.0f}s")
    if n_err > 0.2 * max(1, len(rows)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
