#!/usr/bin/env python3
"""Leakage-safe training of the human-vs-LLM thread classifier (V2_PLAN Priority 1).

Reviewer-blocking fixes over V1:
- Split is by SIMULATION RUN (file) on the AI side and by USER on the human
  side, so no thread from a training run/user can appear in the test set.
- The split manifest is saved next to the model so eval_detector.py can verify
  that per-run evaluation only touches held-out runs.
- Threads are rendered identically for both classes (author-stripped by
  default) so the classifier cannot key on name formatting.

Inputs:
- --sim-threads-glob: simulation_threads_*.json files from V2 runs
  ([{"id": ..., "messages": [{"role", "content"}, ...]}, ...]).
- --human-data: JSON or JSONL of human threads. Accepted shapes per record:
  {"messages": [{"role"/"author", "content"/"text"}, ...]} or a bare list of
  such message dicts. Use a BluePrint export.

Usage:
    python analysis/train_bert_detector.py \
        --sim-threads-glob 'simulation_threads_*.json' \
        --human-data blueprint_threads.json \
        --base-model Twitter/twhin-bert-base --outdir detector_out
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys


def render_thread(messages, include_authors=False, max_chars=2000):
    lines = []
    for m in messages:
        content = (m.get("content") or m.get("text") or "").strip()
        if not content:
            continue
        if include_authors:
            author = m.get("role") or m.get("author") or "user"
            lines.append(f"{author}: {content}")
        else:
            lines.append(content)
    return "\n".join(lines)[:max_chars]


def load_sim_threads(pattern, min_messages):
    """Returns {run_key: [thread_text, ...]}."""
    runs = {}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                threads = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            continue
        texts = []
        for t in threads:
            messages = t.get("messages") if isinstance(t, dict) else t
            if not messages or len(messages) < min_messages:
                continue
            text = render_thread(messages)
            if text:
                texts.append(text)
        if texts:
            runs[os.path.basename(path)] = texts
    return runs


def load_human_threads(path, min_messages):
    """Returns {user_key: [thread_text, ...]}, keyed by the thread's first author."""
    records = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
    by_user = {}
    for rec in records:
        messages = rec.get("messages") if isinstance(rec, dict) else rec
        if not messages or len(messages) < min_messages:
            continue
        first = messages[0]
        user = str(first.get("role") or first.get("author") or "unknown")
        text = render_thread(messages)
        if text:
            by_user.setdefault(user, []).append(text)
    return by_user


def grouped_split(groups: dict, test_frac, rng):
    """Splits group keys (runs or users), never individual samples."""
    keys = sorted(groups)
    rng.shuffle(keys)
    n_test = max(1, int(len(keys) * test_frac))
    test_keys = set(keys[:n_test])
    train, test = [], []
    for k, texts in groups.items():
        (test if k in test_keys else train).extend(texts)
    return train, test, sorted(test_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-threads-glob", required=True)
    parser.add_argument("--human-data", required=True)
    parser.add_argument("--base-model", default="Twitter/twhin-bert-base")
    parser.add_argument("--outdir", default="detector_out")
    parser.add_argument("--min-messages", type=int, default=2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sim_runs = load_sim_threads(args.sim_threads_glob, args.min_messages)
    human_users = load_human_threads(args.human_data, args.min_messages)
    print(f"AI runs: {len(sim_runs)} ({sum(map(len, sim_runs.values()))} threads); "
          f"human users: {len(human_users)} ({sum(map(len, human_users.values()))} threads)")
    if not sim_runs or not human_users:
        sys.exit("Need both AI and human threads.")

    ai_train, ai_test, ai_test_keys = grouped_split(sim_runs, args.test_frac, rng)
    hu_train, hu_test, hu_test_keys = grouped_split(human_users, args.test_frac, rng)

    for split in (ai_train, ai_test, hu_train, hu_test):
        rng.shuffle(split)
    ai_train, hu_train = ai_train[:args.max_per_class], hu_train[:args.max_per_class]

    manifest = {
        "base_model": args.base_model,
        "seed": args.seed,
        "test_runs": ai_test_keys,
        "test_users": hu_test_keys,
        "counts": {
            "ai_train": len(ai_train), "ai_test": len(ai_test),
            "human_train": len(hu_train), "human_test": len(hu_test),
        },
    }
    with open(os.path.join(args.outdir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Split manifest:", json.dumps(manifest["counts"]))

    # Heavy imports after data validation.
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    def make_ds(human, ai):
        texts = human + ai
        labels = [0] * len(human) + [1] * len(ai)  # 1 = AI-generated
        ds = Dataset.from_dict({"text": texts, "label": labels}).shuffle(seed=args.seed)
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=512),
            batched=True, remove_columns=["text"],
        )

    train_ds = make_ds(hu_train, ai_train)
    test_ds = make_ds(hu_test, ai_test)

    def metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.outdir, "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=100,
            seed=args.seed,
            report_to=[],
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=metrics_fn,
    )
    trainer.train()
    print("Held-out (run/user-disjoint) evaluation:", trainer.evaluate())

    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"Saved detector + manifest to {args.outdir}")


if __name__ == "__main__":
    main()
