#!/usr/bin/env python3
"""Per-run and out-of-domain evaluation of the human-vs-LLM detector (Priority 1).

Modes (combinable):
1. Per-run scoring: for every simulation_threads_*.json whose run is in the
   held-out split (or all runs with --include-train-runs, flagged as such),
   computes detector accuracy and writes detector_per_run.json. With
   --annotate-visualizers, writes the score into the matching
   visualizer_v2_*.json under "bert_real_vs_llm_classifier" so
   build_dataset.py picks it up as the bert_accuracy metric.
2. --external-ai-dir: threads from a DIFFERENT simulator (e.g. OASIS-style
   rollouts) -> cross-simulator generalization (R4-Q1, R2-W2).
3. --external-human: held-out human text from users never seen in training ->
   in-domain-bias quantification.
4. --crosscheck-model: an off-the-shelf HF AI-text detector scored on the same
   held-out sample; reports agreement with our classifier (R4-Q2).

Usage:
    python analysis/eval_detector.py --model-dir detector_out \
        --sim-threads-glob 'simulation_threads_*.json' \
        [--annotate-visualizers --visualizer-glob 'visualizer_v2_*.json'] \
        [--external-ai-dir oasis_threads/] [--external-human blueprint_holdout.json] \
        [--crosscheck-model desklib/ai-text-detector-v1.01]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from train_bert_detector import load_human_threads, load_sim_threads


def batch_p_ai(texts, tokenizer, model, device, batch_size=32):
    import torch
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        if logits.shape[-1] == 1:
            p = torch.sigmoid(logits[:, 0])
        else:
            p = torch.softmax(logits, dim=-1)[:, 1]
        probs.extend(p.cpu().tolist())
    return probs


def jeffreys_ci(successes, total, level=0.95):
    from scipy import stats
    lo, hi = stats.beta.interval(level, successes + 0.5, total - successes + 0.5)
    return [float(lo), float(hi)]


def run_key_to_visualizer(run_file: str) -> str | None:
    m = re.match(r"simulation_threads_(\d+)_(\d+)\.json$", os.path.basename(run_file))
    return m.groups() if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sim-threads-glob", default="simulation_threads_*.json")
    parser.add_argument("--visualizer-glob", default="visualizer_v2_*.json")
    parser.add_argument("--annotate-visualizers", action="store_true")
    parser.add_argument("--include-train-runs", action="store_true",
                        help="Also score runs used for detector training (flagged in output; excluded from headline numbers).")
    parser.add_argument("--external-ai-dir", default=None)
    parser.add_argument("--external-human", default=None)
    parser.add_argument("--crosscheck-model", default=None)
    parser.add_argument("--min-messages", type=int, default=2)
    parser.add_argument("--outdir", default="detector_out")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device).eval()

    manifest_path = os.path.join(args.model_dir, "split_manifest.json")
    test_runs = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        test_runs = set(manifest.get("test_runs", []))
    else:
        print("WARNING: no split_manifest.json found; cannot verify leakage-safety.", file=sys.stderr)

    results = {"per_run": [], "external": {}}

    # 1. Per-run accuracy on simulation threads.
    sim_runs = load_sim_threads(args.sim_threads_glob, args.min_messages)
    for run_file, texts in sorted(sim_runs.items()):
        in_test = run_file in test_runs if test_runs else None
        if test_runs and not in_test and not args.include_train_runs:
            continue
        probs = batch_p_ai(texts, tokenizer, model, device)
        correct = int(sum(p >= 0.5 for p in probs))
        entry = {
            "run_file": run_file,
            "n_threads": len(texts),
            "accuracy": correct / len(texts),
            "accuracy_ci95": jeffreys_ci(correct, len(texts)),
            "mean_p_ai": float(np.mean(probs)),
            "held_out": in_test,
        }
        results["per_run"].append(entry)

        if args.annotate_visualizers:
            ids = run_key_to_visualizer(run_file)
            if ids:
                job_id, array_id = ids
                for vis_path in glob.glob(args.visualizer_glob):
                    try:
                        with open(vis_path) as f:
                            vis = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue
                    params = vis.get("run_parameters") or {}
                    if str(params.get("job_id")) == job_id and str(params.get("array_id")) == array_id:
                        vis["bert_real_vs_llm_classifier"] = {
                            "accuracy": entry["accuracy"],
                            "n_threads": entry["n_threads"],
                            "held_out": in_test,
                        }
                        with open(vis_path, "w") as f:
                            json.dump(vis, f)
                        break

    held = [r for r in results["per_run"] if r["held_out"] is not False]
    if held:
        accs = [r["accuracy"] for r in held]
        results["held_out_summary"] = {
            "n_runs": len(held),
            "mean_accuracy": float(np.mean(accs)),
            "std": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        }
        print(f"Held-out runs: {len(held)}, mean per-run accuracy {np.mean(accs):.4f}")

    # 2. Cross-simulator AI text.
    if args.external_ai_dir:
        ext = load_sim_threads(os.path.join(args.external_ai_dir, "*.json"), args.min_messages)
        texts = [t for v in ext.values() for t in v]
        if texts:
            probs = batch_p_ai(texts, tokenizer, model, device)
            correct = int(sum(p >= 0.5 for p in probs))
            results["external"]["cross_simulator_ai"] = {
                "n": len(texts),
                "accuracy": correct / len(texts),
                "accuracy_ci95": jeffreys_ci(correct, len(texts)),
            }
            print(f"Cross-simulator AI text: acc {correct / len(texts):.4f} (n={len(texts)})")

    # 3. Out-of-domain human text (label 0 -> correct means p < 0.5).
    if args.external_human:
        humans = load_human_threads(args.external_human, args.min_messages)
        texts = [t for v in humans.values() for t in v]
        if texts:
            probs = batch_p_ai(texts, tokenizer, model, device)
            correct = int(sum(p < 0.5 for p in probs))
            results["external"]["ood_human"] = {
                "n": len(texts),
                "accuracy": correct / len(texts),
                "accuracy_ci95": jeffreys_ci(correct, len(texts)),
            }
            print(f"OOD human text: acc {correct / len(texts):.4f} (n={len(texts)})")

    # 4. Off-the-shelf detector agreement on held-out sim threads.
    if args.crosscheck_model:
        cc_tok = AutoTokenizer.from_pretrained(args.crosscheck_model)
        cc_model = AutoModelForSequenceClassification.from_pretrained(args.crosscheck_model).to(device).eval()
        sample_texts = []
        for r in held[:50]:
            sample_texts.extend(sim_runs[r["run_file"]][:10])
        if sample_texts:
            ours = np.array(batch_p_ai(sample_texts, tokenizer, model, device)) >= 0.5
            theirs = np.array(batch_p_ai(sample_texts, cc_tok, cc_model, device)) >= 0.5
            results["external"]["crosscheck"] = {
                "model": args.crosscheck_model,
                "n": len(sample_texts),
                "agreement": float((ours == theirs).mean()),
                "ours_flag_rate": float(ours.mean()),
                "theirs_flag_rate": float(theirs.mean()),
            }
            print(f"Cross-check vs {args.crosscheck_model}: agreement {(ours == theirs).mean():.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "detector_per_run.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
