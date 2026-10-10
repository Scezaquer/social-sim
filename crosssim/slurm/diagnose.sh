#!/bin/bash
# Print the key error lines of a design (default: smoke) so they can be pasted back.
#   bash crosssim/slurm/diagnose.sh [DESIGN_DIR]
cd "$(dirname "$0")/../.."; source crosssim/slurm/env.sh
D="${1:-$CROSSSIM_SMOKE}"
echo "### vLLM server logs: first error per failed attempt"
for f in "$D"/server_logs/*.log; do
  [[ -f "$f" ]] || continue
  e=$(grep -m1 -E "Error|error:|OutOfMemory|CUDA out of memory" "$f")
  [[ -n "$e" ]] && echo "$(basename "$f"): $e"
done
echo; echo "### per run: runner exit, failed turns, survey errors"
for r in "$D"/runs/*; do
  [[ -f "$r/run_config.json" ]] || continue
  printf '%s | DONE=%s | ' "$(basename "$r")" "$([[ -f $r/DONE ]] && echo y || echo n)"
  "$ENVS/tools/bin/python" - "$r" <<'PY'
import json, sys, os, collections
r = sys.argv[1]
m = json.load(open(f"{r}/run_meta.json")) if os.path.exists(f"{r}/run_meta.json") else {}
print(f"turns={m.get('n_turns')} failed={m.get('n_failed_turns')} posts={m.get('n_posts')}", end=" | ")
errs = collections.Counter()
if os.path.exists(f"{r}/actions.jsonl"):
    for l in open(f"{r}/actions.jsonl"):
        a = json.loads(l)
        if not a.get("ok", True):
            errs[str(a.get("error") or a.get("type"))[:120]] += 1
print("action errors:", dict(errs.most_common(3)), end=" | ")
if os.path.exists(f"{r}/surveys.json"):
    s = json.load(open(f"{r}/surveys.json"))
    first = next((v["error"] for x in s["surveys"] for v in x["detail"].values() if "error" in v), None)
    print(f"survey errors={s.get('n_errors')}", (f"first: {first[:400]}" if first else ""))
else:
    print("no surveys.json")
PY
done
