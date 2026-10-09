#!/usr/bin/env bash
# End-to-end tests for crosssim/run_silisocs.py (needs silisocs==0.4.0 in $PY's env).
# 1) scripted provider (no server), 2) stub OpenAI server with tool calls, 3) --tool-mode none.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"; PY="${PY:-python}"
STUB_LOG="$HERE/stub_requests.jsonl" python3 "$HERE/stub_server.py" 18432 & STUB=$!
trap 'kill $STUB' EXIT; sleep 1
for m in normal scrambled; do
  $PY "$ROOT/run_silisocs.py" --config "$HERE/run_config_$m.json" --scripted
  $PY "$ROOT/run_silisocs.py" --config "$HERE/run_config_${m}_toolnone.json" --tool-mode none
  $PY "$ROOT/run_silisocs.py" --config "$HERE/run_config_$m.json"
done
for d in "$HERE"/out_*; do echo "$d"; cat "$d/run_meta.json" | head -8; done
