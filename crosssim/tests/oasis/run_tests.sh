#!/usr/bin/env bash
# End-to-end tests for crosssim/run_oasis.py against the stub OpenAI server (tool calls).
# Needs PY = python with camel-oasis==0.2.5, "mcp<2", networkx (Python 3.10/3.11).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"; PY="${PY:-python}"
rm -f "$HERE/stub_requests.jsonl"
STUB_LOG="$HERE/stub_requests.jsonl" python3 "$HERE/stub_server.py" 18433 & STUB=$!
trap 'kill $STUB' EXIT; sleep 1
for m in normal scrambled; do
  $PY "$ROOT/run_oasis.py" --config "$HERE/run_config_$m.json" > "$HERE/out_$m.log" 2>&1
  echo "$m exit=$?"; cat "$HERE/out_$m/run_meta.json"
done
