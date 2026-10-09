#!/usr/bin/env bash
# Integration test: prep.py smoke design -> each runner (stub server) -> survey.py (logprob stub) -> analyze.py
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; X="$(cd "$T/.." && pwd)"
PY_CONC=${PY_CONC:-python}; PY_SILI=${PY_SILI:-python}; PY_OASIS=${PY_OASIS:-python}; PY_TOOLS=${PY_TOOLS:-python3}
OUT=${OUT:-/tmp/xsim_integration}; rm -rf "$OUT"
$PY_TOOLS "$X/prep.py" design --out "$OUT" --smoke --n_agents 8 --rounds 3 --survey_every 1 --seeds 1 --questions 29 --families qwen >/dev/null
python3 "$T/concordia/stub_server.py" 18501 >/dev/null 2>&1 & P1=$!
python3 "$T/oasis/stub_server.py" 18502 >/dev/null 2>&1 & P2=$!
python3 "$T/silisocs/stub_server.py" 18503 >/dev/null 2>&1 & P3=$!
python3 "$T/survey/stub_logprob_server.py" 18504 >/dev/null 2>&1 & P4=$!
trap 'kill $P1 $P2 $P3 $P4 2>/dev/null' EXIT; sleep 2
declare -A PORT=([concordia]=18501 [oasis]=18502 [silisocs]=18503)
declare -A PY=([concordia]=$PY_CONC [oasis]=$PY_OASIS [silisocs]=$PY_SILI)
for rdir in "$OUT"/runs/*; do
  sim=$(basename "$rdir" | cut -d_ -f1)
  $PY_TOOLS - "$rdir/run_config.json" "http://127.0.0.1:${PORT[$sim]}/v1" <<'PYEOF'
import json,sys; p,u=sys.argv[1:]; c=json.load(open(p)); c["base_url"]=u; json.dump(c,open(p,"w"))
PYEOF
  ${PY[$sim]} "$X/run_$sim.py" --config "$rdir/run_config.json" > "$rdir/runner.log" 2>&1; rc=$?
  $PY_TOOLS "$X/survey.py" --run_dir "$rdir" --base_url http://127.0.0.1:18504/v1 --hf_model dummy > "$rdir/survey.log" 2>&1; rs=$?
  echo "$(basename $rdir): runner=$rc survey=$rs $(grep -o '"n_failed_turns": [0-9]*' $rdir/run_meta.json 2>/dev/null) ctx_lines=$(wc -l < $rdir/contexts.jsonl 2>/dev/null)"
done
$PY_TOOLS "$X/analyze.py" --root "$OUT" --out "$OUT/results" --n_perm 20 > "$OUT/analyze.log" 2>&1; echo "analyze=$?"
