#!/bin/bash
#SBATCH --job-name=xsim
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/xsim_%A_%a.out
#
# One array task = one (simulator, model family, question, seed) job from
# $DESIGN/jobs.csv. Starts a vLLM server with the base model + its 25 BluePrint
# LoRAs, then runs the job's simulations in order; each finished simulation is
# surveyed in the background while the next one runs. Re-submitting skips runs
# that already have a DONE marker.
#
#   sbatch --array=0-23 --export=ALL,DESIGN=$SCRATCH/crosssim_out crosssim/slurm/job.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source crosssim/slurm/env.sh
DESIGN="${DESIGN:-$CROSSSIM_OUT}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-9000}"   # seconds per simulation run

row=$(awk -v n=$((TASK + 2)) 'NR == n' "$DESIGN/jobs.csv" | tr -d '\r')
[[ -z "$row" ]] && { echo "no job row $TASK in $DESIGN/jobs.csv"; exit 1; }
IFS=, read -r JOB_INDEX SIM FAM Q SEED RUNS <<< "$row"
source crosssim/slurm/models.sh "$FAM" || {
  echo "FATAL: model family '$FAM' in $DESIGN/jobs.csv is not defined in crosssim/slurm/models.sh."
  echo "The design is probably stale (generated before a code change). Regenerate it with crosssim/prep.py design (see crosssim/README.md)."
  exit 1; }
echo "== job $JOB_INDEX: sim=$SIM family=$FAM q=$Q seed=$SEED host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

TMP="${SLURM_TMPDIR:-/tmp}/xsim_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMP"
PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=EMPTY

# Local copy of the base model (searches all HF caches; downloads if missing and online).
if [[ -n "${XSIM_TOKENIZER:-}" ]]; then MODEL_PATH="$HF_MODEL"; else
  MODEL_PATH=$("$ENVS/tools/bin/python" crosssim/resolve_model.py "$HF_MODEL" --download) \
    || { echo "FATAL: base model $HF_MODEL not available (run crosssim/slurm/fetch_models.sh on a login node)"; exit 1; }
fi
echo "base model: $HF_MODEL -> $MODEL_PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------- vLLM server
TEMPLATE="$TMP/chat_template.jinja"
TOKENIZER="${XSIM_TOKENIZER:-$MODEL_PATH}"   # override only in tests
PARSER=$("$ENVS/tools/bin/python" crosssim/make_template.py "$TOKENIZER" "$TEMPLATE" "$HF_MODEL" | tail -1)
[[ -s "$TEMPLATE" && -n "$PARSER" ]] || { echo "FATAL: chat template generation failed"; exit 1; }
LORA_ARGS=(); MAX_RANK=16
for i in $(seq 0 24); do
  p=$(lora_path "$i")
  [[ -f "$p/adapter_config.json" ]] || { echo "missing LoRA $p"; exit 1; }
  fixed="$SCRATCH/crosssim_loras/$FAM/lora$i"     # vLLM-ready copy made by crosssim/fix_loras.py
  [[ -f "$fixed/adapter_config.json" ]] && p="$fixed"
  LORA_ARGS+=("lora$i=$p")
done
MAX_RANK=$("$ENVS/tools/bin/python" - "$LORA_DIR" <<'PY'
import json, sys, glob
rs = [json.load(open(f)).get("r", 16) for f in glob.glob(sys.argv[1] + "/*/adapter_config.json")]
r = max(rs) if rs else 16
print(min(x for x in (8, 16, 32, 64, 128, 256) if x >= r))
PY
)
echo "vLLM env: ${VLLM_ENV:-vllm} ($("$ENVS/${VLLM_ENV:-vllm}/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null))"
echo "chat template parser=$PARSER max_lora_rank=$MAX_RANK lora0=${LORA_ARGS[0]}"

start_server() {   # $1 = attempt number
  local extra=() envs=()
  case "$1" in
    1) ;;
    2) envs=(VLLM_ATTENTION_BACKEND=TRITON_ATTN); extra=(--enforce-eager) ;;
    *) envs=(VLLM_ATTENTION_BACKEND=FLEX_ATTENTION); extra=(--enforce-eager) ;;
  esac
  [[ "$PARSER" != "none" ]] && extra+=(--enable-auto-tool-choice --tool-call-parser "$PARSER")
  env "${envs[@]}" "$ENVS/${VLLM_ENV:-vllm}/bin/vllm" serve "$MODEL_PATH" \
    --served-model-name base --dtype half --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --enable-lora --max-loras 25 --max-cpu-loras 25 --max-lora-rank "$MAX_RANK" \
    --lora-modules "${LORA_ARGS[@]}" --chat-template "$TEMPLATE" \
    --max-logprobs 20 --port "$PORT" "${extra[@]}" \
    > "$TMP/vllm_attempt$1.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 240); do   # up to 20 min (cold load from network FS)
    sleep 5
    curl -sf "$BASE_URL/models" >/dev/null && { echo "vLLM up (attempt $1, pid $SERVER_PID)"; return 0; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
  done
  echo "vLLM attempt $1 failed; tail of log:"; tail -40 "$TMP/vllm_attempt$1.log"
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
  return 1
}
ATTEMPT=0
ensure_server() {
  [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null && curl -sf "$BASE_URL/models" >/dev/null && return 0
  while (( ATTEMPT < 3 )); do
    ATTEMPT=$((ATTEMPT + 1))
    start_server "$ATTEMPT" && return 0
  done
  return 1
}
ensure_server || { echo "FATAL: could not start vLLM"; exit 1; }
mkdir -p "$DESIGN/server_logs"
save_logs() { for f in "$TMP"/vllm_attempt*.log; do [[ -f "$f" ]] && cp "$f" "$DESIGN/server_logs/job${JOB_INDEX}_${SLURM_JOB_ID:-x}_$(basename "$f")"; done; }
trap 'kill $SERVER_PID 2>/dev/null; save_logs' EXIT

# Quick functional check: one completion with prompt_logprobs on a LoRA, one tool call.
"$ENVS/tools/bin/python" - "$BASE_URL" <<'PY'
import json, sys, urllib.request
u = sys.argv[1]
def post(path, body):
    req = urllib.request.Request(u + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())
r = post("/completions", {"model": "lora0", "prompt": "Hello, my name is", "max_tokens": 8})
print("[check] completion:", repr(r["choices"][0]["text"]))
# forced-token logprob must come from the raw distribution (strongly negative for an unlikely token)
r = post("/completions", {"model": "lora0", "prompt": "The capital of France is", "max_tokens": 1, "temperature": 0,
                          "logprobs": 1, "allowed_token_ids": [12345]})
print("[check] forced-token logprob (should be << 0):", r["choices"][0]["logprobs"]["token_logprobs"][0])
tools = [{"type": "function", "function": {"name": "create_post", "description": "Create a post",
          "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}}}]
try:
    r = post("/chat/completions", {"model": "lora0", "messages": [{"role": "user", "content": "Write a short post about the weather."}],
                                   "tools": tools, "tool_choice": "required", "max_tokens": 80})
    print("[check] tool call:", json.dumps(r["choices"][0]["message"].get("tool_calls"))[:300])
except Exception as e:
    print("[check] tool call FAILED:", e)
PY

# ---------------------------------------------------------------- runs
SURVEY_PIDS=()
IFS=';' read -ra RUN_LIST <<< "$RUNS"
for run in "${RUN_LIST[@]}"; do
  rdir="$DESIGN/runs/$run"
  [[ -f "$rdir/DONE" ]] && { echo "skip $run (done)"; continue; }
  ensure_server || { echo "server down, aborting"; break; }
  "$ENVS/tools/bin/python" - "$rdir/run_config.json" "$BASE_URL" <<'PY'
import json, sys
p, u = sys.argv[1:]
c = json.load(open(p)); c["base_url"] = u; json.dump(c, open(p, "w"))
PY
  for attempt in 1 2; do   # one retry: transient import/FS errors on the network filesystem
    rm -f "$rdir/contexts.jsonl" "$rdir/actions.jsonl"
    echo "[$(date +%T)] run $run (attempt $attempt)"
    t0=$(date +%s)
    ( cd "$rdir" && timeout "$RUN_TIMEOUT" "$ENVS/$SIM/bin/python" "$OLDPWD/crosssim/run_$SIM.py" \
        --config "$rdir/run_config.json" ${RUNNER_ARGS:-} ) > "$rdir/runner.log" 2>&1
    rc=$?
    echo "[$(date +%T)] runner exit=$rc after $(( $(date +%s) - t0 ))s; $(cat "$rdir/run_meta.json" 2>/dev/null | head -c 300)"
    [[ -s "$rdir/contexts.jsonl" ]] && break
    ensure_server || break
  done
  if [[ -s "$rdir/contexts.jsonl" ]]; then
    ( for sa in 1 2; do
        for _ in $(seq 1 90); do curl -sf "$BASE_URL/models" >/dev/null && break; sleep 10; done   # main loop restarts a dead server
        "$ENVS/tools/bin/python" crosssim/survey.py --run_dir "$rdir" --base_url "$BASE_URL" --hf_model "$TOKENIZER" \
            --model_name "$HF_MODEL" --workers "${SURVEY_WORKERS:-12}" > "$rdir/survey.log" 2>&1 && { touch "$rdir/DONE"; break; }
      done; tail -3 "$rdir/survey.log" ) &
    SURVEY_PIDS+=($!)
  else
    echo "no contexts for $run; tail runner.log:"; tail -20 "$rdir/runner.log"
  fi
done
for p in "${SURVEY_PIDS[@]}"; do wait "$p"; done
echo "== job $JOB_INDEX finished: $(ls "$DESIGN"/runs/*/DONE 2>/dev/null | wc -l) runs DONE in design"
