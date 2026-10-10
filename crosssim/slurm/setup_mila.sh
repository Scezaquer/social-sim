#!/bin/bash
# One-time setup on a Mila LOGIN node (needs internet). ~15-25 min.
#   cd ~/social-sim && bash crosssim/slurm/setup_mila.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source crosssim/slurm/env.sh
mkdir -p "$ENVS" logs

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mk() { [[ -x "$ENVS/$1/bin/python" ]] || uv venv -q -p "$2" "$ENVS/$1"; }
# install <env> <python> <import-check> <packages...>: skipped when the env already passes its check,
# so re-running setup never touches an env that running jobs may be using.
install() {
  local env=$1 py=$2 check=$3; shift 3
  if [[ -x "$ENVS/$env/bin/python" ]] && "$ENVS/$env/bin/python" -c "$check" 2>/dev/null; then
    echo "[setup] $env: ok (skipped)"; return; fi
  echo "[setup] $env: installing $*"
  local re=(); [[ -x "$ENVS/$env/bin/python" ]] && re=(--reinstall)   # repair a broken/partial env
  mk "$env" "$py"
  uv pip install -q "${re[@]}" -p "$ENVS/$env/bin/python" "$@"
  "$ENVS/$env/bin/python" -c "$check" || { echo "[setup] $env: import check FAILED"; exit 1; }
}

# integrity check: every file recorded by every installed package must exist (catches half-finished installs)
INTACT='import importlib.metadata as m, sys; bad=[str(f) for d in m.distributions() for f in (d.files or []) if f.suffix==".py" and not f.locate().exists()]; sys.exit(1 if bad else 0)'
install vllm 3.12 "import vllm, transformers, uvicorn; $INTACT" "vllm==$VLLM_VERSION"
install tools 3.12 "import transformers, jinja2, pandas, statsmodels, networkx, datasets, tabulate, huggingface_hub; $INTACT" \
  "transformers>=4.46" jinja2 numpy pandas scipy statsmodels networkx datasets tabulate huggingface_hub
if ! "$ENVS/oasis/bin/python" -c "import oasis" 2>/dev/null; then
  mk oasis 3.11
  uv pip install -q -p "$ENVS/oasis/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
fi
install oasis 3.11 "import oasis, networkx" "camel-oasis==0.2.5" "mcp<2" networkx
install concordia 3.12 "import concordia, openai, networkx" "gdm-concordia[openai]==2.4.0" networkx
install silisocs 3.12 "import silisocs, networkx" "silisocs==0.4.0" networkx

echo "[setup] base models"
bash crosssim/slurm/fetch_models.sh

echo "[setup] personas + designs"
"$ENVS/tools/bin/python" crosssim/prep.py personas
"$ENVS/tools/bin/python" crosssim/prep.py design --out "$CROSSSIM_SMOKE" --smoke --n_agents 16 --rounds 2 --survey_every 1 --seeds 1 --questions 29
"$ENVS/tools/bin/python" crosssim/prep.py design --out "$CROSSSIM_OUT"

# Check that base models and all LoRAs are where job.sh expects them.
for fam in qwen minitaur; do
  source crosssim/slurm/models.sh "$fam"
  missing=0; for i in $(seq 0 24); do [[ -f "$(lora_path $i)/adapter_config.json" ]] || missing=$((missing+1)); done
  echo "[setup] $fam: $((25-missing))/25 LoRA adapters found under $LORA_DIR"
done
echo "[setup] done. Next: sbatch --array=0-5 --export=ALL,DESIGN=$CROSSSIM_SMOKE crosssim/slurm/job.sh"
