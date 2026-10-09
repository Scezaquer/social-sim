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

echo "[setup] vllm==$VLLM_VERSION"
mk vllm 3.12
uv pip install -q -p "$ENVS/vllm/bin/python" "vllm==$VLLM_VERSION"

echo "[setup] tools (prep / survey / analysis)"
mk tools 3.12
uv pip install -q -p "$ENVS/tools/bin/python" "transformers>=4.46" jinja2 numpy pandas scipy statsmodels networkx datasets tabulate

echo "[setup] oasis"
mk oasis 3.11
uv pip install -q -p "$ENVS/oasis/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q -p "$ENVS/oasis/bin/python" "camel-oasis==0.2.5" "mcp<2" networkx

echo "[setup] concordia"
mk concordia 3.12
uv pip install -q -p "$ENVS/concordia/bin/python" "gdm-concordia[openai]==2.4.0" networkx

echo "[setup] silisocs"
mk silisocs 3.12
uv pip install -q -p "$ENVS/silisocs/bin/python" "silisocs==0.4.0" networkx

for e in oasis concordia silisocs; do
  mod=$e; [[ $e == concordia ]] && mod=concordia
  "$ENVS/$e/bin/python" -c "import $mod; print('[setup] import $mod ok')"
done
"$ENVS/vllm/bin/python" -c "import vllm; print('[setup] vllm', vllm.__version__)"

echo "[setup] personas + designs"
"$ENVS/tools/bin/python" crosssim/prep.py personas
"$ENVS/tools/bin/python" crosssim/prep.py design --out "$CROSSSIM_SMOKE" --smoke --n_agents 16 --rounds 2 --survey_every 1 --seeds 1 --questions 29
"$ENVS/tools/bin/python" crosssim/prep.py design --out "$CROSSSIM_OUT"

# Check that base models and all LoRAs are where job.sh expects them.
for fam in qwen llama3.1; do
  source crosssim/slurm/models.sh "$fam"
  ls "$HF_HUB_CACHE" | grep -q "models--${HF_MODEL//\//--}" && echo "[setup] $HF_MODEL in HF cache" || echo "[setup] WARNING: $HF_MODEL not found in $HF_HUB_CACHE"
  missing=0; for i in $(seq 0 24); do [[ -f "$(lora_path $i)/adapter_config.json" ]] || missing=$((missing+1)); done
  echo "[setup] $fam: $((25-missing))/25 LoRA adapters found under $LORA_DIR"
done
echo "[setup] done. Next: sbatch --array=0-5 --export=ALL,DESIGN=$CROSSSIM_SMOKE crosssim/slurm/job.sh"
