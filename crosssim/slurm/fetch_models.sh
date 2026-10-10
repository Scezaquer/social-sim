#!/bin/bash
# Run on a Mila LOGIN node: makes sure both base models are available locally
# (prints where they were found, downloads them into $HF_HUB_CACHE otherwise).
set -euo pipefail
cd "$(dirname "$0")/../.."
source crosssim/slurm/env.sh
for fam in qwen minitaur; do
  source crosssim/slurm/models.sh "$fam"
  p=$("$ENVS/tools/bin/python" crosssim/resolve_model.py "$HF_MODEL" --download)
  echo "[fetch] $fam: $HF_MODEL -> $p"
  "$ENVS/tools/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$p'); print('[fetch] tokenizer ok')"
done
