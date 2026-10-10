#!/bin/bash
# Creates a second, newer vLLM env ($ENVS/vllm2) used for Minitaur only. Safe to run while jobs are
# running: it never touches the existing envs. Run on a LOGIN node.
set -euo pipefail
cd "$(dirname "$0")/../.."
source crosssim/slurm/env.sh
V2="${VLLM2_VERSION:-0.12.0}"
[[ -x "$ENVS/vllm2/bin/python" ]] || uv venv -q -p 3.12 "$ENVS/vllm2"
uv pip install -q -p "$ENVS/vllm2/bin/python" "vllm==$V2" "transformers>=4.56,<5" \
  || uv pip install -q -p "$ENVS/vllm2/bin/python" "vllm==$V2"
"$ENVS/vllm2/bin/python" -c "import vllm, transformers; print('[vllm2] vllm', vllm.__version__, 'transformers', transformers.__version__)"
"$ENVS/vllm2/bin/vllm" serve --help=LoRAConfig 2>/dev/null | grep -i -c "extra-vocab" | sed 's/^/[vllm2] lora extra-vocab options (want 0): /' || true
