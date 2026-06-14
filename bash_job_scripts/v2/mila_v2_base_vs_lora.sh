#!/bin/bash
#SBATCH --job-name=V2BaseLora
#SBATCH --array=0-143
#SBATCH --time=12:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:RTX8000:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E2: no-LoRA arm for ALL four base models (144 runs, N=256).
# Crossed with E1's LoRA runs this gives the 4(model) x 2(LoRA on/off) design
# that breaks the V1 confound where fine-tuning status was nested under model
# identity (only Qwen had a base-only condition).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/v2_run_common.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	found="$(find "$SLURM_SUBMIT_DIR" -maxdepth 4 -type f -name 'v2_run_common.sh' -print -quit 2>/dev/null || true)"
	if [[ -n "$found" ]]; then
		SCRIPT_DIR="$(dirname "$found")"
	fi
fi
DESIGN_CSV="$SCRIPT_DIR/designs/e2_base_vs_lora.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
