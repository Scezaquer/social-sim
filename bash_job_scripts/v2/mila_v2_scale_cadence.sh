#!/bin/bash
#SBATCH --job-name=V2Scale
#SBATCH --array=0-71
#SBATCH --time=48:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E5: scale study with per-agent-matched survey cadence (72 runs).
# N(64/256/1024) x model(Minitaur, Qwen) x question(3) x 4 reps with
# survey_interval = N and max_steps = 8N, so the expected number of actions per
# agent between surveys is constant across scales. Tests whether V1's
# OSR-vs-scale trend was a cadence artifact.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/v2_run_common.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	found="$(find "$SLURM_SUBMIT_DIR" -maxdepth 4 -type f -name 'v2_run_common.sh' -print -quit 2>/dev/null || true)"
	if [[ -n "$found" ]]; then
		SCRIPT_DIR="$(dirname "$found")"
	fi
fi
DESIGN_CSV="$SCRIPT_DIR/designs/e5_scale_cadence.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
