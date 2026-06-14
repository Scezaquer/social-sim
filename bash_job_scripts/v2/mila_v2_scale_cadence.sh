#!/bin/bash
#SBATCH --job-name=V2Scale
#SBATCH --array=0-71
#SBATCH --time=48:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E5: scale study with per-agent-matched survey cadence (72 runs).
# N(64/256/1024) x model(Minitaur, Qwen) x question(3) x 4 reps with
# survey_interval = N and max_steps = 8N, so the expected number of actions per
# agent between surveys is constant across scales. Tests whether V1's
# OSR-vs-scale trend was a cadence artifact.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_CSV="$SCRIPT_DIR/designs/e5_scale_cadence.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
