#!/bin/bash
#SBATCH --job-name=V2Noise
#SBATCH --array=0-95
#SBATCH --time=6:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E4: measurement-validity baselines (96 runs, N=256).
# stimulus(none/scrambled) x ctx(2) x model(4) x question(3) x 2 reps.
# Requires SCRAMBLED_CORPUS for the scrambled rows: a simulation_threads_*.json
# from any E1 run (use a different model/question than the row being run if you
# want maximal independence), e.g.:
#   export SCRAMBLED_CORPUS="$PWD/simulation_threads_<jobid>_<arrayid>.json"
#   sbatch bash_job_scripts/v2/mila_v2_noise_floor.sh

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
DESIGN_CSV="$SCRIPT_DIR/designs/e4_noise_floor.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
