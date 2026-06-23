#!/bin/bash
#SBATCH --job-name=V2Core
#SBATCH --array=101,144,156,169,176,208,221,259,260,283,298,305,356,369,371,400,430,439,451,457,473,488,491,574
#SBATCH --time=48:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E1: balanced core design-space sweep (576 runs).
# model(4) x proportions(4) x question(3) x N(64/256/1024) x graph(2)
# x homophily(2) x ctx(2) x news(0/1/4) x activity_exponent(0/.5/1).
# Every run dual-order surveys (order-consistency + log-prob margins recorded).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If the script was copied into Slurm's spool dir,the common file won't be
# next to the running script. Try to locate the repository copy under the
# submission directory if the common file is missing.
if [[ ! -f "$SCRIPT_DIR/v2_run_common.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	found="$(find "$SLURM_SUBMIT_DIR" -maxdepth 4 -type f -name 'v2_run_common.sh' -print -quit 2>/dev/null || true)"
	if [[ -n "$found" ]]; then
		SCRIPT_DIR="$(dirname "$found")"
	fi
fi
DESIGN_CSV="$SCRIPT_DIR/designs/e1_core_sweep.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
