#!/bin/bash
#SBATCH --job-name=V2Topology
#SBATCH --array=5, 7, 8, 10, 13, 14, 24, 28, 32, 34, 36, 38, 39, 42, 43, 47, 50, 55, 56, 61, 62, 69, 89, 92, 99, 105, 106, 109, 110, 115, 121, 123, 125, 127, 134, 135, 137, 139, 141, 147, 153, 161, 162, 168, 172, 175, 177, 178, 179, 182, 184, 185, 193, 201, 205, 207, 208, 212, 217, 218, 219, 222, 226, 231, 233, 241, 242, 247, 250, 259, 263, 266, 269, 277, 287, 288, 290, 295, 297, 306, 314, 323, 327, 333
#SBATCH --time=48:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E3: controlled topology study (336 runs, N=256).
# topology(7: ER, powerlaw-cluster, Barabasi-Albert, SBM, forest fire,
# fully-connected, cycle) x model(4) x question(3) x 4 replicates, everything
# else fixed. Parameterizable topologies are density-matched (mean degree ~16);
# empirical mean degree is logged per run for the density-covariate check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/v2_run_common.sh" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	found="$(find "$SLURM_SUBMIT_DIR" -maxdepth 4 -type f -name 'v2_run_common.sh' -print -quit 2>/dev/null || true)"
	if [[ -n "$found" ]]; then
		SCRIPT_DIR="$(dirname "$found")"
	fi
fi
DESIGN_CSV="$SCRIPT_DIR/designs/e3_topology.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
