#!/bin/bash
#SBATCH --job-name=V2Topology
#SBATCH --array=0-335
#SBATCH --time=12:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:RTX8000:1
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
