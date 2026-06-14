#!/bin/bash
#SBATCH --job-name=V2Topology
#SBATCH --array=0-335
#SBATCH --time=12:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E3: controlled topology study (336 runs, N=256).
# topology(7: ER, powerlaw-cluster, Barabasi-Albert, SBM, forest fire,
# fully-connected, cycle) x model(4) x question(3) x 4 replicates, everything
# else fixed. Parameterizable topologies are density-matched (mean degree ~16);
# empirical mean degree is logged per run for the density-covariate check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_CSV="$SCRIPT_DIR/designs/e3_topology.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
