#!/bin/bash
#SBATCH --job-name=V2Core
#SBATCH --array=0-575
#SBATCH --time=24:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long

# E1: balanced core design-space sweep (576 runs).
# model(4) x proportions(4) x question(3) x N(64/256/1024) x graph(2)
# x homophily(2) x ctx(2) x news(0/1/4) x activity_exponent(0/.5/1).
# Every run dual-order surveys (order-consistency + log-prob margins recorded).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESIGN_CSV="$SCRIPT_DIR/designs/e1_core_sweep.csv"
source "$SCRIPT_DIR/v2_run_common.sh"
