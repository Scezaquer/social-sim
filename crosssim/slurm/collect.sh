#!/bin/bash
# Run on the Mila login node when jobs are done (or partially done):
#   bash crosssim/slurm/collect.sh            -> analysis in crosssim/results + crosssim_results.tgz
# The tarball (no raw contexts) is small enough to scp back / commit.
set -euo pipefail
cd "$(dirname "$0")/../.."
source crosssim/slurm/env.sh
DESIGN="${DESIGN:-$CROSSSIM_OUT}"
echo "DONE runs: $(ls "$DESIGN"/runs/*/DONE 2>/dev/null | wc -l) / $(ls -d "$DESIGN"/runs/* | wc -l)"
"$ENVS/tools/bin/python" crosssim/analyze.py --root "$DESIGN" --out crosssim/results --n_perm 200
mkdir -p cookbook_rebuttal/v3/Tables
tar czf crosssim_results.tgz crosssim/results \
  -C "$DESIGN" jobs.csv server_logs \
  $(cd "$DESIGN" && ls runs/*/{surveys.json,run_meta.json,run_config.json,actions.jsonl,runner.log,survey.log} 2>/dev/null)
cp crosssim/results/table_xsim_summary.tex cookbook_rebuttal/v3/Tables/xsim_summary.tex 2>/dev/null || true
echo "wrote crosssim_results.tgz ($(du -h crosssim_results.tgz | cut -f1))"
