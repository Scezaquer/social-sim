#!/bin/bash
# Progress overview: bash crosssim/slurm/status.sh
cd "$(dirname "$0")/../.."; source crosssim/slurm/env.sh; DESIGN="${DESIGN:-$CROSSSIM_OUT}"
squeue -u "$USER" -n xsim -o "%.18i %.8T %.10M %.20R" 2>/dev/null
for sim in oasis concordia silisocs; do
  tot=$(ls -d "$DESIGN"/runs/${sim}_* 2>/dev/null | wc -l); done_=$(ls "$DESIGN"/runs/${sim}_*/DONE 2>/dev/null | wc -l)
  echo "$sim: $done_ / $tot runs done"
done
grep -h "runner exit" logs/xsim_*.out 2>/dev/null | awk '{print $4}' | sort | uniq -c
grep -l "tool call FAILED\|FATAL" logs/xsim_*.out 2>/dev/null | head
