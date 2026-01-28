#!/bin/bash
#SBATCH --job-name=PowerLaw
#SBATCH --array=0-7
#SBATCH --time=8:00:00
#SBATCH --mem=16Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
export UNSLOTH_CACHE_DIR=$SLURM_TMPDIR/unsloth-cache
python src/main_unsloth.py --loras_path $SCRATCH/qwen-loras --survey_output survey_powerlaw_${SLURM_ARRAY_TASK_ID}.json