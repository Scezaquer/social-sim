#!/bin/bash
#SBATCH --job-name=RandomGraph
#SBATCH --array=0
#SBATCH --time=7:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
python src/main.py --loras_path $SCRATCH/qwen-loras