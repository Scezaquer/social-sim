#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1

module load python/3.11
module load scipy-stack

source ../ENV-concordia/bin/activate
export HF_HUB_CACHE="/home/s4yor1/scratch/HF-cache"
export HF_HUB_OFFLINE=1
python src/main.py
# pip freeze --local
