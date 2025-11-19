#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus-per-node=1

module load python/3.11
module load scipy-stack

echo "Activating virtual environment..."
source ../concordia/ENV-concordia/bin/activate
export HF_HUB_CACHE="/home/s4yor1/scratch/HF-cache"
export HF_HUB_OFFLINE=1
echo "Starting simulation..."
python src/main.py
# pip freeze --local
