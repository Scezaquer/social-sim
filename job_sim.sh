#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=10:15:00
#SBATCH --mem-per-cpu=125G
#SBATCH --gpus-per-node=1

START_TIME=$(date +%s)

module load python/3.11
module load scipy-stack

echo "Activating virtual environment..."
source ../concordia/ENV-concordia/bin/activate
export HF_HUB_CACHE="/home/s4yor1/scratch/HF-cache"
export HF_HUB_OFFLINE=1
echo "Starting simulation..."
python src/main.py --start_time $START_TIME --duration 36000
# pip freeze --local
