#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --time=10:15:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-node=a100:1

START_TIME=$(date +%s)
nvidia-smi
lscpu

mkdir -p $SLURM_TMPDIR/model
echo "Syncing model and LoRAs to local storage..."
SYNC_START=$(date +%s)
echo "Using rsync to copy files..."
rsync -av /home/s4yor1/scratch/qwen-loras $SLURM_TMPDIR/
rsync -av /home/s4yor1/scratch/HF-cache/models--Qwen--Qwen2.5-7B-Instruct $SLURM_TMPDIR/model
SYNC_END=$(date +%s)
SYNC_DURATION=$((SYNC_END - SYNC_START))
echo "Syncing completed in ${SYNC_DURATION} seconds"

module load python/3.11
module load scipy-stack

echo "Activating virtual environment..."
source ../concordia/ENV-concordia/bin/activate
export HF_HUB_CACHE="$SLURM_TMPDIR/model"
export HF_HUB_OFFLINE=1
echo "Starting simulation..."
python src/main.py --start_time $START_TIME --duration 36000 --loras_path $SLURM_TMPDIR/qwen-loras
# pip freeze --local
