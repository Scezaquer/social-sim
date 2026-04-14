#!/bin/bash
#SBATCH --job-name=Distribution
#SBATCH --array=0-4
#SBATCH --time=5:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
export UNSLOTH_CACHE_DIR=$SLURM_TMPDIR/unsloth-cache

python -u src/main_unsloth.py \
     --loras_path $SCRATCH/marcelbinz \
      --survey_output survey_distribution_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
      --array_id ${SLURM_ARRAY_TASK_ID} \
      --job_id ${SLURM_JOB_ID} \
      --homophily \
      --question_number 28 \
      --tweet_files ai_copyright_tweets.json \
      --proportions 0.08284974529253673 0 6.031361684446227e-14 1.3711747453009344e-13 1.4871144523281122e-13 0.019194578110743203 0.2492881841131293 0 0.31911160366705815 1.3501072995125138e-15 1.6834892806694375e-13 0.02589749033958161 1.2423799577408848e-13 0 0.020674655085774138 1.1662445445124316e-13 0 0 1.215824871946409e-13 0.05281378426193962 9.317469100983314e-14 0.059985687449042396 0.16185494294730227 0 0.008329328720798508