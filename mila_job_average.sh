#!/bin/bash
#SBATCH --job-name=Average
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
      --survey_output survey_ai_copyright_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
      --array_id ${SLURM_ARRAY_TASK_ID} \
      --job_id ${SLURM_JOB_ID} \
      --homophily \
      --question_number 28 \
      --tweet_files ai_copyright_tweets.json \
      --proportions 0 0 0 0 0 0 0.27193296965726726 0 0.13913508605256175 0 0 0 0 0 0.2679288109799589 0 0 0 0 0 0 0.024487206302543963 0.20117095954339054 0 0.09534496756853536