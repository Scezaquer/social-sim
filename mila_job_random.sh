#!/bin/bash
#SBATCH --job-name=RandomGraph
#SBATCH --array=0-4
#SBATCH --time=24:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=unkillable

module load python/3.10
source $HOME/ENV/bin/activate
export HF_HUB_CACHE=$SCRATCH/HF-cache
export UNSLOTH_CACHE_DIR=$SLURM_TMPDIR/unsloth-cache

python -u src/main_unsloth.py \
     --loras_path $SCRATCH/marcelbinz \
      --random_graph \ 
      --survey_output survey_random_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
      --array_id ${SLURM_ARRAY_TASK_ID} \
      --job_id ${SLURM_JOB_ID} \
      --homophily \
      --question_number 28 \
      --tweet_files ai_copyright_tweets.json
