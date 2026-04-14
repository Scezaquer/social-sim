#!/bin/bash
#SBATCH --job-name=KnowingSurvey
#SBATCH --array=0
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
    --survey_output survey_knowing_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json  \
    --array_id ${SLURM_ARRAY_TASK_ID} \
    --job_id ${SLURM_JOB_ID} \
    --homophily \
    --question_number 28 \
    --tweet_files ai_copyright_tweets.json \
    --visualizer_output visualizer_knowing_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json \
    --add_survey_to_context