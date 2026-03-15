#!/bin/bash
#SBATCH --job-name=RandomizedSim
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=main

set -euo pipefail

pick_random() {
    local -n arr_ref=$1
    echo "${arr_ref[$RANDOM % ${#arr_ref[@]}]}"
}

generate_random_proportions() {
    local count=$1
    local values=(1 2 3 5 8 13)
    local result=()
    local i
    for ((i = 0; i < count; i++)); do
        result+=("${values[$RANDOM % ${#values[@]}]}")
    done
    echo "${result[*]}"
}

module load python/3.10
source "$HOME/ENV/bin/activate"
export HF_HUB_CACHE="$SCRATCH/HF-cache"
export UNSLOTH_CACHE_DIR="$SLURM_TMPDIR/unsloth-cache"

QUESTION_CHOICES=(
    "25:genetic_enhancements_tweets.json"
    "28:ai_copyright_tweets.json"
    "29:environmental_protection_tweets.json"
)
GRAPH_CHOICES=("random" "powerlaw")
HOMOPHILY_CHOICES=("on" "off")
SURVEY_CONTEXT_CHOICES=("on" "off")
NUM_AGENTS_CHOICES=(50 500 1000 1500)
NUM_NEWS_AGENTS_CHOICES=(0 1)
MODEL_PROFILE_CHOICES=("minitaur_loras") # "qwen_base" "qwen_loras")

QUESTION_PICK=$(pick_random QUESTION_CHOICES)
QUESTION_NUMBER="${QUESTION_PICK%%:*}"
TWEET_FILE="${QUESTION_PICK#*:}"

GRAPH_TYPE=$(pick_random GRAPH_CHOICES)
HOMOPHILY_FLAG=$(pick_random HOMOPHILY_CHOICES)
SURVEY_CONTEXT_FLAG=$(pick_random SURVEY_CONTEXT_CHOICES)
NUM_AGENTS=$(pick_random NUM_AGENTS_CHOICES)
NUM_NEWS_AGENTS=$(pick_random NUM_NEWS_AGENTS_CHOICES)
MODEL_PROFILE=$(pick_random MODEL_PROFILE_CHOICES)

BASE_MODEL=""
LORAS_PATH=""
LORA_NAME_TEMPLATE=""
NUM_LORAS=0
LORA_INDEX_SET=""
PROPORTIONS=""

if [[ "$MODEL_PROFILE" == "minitaur_loras" ]]; then
    BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
    LORAS_PATH="$SCRATCH/marcelbinz"
    LORA_NAME_TEMPLATE="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
    NUM_LORAS=25
    MINITAUR_INDEX_SETS=(
        "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
        # "0 1 2 3 4"
        # "5 6 7 8 9"
        # "10 11 12 13 14"
        # "15 16 17 18 19"
        # "20 21 22 23 24"
        # "0 3 7 11 18 22"
        # "0 1 2 3 4 5 6 7 8 9"
    )
    LORA_INDEX_SET=$(pick_random MINITAUR_INDEX_SETS)
elif [[ "$MODEL_PROFILE" == "qwen_loras" ]]; then
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    LORAS_PATH="$SCRATCH/Qwen"
    LORA_NAME_TEMPLATE="Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
    NUM_LORAS=10
    QWEN_INDEX_SETS=(
        "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
        # "5 6 7 8 9"
        # "0 2 4 6 8"
        # "1 3 5 7 9"
        # "0 1 2 3 4 5 6 7 8 9"
    )
    LORA_INDEX_SET=$(pick_random QWEN_INDEX_SETS)
else
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
fi

SURVEY_OUTPUT="survey_randomized_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
VISUALIZER_OUTPUT="visualizer_randomized_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

CMD=(
    python -u src/main_unsloth.py
    --survey_output "$SURVEY_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --question_number "$QUESTION_NUMBER"
    --tweet_files "$TWEET_FILE"
    --base_model "$BASE_MODEL"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --visualizer_output "$VISUALIZER_OUTPUT"
)

if [[ "$GRAPH_TYPE" == "random" ]]; then
    CMD+=(--random_graph)
fi
if [[ "$HOMOPHILY_FLAG" == "on" ]]; then
    CMD+=(--homophily)
fi
if [[ "$SURVEY_CONTEXT_FLAG" == "on" ]]; then
    CMD+=(--add_survey_to_context)
fi

if [[ -n "$LORAS_PATH" ]]; then
    read -r -a LORA_INDICES <<< "$LORA_INDEX_SET"
    PROPORTIONS=$(generate_random_proportions "${#LORA_INDICES[@]}")
    read -r -a PROPORTION_VALUES <<< "$PROPORTIONS"

    CMD+=(
        --loras_path "$LORAS_PATH"
        --lora_name_template "$LORA_NAME_TEMPLATE"
        --num_loras "$NUM_LORAS"
        --lora_indices "${LORA_INDICES[@]}"
        # --proportions "${PROPORTION_VALUES[@]}"
    )
fi

echo "========== Randomized simulation parameters =========="
echo "job_id=${SLURM_JOB_ID}"
echo "array_id=${SLURM_ARRAY_TASK_ID}"
echo "model_profile=${MODEL_PROFILE}"
echo "base_model=${BASE_MODEL}"
echo "loras_path=${LORAS_PATH:-none}"
echo "lora_name_template=${LORA_NAME_TEMPLATE:-none}"
echo "num_loras=${NUM_LORAS}"
echo "lora_indices=${LORA_INDEX_SET:-none}"
echo "proportions=${PROPORTIONS:-none}"
echo "graph_type=${GRAPH_TYPE}"
echo "homophily=${HOMOPHILY_FLAG}"
echo "add_survey_to_context=${SURVEY_CONTEXT_FLAG}"
echo "question_number=${QUESTION_NUMBER}"
echo "tweet_file=${TWEET_FILE}"
echo "num_agents=${NUM_AGENTS}"
echo "num_news_agents=${NUM_NEWS_AGENTS}"
echo "survey_output=${SURVEY_OUTPUT}"
echo "visualizer_output=${VISUALIZER_OUTPUT}"
echo "====================================================="

"${CMD[@]}"
