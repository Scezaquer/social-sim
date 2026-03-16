#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --job-name=RandomizedSim
#SBATCH --array=0-19
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-node=a100:1

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

module load python/3.11
module load scipy-stack
source "${NARVAL_VENV_PATH:-../concordia/ENV-concordia/bin/activate}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "$SLURM_SUBMIT_DIR/src/main_unsloth.py" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -f "$PWD/src/main_unsloth.py" ]]; then
    REPO_ROOT="$PWD"
else
    echo "Could not locate repository root with src/main_unsloth.py" >&2
    exit 1
fi

LOCAL_ROOT="${SLURM_TMPDIR:?}/social-sim-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
LOCAL_REPO="$LOCAL_ROOT/repo"
LOCAL_HF_HOME="$LOCAL_ROOT/HF-cache"
LOCAL_HF_HUB_CACHE="$LOCAL_HF_HOME/hub"
LOCAL_UNSLOTH_CACHE_DIR="$LOCAL_ROOT/unsloth-cache"
LOCAL_OUTPUT_DIR="$LOCAL_ROOT/output"
FINAL_OUTPUT_DIR="${RANDOMIZED_OUTPUT_DIR:-$SCRATCH/social-sim-randomized}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_REPO" "$LOCAL_HF_HUB_CACHE" "$LOCAL_UNSLOTH_CACHE_DIR" "$LOCAL_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

export HF_HOME="$LOCAL_HF_HOME"
export HF_HUB_CACHE="$LOCAL_HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$LOCAL_HF_HUB_CACHE"
export UNSLOTH_CACHE_DIR="$LOCAL_UNSLOTH_CACHE_DIR"
export TMPDIR="$SLURM_TMPDIR"
export TOKENIZERS_PARALLELISM=false

cleanup() {
    local exit_code="$1"
    mkdir -p "$FINAL_OUTPUT_DIR"
    if [[ -d "$LOCAL_OUTPUT_DIR" ]]; then
        cp -a "$LOCAL_OUTPUT_DIR/." "$FINAL_OUTPUT_DIR/" || true
    fi
    if [[ -n "${THREADS_OUTPUT:-}" ]] && [[ -f "$THREADS_OUTPUT" ]]; then
        cp -a "$THREADS_OUTPUT" "$FINAL_OUTPUT_DIR/" || true
    fi
    return "$exit_code"
}

trap 'cleanup "$?"' EXIT

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
MODEL_PROFILE_CHOICES=("minitaur_loras")

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
LOCAL_LORAS_PATH=""
LORA_NAME_TEMPLATE=""
NUM_LORAS=0
LORA_INDEX_SET=""
PROPORTIONS=""
declare -a LORA_INDICES=()

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
METRICS_OUTPUT="behavioral_metrics_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

LOCAL_SURVEY_OUTPUT="$LOCAL_OUTPUT_DIR/$SURVEY_OUTPUT"
LOCAL_VISUALIZER_OUTPUT="$LOCAL_OUTPUT_DIR/$VISUALIZER_OUTPUT"
LOCAL_METRICS_OUTPUT="$LOCAL_OUTPUT_DIR/$METRICS_OUTPUT"
THREADS_OUTPUT="$LOCAL_REPO/simulation_threads_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "Staging repository code and data into $LOCAL_REPO"
mkdir -p "$LOCAL_REPO/src"
rsync -a "$REPO_ROOT/src/" "$LOCAL_REPO/src/"

REQUIRED_FILES=(
    "divisive_questions_probabilities.json"
    "genetic_enhancements_tweets.json"
    "ai_copyright_tweets.json"
    "environmental_protection_tweets.json"
)
for file_name in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$REPO_ROOT/$file_name" ]]; then
        echo "Missing required file: $REPO_ROOT/$file_name" >&2
        exit 1
    fi
    cp -a "$REPO_ROOT/$file_name" "$LOCAL_REPO/$file_name"
done

SOURCE_HF_HUB_CACHE="${SOURCE_HF_HUB_CACHE:-$SCRATCH/HF-cache/hub}"
MODEL_CACHE_DIR="models--${BASE_MODEL//\//--}"
SOURCE_MODEL_CACHE_DIR="$SOURCE_HF_HUB_CACHE/$MODEL_CACHE_DIR"
LOCAL_MODEL_CACHE_DIR="$LOCAL_HF_HUB_CACHE/$MODEL_CACHE_DIR"

echo "Staging model cache for $BASE_MODEL into $LOCAL_MODEL_CACHE_DIR"
if [[ -d "$SOURCE_MODEL_CACHE_DIR" ]]; then
    mkdir -p "$LOCAL_MODEL_CACHE_DIR"
    rsync -a "$SOURCE_MODEL_CACHE_DIR/" "$LOCAL_MODEL_CACHE_DIR/"
else
    echo "Expected model cache not found at $SOURCE_MODEL_CACHE_DIR" >&2
    exit 1
fi

CMD=(
    python -u "$LOCAL_REPO/src/main_unsloth.py"
    --survey_output "$LOCAL_SURVEY_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --question_number "$QUESTION_NUMBER"
    --tweet_files "$TWEET_FILE"
    --base_model "$BASE_MODEL"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --visualizer_output "$LOCAL_VISUALIZER_OUTPUT"
    --metrics_output "$LOCAL_METRICS_OUTPUT"
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

    LOCAL_LORAS_PATH="$LOCAL_ROOT/loras"
    mkdir -p "$LOCAL_LORAS_PATH"

    echo "Staging selected LoRAs into $LOCAL_LORAS_PATH"
    for idx in "${LORA_INDICES[@]}"; do
        lora_dir_name="${LORA_NAME_TEMPLATE//\{i\}/$idx}"
        source_lora_dir="$LORAS_PATH/$lora_dir_name"
        target_lora_dir="$LOCAL_LORAS_PATH/$lora_dir_name"
        if [[ ! -d "$source_lora_dir" ]]; then
            echo "Missing LoRA directory: $source_lora_dir" >&2
            exit 1
        fi
        mkdir -p "$target_lora_dir"
        rsync -a "$source_lora_dir/" "$target_lora_dir/"
    done

    CMD+=(
        --loras_path "$LOCAL_LORAS_PATH"
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
echo "local_loras_path=${LOCAL_LORAS_PATH:-none}"
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
echo "local_repo=${LOCAL_REPO}"
echo "local_hf_cache=${LOCAL_HF_HUB_CACHE}"
echo "survey_output=${LOCAL_SURVEY_OUTPUT}"
echo "visualizer_output=${LOCAL_VISUALIZER_OUTPUT}"
echo "metrics_output=${LOCAL_METRICS_OUTPUT}"
echo "final_output_dir=${FINAL_OUTPUT_DIR}"
echo "====================================================="

cd "$LOCAL_REPO"
"${CMD[@]}"
