#!/bin/bash
#SBATCH --job-name=RandomizedSim
#SBATCH --array=0-50
#SBATCH --time=6:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=main


set -euo pipefail

pick_random() {
    local -n arr_ref=$1
    echo "${arr_ref[$RANDOM % ${#arr_ref[@]}]}"
}

extract_lora_id() {
    local model_file=$1

    if [[ "$model_file" =~ -([0-9]+)-no-focal_token_prob_pop\.pkl$ ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$model_file" =~ -([0-9]+)_token_prob_pop\.pkl$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "Could not extract LoRA id from model file name: $model_file" >&2
        exit 1
    fi
}

load_weights_from_csv() {
    local csv_path=$1
    local -n out_weights_ref=$2
    local -a indexed_weights=()

    if [[ ! -f "$csv_path" ]]; then
        echo "Weights file not found: $csv_path" >&2
        exit 1
    fi

    while IFS=, read -r model_file weight; do
        [[ "$model_file" == "Model_File" ]] && continue
        [[ -z "$model_file" ]] && continue
        local idx
        idx=$(extract_lora_id "$model_file")
        indexed_weights[$idx]="$weight"
    done < "$csv_path"

    out_weights_ref=()
    for ((i = 0; i < NUM_LORAS; i++)); do
        if [[ -z "${indexed_weights[$i]:-}" ]]; then
            echo "Missing weight for LoRA id $i in $csv_path" >&2
            exit 1
        fi
        out_weights_ref+=("${indexed_weights[$i]}")
    done
}

module load python/3.10
source "$HOME/ENV/bin/activate"
export HF_HUB_CACHE="$SCRATCH/HF-cache"
export UNSLOTH_CACHE_DIR="$SLURM_TMPDIR/unsloth-cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="./proportions"

QUESTION_CHOICES=(
    "25:genetic_enhancements_tweets.json"
    "28:ai_copyright_tweets.json"
    "29:environmental_protection_tweets.json"
)
GRAPH_CHOICES=("random" "powerlaw")
HOMOPHILY_CHOICES=("on" "off")
SURVEY_CONTEXT_CHOICES=("on" "off")
NUM_AGENTS_CHOICES=(64 256 1024 4096)
NUM_NEWS_AGENTS_CHOICES=(0 1)
MODEL_PROFILE_CHOICES=("qwen_loras" "gemma_loras") # "minitaure_loras")
PROPORTIONS_OPTION_CHOICES=("uniform" "blueprint" "average" "distribution")

QUESTION_PICK=$(pick_random QUESTION_CHOICES)
QUESTION_NUMBER="${QUESTION_PICK%%:*}"
TWEET_FILE="${QUESTION_PICK#*:}"

GRAPH_TYPE=$(pick_random GRAPH_CHOICES)
HOMOPHILY_FLAG=$(pick_random HOMOPHILY_CHOICES)
SURVEY_CONTEXT_FLAG=$(pick_random SURVEY_CONTEXT_CHOICES)
NUM_AGENTS=$(pick_random NUM_AGENTS_CHOICES)
NUM_NEWS_AGENTS=$(pick_random NUM_NEWS_AGENTS_CHOICES)
MODEL_PROFILE=$(pick_random MODEL_PROFILE_CHOICES)
PROPORTIONS_OPTION=$(pick_random PROPORTIONS_OPTION_CHOICES)

BASE_MODEL=""
LORAS_PATH=""
LORA_NAME_TEMPLATE=""
NUM_LORAS=0
LORA_INDEX_SET=""
AVERAGE_WEIGHTS_CSV=""
DISTRIBUTION_WEIGHTS_CSV=""
PROPORTIONS=""
declare -a PROPORTION_VALUES=()

case "$MODEL_PROFILE" in
    minitaure_loras)
        BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
        LORAS_PATH="$SCRATCH/marcelbinz"
        LORA_NAME_TEMPLATE="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/minitaure_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/minitaure_optimized_convex_weights_cvxpy.csv"
        ;;
    qwen_loras)
        BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
        LORAS_PATH="$SCRATCH/Qwen"
        LORA_NAME_TEMPLATE="Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/qwen2.5-7Boptimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/qwen2.5-7B_optimized_convex_weights_cvxpy.csv"
        ;;
    llama3.1_loras)
        BASE_MODEL="meta-llama/Llama-3.1-8B"
        LORAS_PATH="$SCRATCH/meta-llama"
        LORA_NAME_TEMPLATE="Llama-3.1-8B-lora-finetuned-unsloth-{i}"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/llama3.1_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/llama3.1_optimized_convex_weights_cvxpy.csv"
        ;;
    gemma_loras)
        BASE_MODEL="google/gemma-3-4b-pt"
        LORAS_PATH="$SCRATCH/google"
        LORA_NAME_TEMPLATE="gemma-3-4b-pt-lora-finetuned-unsloth-{i}_token_prob_pop.pkl"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/gemma_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/gemma_optimized_convex_weights_cvxpy.csv"
        ;;
    *)
        echo "Unknown model profile: $MODEL_PROFILE" >&2
        exit 1
        ;;
esac

MODEL_INDEX_SETS=(
    "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
)
LORA_INDEX_SET=$(pick_random MODEL_INDEX_SETS)

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

    COMMON_BLUEPRINT_WEIGHTS=(
        80578 170583 2225632 107699 257398 406647 1014601 45071 73774
        4668716 116932 304804 104602 814539 70434 32923 149759 422736
        900057 43742 62087 739085 164314 212932 491830
    )

    declare -a CSV_AVERAGE_WEIGHTS=()
    declare -a CSV_DISTRIBUTION_WEIGHTS=()
    load_weights_from_csv "$AVERAGE_WEIGHTS_CSV" CSV_AVERAGE_WEIGHTS
    load_weights_from_csv "$DISTRIBUTION_WEIGHTS_CSV" CSV_DISTRIBUTION_WEIGHTS

    case "$PROPORTIONS_OPTION" in
        uniform)
            for _ in "${LORA_INDICES[@]}"; do
                PROPORTION_VALUES+=("1")
            done
            ;;
        blueprint)
            for idx in "${LORA_INDICES[@]}"; do
                PROPORTION_VALUES+=("${COMMON_BLUEPRINT_WEIGHTS[$idx]}")
            done
            ;;
        average)
            for idx in "${LORA_INDICES[@]}"; do
                PROPORTION_VALUES+=("${CSV_AVERAGE_WEIGHTS[$idx]}")
            done
            ;;
        distribution)
            for idx in "${LORA_INDICES[@]}"; do
                PROPORTION_VALUES+=("${CSV_DISTRIBUTION_WEIGHTS[$idx]}")
            done
            ;;
        *)
            echo "Unknown proportions option: $PROPORTIONS_OPTION" >&2
            exit 1
            ;;
    esac

    PROPORTIONS="${PROPORTION_VALUES[*]}"

    CMD+=(
        --loras_path "$LORAS_PATH"
        --lora_name_template "$LORA_NAME_TEMPLATE"
        --num_loras "$NUM_LORAS"
        --lora_indices "${LORA_INDICES[@]}"
        --proportions_option "$PROPORTIONS_OPTION"
        --proportions "${PROPORTION_VALUES[@]}"
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
echo "proportions_option=${PROPORTIONS_OPTION:-none}"
echo "proportions=${PROPORTIONS:-none}"
echo "average_weights_csv=${AVERAGE_WEIGHTS_CSV:-none}"
echo "distribution_weights_csv=${DISTRIBUTION_WEIGHTS_CSV:-none}"
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
