#!/bin/bash
#SBATCH --job-name=NoNews
#SBATCH --array=1-70
#SBATCH --time=6:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1


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

sanitize_weight() {
    local raw_weight=$1
    awk -v w="$raw_weight" 'BEGIN {
        v = w + 0
        if (v < 0) {
            v = 0
        }
        if (v < 0) {
            exit 2
        }
        printf "%.17g", v
    }'
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
        local sanitized_weight
        idx=$(extract_lora_id "$model_file")
        if ! sanitized_weight=$(sanitize_weight "$weight"); then
            echo "Invalid negative weight for LoRA id $idx in $csv_path: $weight" >&2
            exit 1
        fi
        indexed_weights[$idx]="$sanitized_weight"
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
    "25:news_tweets/genetic_enhancements_tweets.json"
    "28:news_tweets/ai_copyright_tweets.json"
    "29:news_tweets/environmental_protection_tweets.json"
)
GRAPH_CHOICES=(
    "random"
    "powerlaw_cluster"
    "barabasi_albert"
    "stochastic_block"
    "forest_fire"
    "fully_connected"
    "cycle"
)
HOMOPHILY_CHOICES=("on" "off")
SURVEY_CONTEXT_CHOICES=("on" "off")
NUM_AGENTS_CHOICES=(64 256 1024 4096)
NUM_NEWS_AGENTS_CHOICES=(0 1)
MODEL_PROFILE_CHOICES=("qwen_loras" "gemma_loras" "llama3.1_loras" "minitaure_loras")
PROPORTIONS_OPTION_CHOICES=("average" "distribution" "uniform" "blueprint" )

QUESTION_PICK="28:news_tweets/ai_copyright_tweets.json"
QUESTION_NUMBER="${QUESTION_PICK%%:*}"
TWEET_FILE="${QUESTION_PICK#*:}"

# Deterministic mapping: graphs(7) x homophily(2) x runs(5) = 70
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set. Run as job array (1-70)." >&2
    exit 1
fi

idx=$((SLURM_ARRAY_TASK_ID - 1))
runs_per=5
num_homophily=${#HOMOPHILY_CHOICES[@]}
num_graphs=${#GRAPH_CHOICES[@]}

run_index=$(( idx % runs_per ))
idx=$(( idx / runs_per ))

homophily_index=$(( idx % num_homophily ))
idx=$(( idx / num_homophily ))

graph_index=$(( idx % num_graphs ))

GRAPH_TYPE="${GRAPH_CHOICES[$graph_index]}"
HOMOPHILY_FLAG="${HOMOPHILY_CHOICES[$homophily_index]}"
SURVEY_CONTEXT_FLAG="off"
NUM_AGENTS=256
NUM_NEWS_AGENTS=0
MODEL_PROFILE="minitaure_loras"
PROPORTIONS_OPTION="uniform"

# Expose run number
RUN_NUMBER=$((run_index + 1))

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
    qwen_base)
        BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
        PROPORTIONS_OPTION=""
        ;;
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
        LORA_NAME_TEMPLATE="gemma-3-4b-pt-lora-finetuned-unsloth-{i}"
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
LORA_INDEX_SET="${MODEL_INDEX_SETS[0]}"

SURVEY_OUTPUT="survey_randomized_emnlp2026nonews_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
VISUALIZER_OUTPUT="visualizer_randomized_emnlp2026nonews_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

CMD=(
    python -u src/main.py
    --survey_output "$SURVEY_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --question_number "$QUESTION_NUMBER"
    --tweet_files "$TWEET_FILE"
    --base_model "$BASE_MODEL"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --visualizer_output "$VISUALIZER_OUTPUT"
    --graph_model "$GRAPH_TYPE"
)

if [[ "$MODEL_PROFILE" == "qwen_base" ]]; then
    CMD+=(--base_only)
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
