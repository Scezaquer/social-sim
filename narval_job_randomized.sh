#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --job-name=RandomizedSim
#SBATCH --array=0-19
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus-per-node=a100:1

# source ../concordia/ENV-concordia/bin/activate
# python src/main_unsloth.py --base_model "Qwen/Qwen2.5-7B-Instruct" --loras_path "$SCRATCH/Qwen" --lora_name_template "Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
# python src/main_unsloth.py --base_model "marcelbinz/Llama-3.1-Minitaur-8B" --loras_path "$SCRATCH/marcelbinz" --lora_name_template "Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
# python src/main_unsloth.py --base_model "meta-llama/Llama-3.1-8B" --loras_path "$SCRATCH/meta-llama" --lora_name_template "Llama-3.1-8B-lora-finetuned-unsloth-{i}"
# python src/main_unsloth.py --base_model "google/gemma-3-4b-pt" --loras_path "$SCRATCH/google" --lora_name_template "gemma-3-4b-pt-lora-finetuned-unsloth-{i}"

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
        if (v < 0 && v > -1e-9) {
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
NUM_AGENTS_CHOICES=(64 256 1024 4096)
NUM_NEWS_AGENTS_CHOICES=(0 1)
MODEL_PROFILE_CHOICES=("qwen_loras" "llama3.1_loras" "gemma_loras" "minitaure_loras")
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
LOCAL_LORAS_PATH=""
LORA_NAME_TEMPLATE=""
NUM_LORAS=0
LORA_INDEX_SET=""
AVERAGE_WEIGHTS_CSV=""
DISTRIBUTION_WEIGHTS_CSV=""
PROPORTIONS=""
declare -a PROPORTION_VALUES=()
declare -a LORA_INDICES=()

case "$MODEL_PROFILE" in
    minitaure_loras)
        BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
        LORAS_PATH="$SCRATCH/marcelbinz"
        LORA_NAME_TEMPLATE="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="minitaure_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="minitaure_optimized_convex_weights_cvxpy.csv"
        ;;
    qwen_loras)
        BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
        LORAS_PATH="$SCRATCH/Qwen"
        LORA_NAME_TEMPLATE="Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="qwen2.5-7Boptimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="qwen2.5-7B_optimized_convex_weights_cvxpy.csv"
        ;;
    llama3.1_loras)
        BASE_MODEL="meta-llama/Llama-3.1-8B"
        LORAS_PATH="$SCRATCH/meta-llama"
        LORA_NAME_TEMPLATE="Llama-3.1-8B-lora-finetuned-unsloth-{i}"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="llama3.1_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="llama3.1_optimized_convex_weights_cvxpy.csv"
        ;;
    gemma_loras)
        BASE_MODEL="google/gemma-3-4b-pt"
        LORAS_PATH="$SCRATCH/google"
        LORA_NAME_TEMPLATE="gemma-3-4b-pt-lora-finetuned-unsloth-{i}_token_prob_pop.pkl"
        NUM_LORAS=25
        AVERAGE_WEIGHTS_CSV="gemma_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="gemma_optimized_convex_weights_cvxpy.csv"
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
METRICS_OUTPUT="behavioral_metrics_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

LOCAL_SURVEY_OUTPUT="$LOCAL_OUTPUT_DIR/$SURVEY_OUTPUT"
LOCAL_VISUALIZER_OUTPUT="$LOCAL_OUTPUT_DIR/$VISUALIZER_OUTPUT"
LOCAL_METRICS_OUTPUT="$LOCAL_OUTPUT_DIR/$METRICS_OUTPUT"
THREADS_OUTPUT="$LOCAL_REPO/simulation_threads_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "Staging repository code and data into $LOCAL_REPO"
mkdir -p "$LOCAL_REPO/src"
rsync -a "$REPO_ROOT/src/" "$LOCAL_REPO/src/"
rsync -a "$REPO_ROOT/proportions/" "$LOCAL_REPO/proportions/"

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

    COMMON_BLUEPRINT_WEIGHTS=(
        80578 170583 2225632 107699 257398 406647 1014601 45071 73774
        4668716 116932 304804 104602 814539 70434 32923 149759 422736
        900057 43742 62087 739085 164314 212932 491830
    )

    WEIGHTS_DIR="$LOCAL_REPO/proportions"
    declare -a CSV_AVERAGE_WEIGHTS=()
    declare -a CSV_DISTRIBUTION_WEIGHTS=()
    load_weights_from_csv "$WEIGHTS_DIR/$AVERAGE_WEIGHTS_CSV" CSV_AVERAGE_WEIGHTS
    load_weights_from_csv "$WEIGHTS_DIR/$DISTRIBUTION_WEIGHTS_CSV" CSV_DISTRIBUTION_WEIGHTS

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

    LOCAL_LORAS_PATH="$LOCAL_ROOT/loras"
    mkdir -p "$LOCAL_LORAS_PATH"

    echo "Staging selected LoRAs into $LOCAL_LORAS_PATH"
    for idx in "${LORA_INDICES[@]}"; do
        lora_name="${LORA_NAME_TEMPLATE//\{i\}/$idx}"
        source_lora_path="$LORAS_PATH/$lora_name"
        target_lora_path="$LOCAL_LORAS_PATH/$lora_name"

        if [[ -d "$source_lora_path" ]]; then
            mkdir -p "$target_lora_path"
            rsync -a "$source_lora_path/" "$target_lora_path/"
        elif [[ -f "$source_lora_path" ]]; then
            mkdir -p "$(dirname "$target_lora_path")"
            cp -a "$source_lora_path" "$target_lora_path"
        else
            echo "Missing LoRA path: $source_lora_path" >&2
            exit 1
        fi
    done

    CMD+=(
        --loras_path "$LOCAL_LORAS_PATH"
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
echo "local_loras_path=${LOCAL_LORAS_PATH:-none}"
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
echo "local_repo=${LOCAL_REPO}"
echo "local_hf_cache=${LOCAL_HF_HUB_CACHE}"
echo "survey_output=${LOCAL_SURVEY_OUTPUT}"
echo "visualizer_output=${LOCAL_VISUALIZER_OUTPUT}"
echo "metrics_output=${LOCAL_METRICS_OUTPUT}"
echo "final_output_dir=${FINAL_OUTPUT_DIR}"
echo "====================================================="

cd "$LOCAL_REPO"
"${CMD[@]}"
