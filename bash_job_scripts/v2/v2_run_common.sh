#!/bin/bash
# Common V2 runner: executes one row of a design CSV produced by generate_design.py.
# Sourced by the per-experiment SBATCH scripts, which must set DESIGN_CSV before
# sourcing. The SLURM array task id selects the (0-based) row to run, so every
# run is fully determined by the committed design file.

set -euo pipefail

if [[ -z "${DESIGN_CSV:-}" ]]; then
    echo "DESIGN_CSV must be set before sourcing v2_run_common.sh" >&2
    exit 1
fi
if [[ ! -f "$DESIGN_CSV" ]]; then
    echo "Design file not found: $DESIGN_CSV" >&2
    exit 1
fi

module load python/3.10
source "$HOME/ENV/bin/activate"
export HF_HUB_CACHE="$SCRATCH/HF-cache"
export UNSLOTH_CACHE_DIR="$SLURM_TMPDIR/unsloth-cache"

WEIGHTS_DIR="./proportions"
ROW_INDEX="${SLURM_ARRAY_TASK_ID}"

ROW=$(awk -v n=$((ROW_INDEX + 2)) 'NR == n' "$DESIGN_CSV")
if [[ -z "$ROW" ]]; then
    echo "No design row at index $ROW_INDEX in $DESIGN_CSV" >&2
    exit 1
fi

IFS=, read -r RUN_ID EXPERIMENT SEED MODEL_PROFILE PROPORTIONS_OPTION QUESTION_NUMBER \
    TWEET_FILE NUM_AGENTS GRAPH_MODEL HOMOPHILY SURVEY_CTX NUM_NEWS_AGENTS \
    ACTIVITY_EXPONENT STIMULUS_MODE SURVEY_ORDER_MODE MAX_STEPS SURVEY_INTERVAL <<< "$ROW"

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
        if (v < 0) { v = 0 }
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

BASE_MODEL=""
LORAS_PATH=""
LORA_NAME_TEMPLATE=""
NUM_LORAS=25
AVERAGE_WEIGHTS_CSV=""
DISTRIBUTION_WEIGHTS_CSV=""
BASE_ONLY=0

case "$MODEL_PROFILE" in
    minitaur_loras|minitaur_base)
        BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
        LORAS_PATH="$SCRATCH/marcelbinz"
        LORA_NAME_TEMPLATE="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/minitaure_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/minitaure_optimized_convex_weights_cvxpy.csv"
        ;;
    llama3.1_loras|llama3.1_base)
        BASE_MODEL="meta-llama/Llama-3.1-8B"
        LORAS_PATH="$SCRATCH/meta-llama"
        LORA_NAME_TEMPLATE="Llama-3.1-8B-lora-finetuned-unsloth-{i}"
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/llama3.1_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/llama3.1_optimized_convex_weights_cvxpy.csv"
        ;;
    qwen_loras|qwen_base)
        BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
        LORAS_PATH="$SCRATCH/Qwen"
        LORA_NAME_TEMPLATE="Qwen2.5-7B-Instruct-lora-finetuned-{i}-no-focal"
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/qwen2.5-7Boptimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/qwen2.5-7B_optimized_convex_weights_cvxpy.csv"
        ;;
    gemma_loras|gemma_base)
        BASE_MODEL="google/gemma-3-4b-pt"
        LORAS_PATH="$SCRATCH/google"
        LORA_NAME_TEMPLATE="gemma-3-4b-pt-lora-finetuned-unsloth-{i}"
        AVERAGE_WEIGHTS_CSV="$WEIGHTS_DIR/gemma_optimized_convex_weights_hard.csv"
        DISTRIBUTION_WEIGHTS_CSV="$WEIGHTS_DIR/gemma_optimized_convex_weights_cvxpy.csv"
        ;;
    *)
        echo "Unknown model profile: $MODEL_PROFILE" >&2
        exit 1
        ;;
esac
if [[ "$MODEL_PROFILE" == *_base ]]; then
    BASE_ONLY=1
fi

SURVEY_OUTPUT="survey_v2_${EXPERIMENT}_${RUN_ID}.json"
VISUALIZER_OUTPUT="visualizer_v2_${EXPERIMENT}_${RUN_ID}.json"
METRICS_OUTPUT="behavioral_metrics_v2_${EXPERIMENT}_${RUN_ID}.json"

CMD=(
    python -u src/main.py
    --survey_output "$SURVEY_OUTPUT"
    --visualizer_output "$VISUALIZER_OUTPUT"
    --metrics_output "$METRICS_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --seed "$SEED"
    --question_number "$QUESTION_NUMBER"
    --tweet_files "news_tweets/$TWEET_FILE"
    --base_model "$BASE_MODEL"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --graph_model "$GRAPH_MODEL"
    --activity_exponent "$ACTIVITY_EXPONENT"
    --stimulus_mode "$STIMULUS_MODE"
    --survey_order_mode "$SURVEY_ORDER_MODE"
    --max_steps "$MAX_STEPS"
    --survey_interval "$SURVEY_INTERVAL"
)

if [[ "$HOMOPHILY" == "on" ]]; then
    CMD+=(--homophily)
fi
if [[ "$SURVEY_CTX" == "on" ]]; then
    CMD+=(--add_survey_to_context)
fi
if [[ "$STIMULUS_MODE" == "scrambled" ]]; then
    : "${SCRAMBLED_CORPUS:?SCRAMBLED_CORPUS must point to a threads JSON for scrambled runs}"
    CMD+=(--scrambled_corpus "$SCRAMBLED_CORPUS")
fi

if [[ "$BASE_ONLY" == "1" ]]; then
    CMD+=(--base_only)
else
    declare -a PROPORTION_VALUES=()
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
            for ((i = 0; i < NUM_LORAS; i++)); do PROPORTION_VALUES+=("1"); done
            ;;
        blueprint)
            PROPORTION_VALUES=("${COMMON_BLUEPRINT_WEIGHTS[@]}")
            ;;
        average)
            PROPORTION_VALUES=("${CSV_AVERAGE_WEIGHTS[@]}")
            ;;
        distribution)
            PROPORTION_VALUES=("${CSV_DISTRIBUTION_WEIGHTS[@]}")
            ;;
        *)
            echo "Unknown proportions option: $PROPORTIONS_OPTION" >&2
            exit 1
            ;;
    esac

    LORA_INDICES=()
    for ((i = 0; i < NUM_LORAS; i++)); do LORA_INDICES+=("$i"); done

    CMD+=(
        --loras_path "$LORAS_PATH"
        --lora_name_template "$LORA_NAME_TEMPLATE"
        --num_loras "$NUM_LORAS"
        --lora_indices "${LORA_INDICES[@]}"
        --proportions_option "$PROPORTIONS_OPTION"
        --proportions "${PROPORTION_VALUES[@]}"
    )
fi

echo "========== V2 design row =========="
echo "design_csv=${DESIGN_CSV} row_index=${ROW_INDEX}"
echo "$ROW"
echo "job_id=${SLURM_JOB_ID} base_model=${BASE_MODEL} base_only=${BASE_ONLY}"
echo "==================================="

"${CMD[@]}"
