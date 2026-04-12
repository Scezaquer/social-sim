#!/bin/bash
#SBATCH --account=ctb-liyue
#SBATCH --job-name=RandomizedNetworkSim
#SBATCH --array=0-75
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus-per-node=a100:1

set -euo pipefail

pick_random() {
    local -n arr_ref=$1
    echo "${arr_ref[$RANDOM % ${#arr_ref[@]}]}"
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

LOCAL_ROOT="${SLURM_TMPDIR:?}/social-sim-network-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
LOCAL_REPO="$LOCAL_ROOT/repo"
LOCAL_HF_HOME="$LOCAL_ROOT/HF-cache"
LOCAL_HF_HUB_CACHE="$LOCAL_HF_HOME/hub"
LOCAL_UNSLOTH_CACHE_DIR="$LOCAL_ROOT/unsloth-cache"
LOCAL_OUTPUT_DIR="$LOCAL_ROOT/output"
FINAL_OUTPUT_DIR="${RANDOMIZED_NETWORK_OUTPUT_DIR:-$SCRATCH/social-sim-randomized-network}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_REPO" "$LOCAL_HF_HUB_CACHE" "$LOCAL_UNSLOTH_CACHE_DIR" "$LOCAL_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

export HF_HOME="$LOCAL_HF_HOME"
export HF_HUB_CACHE="$LOCAL_HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$LOCAL_HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
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

# Deterministic run parameters requested.
QUESTION_NUMBER="28"
TWEET_FILE="ai_copyright_tweets.json"
NUM_AGENTS="256"
NUM_NEWS_AGENTS="1"
MODEL_PROFILE="minitaure_loras"
PROPORTIONS_OPTION="uniform"
SURVEY_CONTEXT_FLAG="off"

# Randomized dimensions requested.
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
GRAPH_MODEL=$(pick_random GRAPH_CHOICES)
HOMOPHILY_FLAG=$(pick_random HOMOPHILY_CHOICES)

BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
LORAS_PATH="$SCRATCH/marcelbinz"
LORA_NAME_TEMPLATE="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
NUM_LORAS=25
LORA_INDEX_SET="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
AVERAGE_WEIGHTS_CSV="minitaure_optimized_convex_weights_hard.csv"
DISTRIBUTION_WEIGHTS_CSV="minitaure_optimized_convex_weights_cvxpy.csv"

SURVEY_OUTPUT="survey_randomized_network_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
VISUALIZER_OUTPUT="visualizer_randomized_network_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
METRICS_OUTPUT="behavioral_metrics_randomized_network_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

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
    "ai_copyright_tweets.json"
)
for file_name in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$REPO_ROOT/$file_name" ]]; then
        echo "Missing required file: $REPO_ROOT/$file_name" >&2
        exit 1
    fi
    cp -a "$REPO_ROOT/$file_name" "$LOCAL_REPO/$file_name"
done

MODEL_CACHE_DIR="models--${BASE_MODEL//\//--}"
SOURCE_HF_HUB_CACHE=""
SOURCE_MODEL_CACHE_DIR=""
RESOLVED_BASE_MODEL=""

SOURCE_CACHE_CANDIDATES=(
    "${SOURCE_HF_HUB_CACHE:-}"
    "${SOURCE_HF_HOME:-}/hub"
    "${SOURCE_HF_HOME:-}"
    "$SCRATCH/HF-cache/hub"
    "$SCRATCH/HF-cache"
)
for candidate in "${SOURCE_CACHE_CANDIDATES[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -d "$candidate/$MODEL_CACHE_DIR" ]]; then
        SOURCE_HF_HUB_CACHE="$candidate"
        SOURCE_MODEL_CACHE_DIR="$candidate/$MODEL_CACHE_DIR"
        break
    fi
done

if [[ -z "$SOURCE_MODEL_CACHE_DIR" ]]; then
    echo "Could not find cached model $MODEL_CACHE_DIR in any source HF cache root." >&2
    echo "Checked candidates:" >&2
    for candidate in "${SOURCE_CACHE_CANDIDATES[@]}"; do
        [[ -n "$candidate" ]] && echo "  - $candidate" >&2
    done
    exit 1
fi

LOCAL_MODEL_CACHE_DIR="$LOCAL_HF_HUB_CACHE/$MODEL_CACHE_DIR"

echo "Staging model cache for $BASE_MODEL into $LOCAL_MODEL_CACHE_DIR"
mkdir -p "$LOCAL_MODEL_CACHE_DIR"
rsync -a "$SOURCE_MODEL_CACHE_DIR/" "$LOCAL_MODEL_CACHE_DIR/"

if [[ -f "$LOCAL_MODEL_CACHE_DIR/refs/main" ]]; then
    SNAPSHOT_REVISION=$(<"$LOCAL_MODEL_CACHE_DIR/refs/main")
    SNAPSHOT_REVISION="${SNAPSHOT_REVISION//$'\r'/}"
    SNAPSHOT_REVISION="${SNAPSHOT_REVISION//$'\n'/}"
    RESOLVED_BASE_MODEL="$LOCAL_MODEL_CACHE_DIR/snapshots/$SNAPSHOT_REVISION"
fi

if [[ -z "$RESOLVED_BASE_MODEL" ]]; then
    snapshot_candidates=("$LOCAL_MODEL_CACHE_DIR"/snapshots/*)
    if [[ -e "${snapshot_candidates[0]}" ]]; then
        RESOLVED_BASE_MODEL="${snapshot_candidates[0]}"
    fi
fi

if [[ -z "$RESOLVED_BASE_MODEL" ]] || [[ ! -d "$RESOLVED_BASE_MODEL" ]]; then
    echo "Could not resolve local snapshot directory for $BASE_MODEL under $LOCAL_MODEL_CACHE_DIR" >&2
    exit 1
fi

if [[ ! -f "$RESOLVED_BASE_MODEL/config.json" ]]; then
    echo "Resolved model path does not contain config.json: $RESOLVED_BASE_MODEL" >&2
    exit 1
fi

read -r -a LORA_INDICES <<< "$LORA_INDEX_SET"
declare -a PROPORTION_VALUES=()
for _ in "${LORA_INDICES[@]}"; do
    PROPORTION_VALUES+=("1")
done

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

CMD=(
    python -u "$LOCAL_REPO/src/main_unsloth.py"
    --survey_output "$LOCAL_SURVEY_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --question_number "$QUESTION_NUMBER"
    --tweet_files "$TWEET_FILE"
    --base_model "$RESOLVED_BASE_MODEL"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --visualizer_output "$LOCAL_VISUALIZER_OUTPUT"
    --metrics_output "$LOCAL_METRICS_OUTPUT"
    --graph_model "$GRAPH_MODEL"
    --loras_path "$LOCAL_LORAS_PATH"
    --lora_name_template "$LORA_NAME_TEMPLATE"
    --num_loras "$NUM_LORAS"
    --lora_indices "${LORA_INDICES[@]}"
    --proportions_option "$PROPORTIONS_OPTION"
    --proportions "${PROPORTION_VALUES[@]}"
)

if [[ "$HOMOPHILY_FLAG" == "on" ]]; then
    CMD+=(--homophily)
fi

echo "========== Randomized network simulation parameters =========="
echo "job_id=${SLURM_JOB_ID}"
echo "array_id=${SLURM_ARRAY_TASK_ID}"
echo "model_profile=${MODEL_PROFILE}"
echo "base_model=${BASE_MODEL}"
echo "resolved_base_model=${RESOLVED_BASE_MODEL}"
echo "loras_path=${LORAS_PATH}"
echo "local_loras_path=${LOCAL_LORAS_PATH}"
echo "lora_name_template=${LORA_NAME_TEMPLATE}"
echo "num_loras=${NUM_LORAS}"
echo "lora_indices=${LORA_INDEX_SET}"
echo "proportions_option=${PROPORTIONS_OPTION}"
echo "graph_model=${GRAPH_MODEL}"
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
echo "=============================================================="

cd "$LOCAL_REPO"
"${CMD[@]}"
