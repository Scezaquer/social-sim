#!/bin/bash
#SBATCH --job-name=Comp550MinitaurLoras
#SBATCH --array=1-720
#SBATCH --time=3:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --partition=unkillable

set -euo pipefail

module load python/3.10
source "$HOME/ENV/bin/activate"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -f "$SLURM_SUBMIT_DIR/src/main.py" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
elif [[ -f "$PWD/src/main.py" ]]; then
    REPO_ROOT="$PWD"
else
    echo "Could not locate repository root with src/main.py" >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is not set." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > 720 )); then
    echo "SLURM_ARRAY_TASK_ID must be in [1, 720], got $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

SOURCE_HF_HOME="${HF_HOME:-}"
SOURCE_HF_HUB_CACHE="${HF_HUB_CACHE:-}"
SOURCE_HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

LOCAL_ROOT="${SLURM_TMPDIR:?}/social-sim-comp550-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
LOCAL_REPO="$LOCAL_ROOT/repo"
LOCAL_HF_HOME="$LOCAL_ROOT/HF-cache"
LOCAL_HF_HUB_CACHE="$LOCAL_HF_HOME/hub"
LOCAL_HF_DATASETS_CACHE="$LOCAL_HF_HOME/datasets"
LOCAL_UNSLOTH_CACHE_DIR="$LOCAL_ROOT/unsloth-cache"
LOCAL_OUTPUT_DIR="$LOCAL_ROOT/output"
FINAL_OUTPUT_DIR="${COMP550_OUTPUT_DIR:-$SCRATCH/social-sim-comp550}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$LOCAL_REPO" "$LOCAL_HF_HUB_CACHE" "$LOCAL_HF_DATASETS_CACHE" "$LOCAL_UNSLOTH_CACHE_DIR" "$LOCAL_OUTPUT_DIR" "$FINAL_OUTPUT_DIR"

export HF_HOME="$LOCAL_HF_HOME"
export HF_HUB_CACHE="$LOCAL_HF_HUB_CACHE"
export HF_DATASETS_CACHE="$LOCAL_HF_DATASETS_CACHE"
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

# Fixed run settings requested.
BASE_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"
LORAS_PATH="$SCRATCH/marcelbinz"
LORA_NAME_TEMPLATE="Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-{i}"
NUM_LORAS=25
LORA_INDEX_SET="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"
ADVERSARIAL_MODEL="Qwen/Qwen2.5-7B-Instruct"
GRAPH_MODEL="powerlaw_cluster"
NUM_NEWS_AGENTS="0"
SURVEY_CONTEXT_FLAG="off"
MODEL_PROFILE="minitaur_loras"

# Parameter grid requested.
QUESTIONS=(25 28 29)
PROPORTION_ADVERSARIES=(0 0.0625 0.125 0.25)
NUM_AGENTS_CHOICES=(64 128 256)
CENTRALIZE_CHOICES=(off on)
STRATEGIES=(false_information red_teaming)

COMBO_COUNT=$(( ${#QUESTIONS[@]} * ${#PROPORTION_ADVERSARIES[@]} * ${#NUM_AGENTS_CHOICES[@]} * ${#CENTRALIZE_CHOICES[@]} * ${#STRATEGIES[@]} ))
REPEATS_PER_COMBO=5
TOTAL_RUNS=$(( COMBO_COUNT * REPEATS_PER_COMBO ))

if (( TOTAL_RUNS != 720 )); then
    echo "Internal error: expected 720 runs, got $TOTAL_RUNS" >&2
    exit 1
fi

TASK_INDEX_ZERO_BASED=$(( SLURM_ARRAY_TASK_ID - 1 ))
COMBO_INDEX=$(( TASK_INDEX_ZERO_BASED % COMBO_COUNT ))
REPEAT_INDEX_ZERO_BASED=$(( TASK_INDEX_ZERO_BASED / COMBO_COUNT ))

tmp_idx=$COMBO_INDEX

strategy_idx=$(( tmp_idx % ${#STRATEGIES[@]} ))
tmp_idx=$(( tmp_idx / ${#STRATEGIES[@]} ))

centralize_idx=$(( tmp_idx % ${#CENTRALIZE_CHOICES[@]} ))
tmp_idx=$(( tmp_idx / ${#CENTRALIZE_CHOICES[@]} ))

num_agents_idx=$(( tmp_idx % ${#NUM_AGENTS_CHOICES[@]} ))
tmp_idx=$(( tmp_idx / ${#NUM_AGENTS_CHOICES[@]} ))

prop_adv_idx=$(( tmp_idx % ${#PROPORTION_ADVERSARIES[@]} ))
tmp_idx=$(( tmp_idx / ${#PROPORTION_ADVERSARIES[@]} ))

question_idx=$(( tmp_idx % ${#QUESTIONS[@]} ))

QUESTION_NUMBER="${QUESTIONS[$question_idx]}"
PROPORTION_ADVERSARIAL_AGENTS="${PROPORTION_ADVERSARIES[$prop_adv_idx]}"
NUM_AGENTS="${NUM_AGENTS_CHOICES[$num_agents_idx]}"
CENTRALIZE_FLAG="${CENTRALIZE_CHOICES[$centralize_idx]}"
ADVERSARIAL_STRATEGY="${STRATEGIES[$strategy_idx]}"
REPEAT_NUMBER=$(( REPEAT_INDEX_ZERO_BASED + 1 ))

read -r -a LORA_INDICES <<< "$LORA_INDEX_SET"

SURVEY_OUTPUT="survey_comp550_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
VISUALIZER_OUTPUT="visualizer_comp550_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
METRICS_OUTPUT="behavioral_metrics_comp550_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

LOCAL_SURVEY_OUTPUT="$LOCAL_OUTPUT_DIR/$SURVEY_OUTPUT"
LOCAL_VISUALIZER_OUTPUT="$LOCAL_OUTPUT_DIR/$VISUALIZER_OUTPUT"
LOCAL_METRICS_OUTPUT="$LOCAL_OUTPUT_DIR/$METRICS_OUTPUT"
THREADS_OUTPUT="$LOCAL_REPO/simulation_threads_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "Staging repository code and data into $LOCAL_REPO"
mkdir -p "$LOCAL_REPO/src"
rsync -a "$REPO_ROOT/src/" "$LOCAL_REPO/src/"

REQUIRED_FILES=(
    "divisive_questions_probabilities.json"
)
for file_name in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$REPO_ROOT/$file_name" ]]; then
        echo "Missing required file: $REPO_ROOT/$file_name" >&2
        exit 1
    fi
    cp -a "$REPO_ROOT/$file_name" "$LOCAL_REPO/$file_name"
done

SOURCE_CACHE_CANDIDATES=(
    "$SOURCE_HF_HUB_CACHE"
    "${SOURCE_HF_HOME:+$SOURCE_HF_HOME/hub}"
    "$SOURCE_HF_HOME"
    "$SCRATCH/HF-cache/hub"
    "$SCRATCH/HF-cache"
    "$HOME/.cache/huggingface/hub"
)

stage_model_snapshot() {
    local model_id="$1"
    local model_cache_dir="models--${model_id//\//--}"
    local source_model_cache_dir=""

    for candidate in "${SOURCE_CACHE_CANDIDATES[@]}"; do
        [[ -z "$candidate" ]] && continue
        if [[ -d "$candidate/$model_cache_dir" ]]; then
            source_model_cache_dir="$candidate/$model_cache_dir"
            break
        fi
    done

    if [[ -z "$source_model_cache_dir" ]]; then
        echo "Could not find cached model $model_cache_dir in any source HF cache root." >&2
        echo "Checked candidates:" >&2
        for candidate in "${SOURCE_CACHE_CANDIDATES[@]}"; do
            [[ -n "$candidate" ]] && echo "  - $candidate" >&2
        done
        exit 1
    fi

    local local_model_cache_dir="$LOCAL_HF_HUB_CACHE/$model_cache_dir"
    local resolved_model_path=""

    echo "Staging model cache for $model_id into $local_model_cache_dir"
    mkdir -p "$local_model_cache_dir"
    rsync -a "$source_model_cache_dir/" "$local_model_cache_dir/"

    if [[ -f "$local_model_cache_dir/refs/main" ]]; then
        local snapshot_revision
        snapshot_revision=$(<"$local_model_cache_dir/refs/main")
        snapshot_revision="${snapshot_revision//$'\r'/}"
        snapshot_revision="${snapshot_revision//$'\n'/}"
        resolved_model_path="$local_model_cache_dir/snapshots/$snapshot_revision"
    fi

    if [[ -z "$resolved_model_path" ]]; then
        local snapshot_candidates=("$local_model_cache_dir"/snapshots/*)
        if [[ -e "${snapshot_candidates[0]}" ]]; then
            resolved_model_path="${snapshot_candidates[0]}"
        fi
    fi

    if [[ -z "$resolved_model_path" ]] || [[ ! -d "$resolved_model_path" ]]; then
        echo "Could not resolve local snapshot directory for $model_id under $local_model_cache_dir" >&2
        exit 1
    fi

    if [[ ! -f "$resolved_model_path/config.json" ]]; then
        echo "Resolved model path does not contain config.json: $resolved_model_path" >&2
        exit 1
    fi

    printf "%s\n" "$resolved_model_path"
}

RESOLVED_BASE_MODEL=$(stage_model_snapshot "$BASE_MODEL")
RESOLVED_ADVERSARIAL_MODEL=$(stage_model_snapshot "$ADVERSARIAL_MODEL")

PERSONAS_DATASET_CACHE_DIR="Tianyi-Lab___personas"
SOURCE_PERSONAS_DATASET_DIR=""
SOURCE_DATASETS_CACHE_CANDIDATES=(
    "$SOURCE_HF_DATASETS_CACHE"
    "${SOURCE_HF_HOME:+$SOURCE_HF_HOME/datasets}"
    "$SCRATCH/HF-cache/datasets"
    "$HOME/.cache/huggingface/datasets"
)
for candidate in "${SOURCE_DATASETS_CACHE_CANDIDATES[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ -d "$candidate/$PERSONAS_DATASET_CACHE_DIR" ]]; then
        SOURCE_PERSONAS_DATASET_DIR="$candidate/$PERSONAS_DATASET_CACHE_DIR"
        break
    fi
done

if [[ -z "$SOURCE_PERSONAS_DATASET_DIR" ]]; then
    echo "Could not find cached dataset directory $PERSONAS_DATASET_CACHE_DIR in any source datasets cache root." >&2
    echo "Checked candidates:" >&2
    for candidate in "${SOURCE_DATASETS_CACHE_CANDIDATES[@]}"; do
        [[ -n "$candidate" ]] && echo "  - $candidate" >&2
    done
    exit 1
fi

LOCAL_PERSONAS_DATASET_DIR="$LOCAL_HF_DATASETS_CACHE/$PERSONAS_DATASET_CACHE_DIR"
echo "Staging dataset cache for Tianyi-Lab/Personas into $LOCAL_PERSONAS_DATASET_DIR"
mkdir -p "$LOCAL_PERSONAS_DATASET_DIR"
rsync -a "$SOURCE_PERSONAS_DATASET_DIR/" "$LOCAL_PERSONAS_DATASET_DIR/"

CMD=(
    python -u "$LOCAL_REPO/src/main.py"
    --survey_output "$LOCAL_SURVEY_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --question_number "$QUESTION_NUMBER"
    --base_model "$RESOLVED_BASE_MODEL"
    --loras_path "$LORAS_PATH"
    --lora_name_template "$LORA_NAME_TEMPLATE"
    --num_loras "$NUM_LORAS"
    --lora_indices "${LORA_INDICES[@]}"
    --proportions_option "uniform"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --proportion_adversarial_agents "$PROPORTION_ADVERSARIAL_AGENTS"
    --adversarial_strategy "$ADVERSARIAL_STRATEGY"
    --adversarial_model "$RESOLVED_ADVERSARIAL_MODEL"
    --graph_model "$GRAPH_MODEL"
    --visualizer_output "$LOCAL_VISUALIZER_OUTPUT"
    --metrics_output "$LOCAL_METRICS_OUTPUT"
)

if [[ "$CENTRALIZE_FLAG" == "on" ]]; then
    CMD+=(--centralize_adversaries)
fi

echo "========== COMP550 simulation parameters =========="
echo "job_id=${SLURM_JOB_ID}"
echo "array_id=${SLURM_ARRAY_TASK_ID}"
echo "task_index_zero_based=${TASK_INDEX_ZERO_BASED}"
echo "combo_count=${COMBO_COUNT}"
echo "combo_index=${COMBO_INDEX}"
echo "repeat_number=${REPEAT_NUMBER}"
echo "model_profile=${MODEL_PROFILE}"
echo "base_model=${BASE_MODEL}"
echo "resolved_base_model=${RESOLVED_BASE_MODEL}"
echo "loras_path=${LORAS_PATH}"
echo "lora_name_template=${LORA_NAME_TEMPLATE}"
echo "num_loras=${NUM_LORAS}"
echo "lora_indices=${LORA_INDEX_SET}"
echo "proportions_option=uniform"
echo "adversarial_model=${ADVERSARIAL_MODEL}"
echo "resolved_adversarial_model=${RESOLVED_ADVERSARIAL_MODEL}"
echo "graph_model=${GRAPH_MODEL}"
echo "homophily=off"
echo "add_survey_to_context=${SURVEY_CONTEXT_FLAG}"
echo "question_number=${QUESTION_NUMBER}"
echo "proportion_adversarial_agents=${PROPORTION_ADVERSARIAL_AGENTS}"
echo "num_agents=${NUM_AGENTS}"
echo "num_news_agents=${NUM_NEWS_AGENTS}"
echo "centralize_adversaries=${CENTRALIZE_FLAG}"
echo "adversarial_strategy=${ADVERSARIAL_STRATEGY}"
echo "local_repo=${LOCAL_REPO}"
echo "local_hf_cache=${LOCAL_HF_HUB_CACHE}"
echo "local_hf_datasets_cache=${LOCAL_HF_DATASETS_CACHE}"
echo "source_personas_dataset_dir=${SOURCE_PERSONAS_DATASET_DIR}"
echo "survey_output=${LOCAL_SURVEY_OUTPUT}"
echo "visualizer_output=${LOCAL_VISUALIZER_OUTPUT}"
echo "metrics_output=${LOCAL_METRICS_OUTPUT}"
echo "final_output_dir=${FINAL_OUTPUT_DIR}"
echo "=================================================="

cd "$LOCAL_REPO"
"${CMD[@]}"
