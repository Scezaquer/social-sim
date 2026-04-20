#!/bin/bash
#SBATCH --job-name=Comp550MinitaurLoras
#SBATCH --array=1-720
#SBATCH --time=3:00:00
#SBATCH --mem=24Gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

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

export HF_HUB_CACHE=$SCRATCH/HF-cache
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HUB_CACHE/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export UNSLOTH_CACHE_DIR="${UNSLOTH_CACHE_DIR:-$SLURM_TMPDIR/unsloth-cache}"

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

THREADS_OUTPUT="$REPO_ROOT/simulation_threads_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

REQUIRED_FILES=(
    "divisive_questions_probabilities.json"
)
for file_name in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$REPO_ROOT/$file_name" ]]; then
        echo "Missing required file: $REPO_ROOT/$file_name" >&2
        exit 1
    fi
done

CMD=(
    python -u "$REPO_ROOT/src/main.py"
    --survey_output "$SURVEY_OUTPUT"
    --array_id "${SLURM_ARRAY_TASK_ID}"
    --job_id "${SLURM_JOB_ID}"
    --question_number "$QUESTION_NUMBER"
    --base_model "$BASE_MODEL"
    --loras_path "$LORAS_PATH"
    --lora_name_template "$LORA_NAME_TEMPLATE"
    --num_loras "$NUM_LORAS"
    --lora_indices "${LORA_INDICES[@]}"
    --proportions_option "uniform"
    --num_agents "$NUM_AGENTS"
    --num_news_agents "$NUM_NEWS_AGENTS"
    --proportion_adversarial_agents "$PROPORTION_ADVERSARIAL_AGENTS"
    --adversarial_strategy "$ADVERSARIAL_STRATEGY"
    --adversarial_model "$ADVERSARIAL_MODEL"
    --graph_model "$GRAPH_MODEL"
    --visualizer_output "$VISUALIZER_OUTPUT"
    --metrics_output "$METRICS_OUTPUT"
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
echo "loras_path=${LORAS_PATH}"
echo "lora_name_template=${LORA_NAME_TEMPLATE}"
echo "num_loras=${NUM_LORAS}"
echo "lora_indices=${LORA_INDEX_SET}"
echo "proportions_option=uniform"
echo "adversarial_model=${ADVERSARIAL_MODEL}"
echo "graph_model=${GRAPH_MODEL}"
echo "homophily=off"
echo "add_survey_to_context=${SURVEY_CONTEXT_FLAG}"
echo "question_number=${QUESTION_NUMBER}"
echo "proportion_adversarial_agents=${PROPORTION_ADVERSARIAL_AGENTS}"
echo "num_agents=${NUM_AGENTS}"
echo "num_news_agents=${NUM_NEWS_AGENTS}"
echo "centralize_adversaries=${CENTRALIZE_FLAG}"
echo "adversarial_strategy=${ADVERSARIAL_STRATEGY}"
echo "repo_root=${REPO_ROOT}"
echo "hf_hub_cache=${HF_HUB_CACHE}"
echo "hf_datasets_cache=${HF_DATASETS_CACHE}"
echo "unsloth_cache_dir=${UNSLOTH_CACHE_DIR}"
echo "survey_output=${SURVEY_OUTPUT}"
echo "visualizer_output=${VISUALIZER_OUTPUT}"
echo "metrics_output=${METRICS_OUTPUT}"
echo "=================================================="

cd "$REPO_ROOT"
"${CMD[@]}"
