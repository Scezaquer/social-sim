# Usage: source crosssim/slurm/models.sh <family>   (paths identical to bash_job_scripts/v2/v2_run_common.sh)
case "$1" in
  qwen)
    HF_MODEL="Qwen/Qwen2.5-7B-Instruct"; LORA_DIR="$SCRATCH/Qwen"
    lora_path() { echo "$LORA_DIR/Qwen2.5-7B-Instruct-lora-finetuned-$1-no-focal"; } ;;
  minitaur)
    HF_MODEL="marcelbinz/Llama-3.1-Minitaur-8B"; LORA_DIR="$SCRATCH/marcelbinz"
    lora_path() { echo "$LORA_DIR/Llama-3.1-Minitaur-8B-lora-finetuned-unsloth-$1"; } ;;
  *) echo "unknown family $1" >&2; return 1 ;;
esac
