#!/usr/bin/env bash
# Train a single-GPU GRPO GSM8K checkpoint with chat-formatted prompts.
#
# Default recipe: Qwen2.5-1.5B-Instruct, 1000 train prompts, 250 optimizer
# steps. With batch size 1 and grad accumulation 8, this is about two epochs.

#SBATCH --job-name=pqs-grpo-train
#SBATCH --account=huterer0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/activate_great_lakes.sh

python scripts/cluster_check.py

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SPLIT="${SPLIT:-train}"
DATASET_LIMIT="${DATASET_LIMIT:-1000}"
MAX_STEPS="${MAX_STEPS:-250}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LOSS_TYPE="${LOSS_TYPE:-dr_grpo}"
SCALE_REWARDS="${SCALE_REWARDS:-none}"
BETA="${BETA:-0.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PQS_ROOT/ckpts/qwen2_5_1_5b_grpo_data1000_chat}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

mkdir -p "$(dirname "$OUTPUT_DIR")" "${HF_HOME:-$PQS_ROOT/hf_cache}" "${HF_DATASETS_CACHE:-$PQS_ROOT/hf_datasets}"

cmd=(
  accelerate launch --num_processes 1 scripts/train_grpo_gsm8k.py
  --model "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --split "$SPLIT"
  --dataset_limit "$DATASET_LIMIT"
  --max_steps "$MAX_STEPS"
  --num_generations "$NUM_GENERATIONS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LEARNING_RATE"
  --beta "$BETA"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --logging_steps "$LOGGING_STEPS"
  --save_steps "$SAVE_STEPS"
  --temperature "$TEMPERATURE"
  --loss_type "$LOSS_TYPE"
  --scale_rewards "$SCALE_REWARDS"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

if [[ "${MASK_TRUNCATED_COMPLETIONS:-0}" == "1" ]]; then
  cmd+=(--mask_truncated_completions)
fi

echo "=== GRPO train config ==="
echo "MODEL=$MODEL"
echo "SPLIT=$SPLIT"
echo "DATASET_LIMIT=$DATASET_LIMIT"
echo "MAX_STEPS=$MAX_STEPS"
echo "NUM_GENERATIONS=$NUM_GENERATIONS"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "LOSS_TYPE=$LOSS_TYPE"
echo "SCALE_REWARDS=$SCALE_REWARDS"
echo "BETA=$BETA"
echo "TEMPERATURE=$TEMPERATURE"
echo "MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH"
echo "MAX_COMPLETION_LENGTH=$MAX_COMPLETION_LENGTH"
echo "SAVE_STEPS=$SAVE_STEPS"
echo "LOGGING_STEPS=$LOGGING_STEPS"
echo "OUTPUT_DIR=$OUTPUT_DIR"

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
