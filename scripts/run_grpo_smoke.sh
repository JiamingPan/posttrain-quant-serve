#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PQS_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

: "${PQS_ROOT:=$DEFAULT_PQS_ROOT}"
: "${MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${OUTPUT_DIR:=$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo}"
: "${MAX_STEPS:=5}"
: "${DATASET_LIMIT:=10}"
: "${NUM_GENERATIONS:=4}"
: "${LEARNING_RATE:=1e-6}"
: "${BETA:=0.02}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=4}"
: "${MAX_COMPLETION_LENGTH:=128}"
: "${RESUME_FROM_CHECKPOINT:=}"

mkdir -p "$(dirname "$OUTPUT_DIR")" "${HF_HOME:-$PQS_ROOT/hf_cache}" "${HF_DATASETS_CACHE:-$PQS_ROOT/hf_datasets}"

cmd=(
  accelerate launch --num_processes 1 scripts/train_grpo_gsm8k.py
  --model "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --max_steps "$MAX_STEPS"
  --dataset_limit "$DATASET_LIMIT"
  --num_generations "$NUM_GENERATIONS"
  --per_device_train_batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LEARNING_RATE"
  --beta "$BETA"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
)


optional_opts=(
  "TEMPERATURE:--temperature"
  "TOP_P:--top_p"
  "TOP_K:--top_k"
  "MIN_P:--min_p"
  "REPETITION_PENALTY:--repetition_penalty"
  "SCALE_REWARDS:--scale_rewards"
  "LOSS_TYPE:--loss_type"
  "EPSILON_HIGH:--epsilon_high"
)

for opt in "${optional_opts[@]}"; do
  var="${opt%%:*}"
  flag="${opt#*:}"
  value="${!var:-}"
  if [[ -n "$value" ]]; then
    cmd+=("$flag" "$value")
  fi
done

if [[ "${MASK_TRUNCATED_COMPLETIONS:-0}" == "1" ]]; then
  cmd+=(--mask_truncated_completions)
fi


if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
