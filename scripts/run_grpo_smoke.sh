#!/usr/bin/env bash
set -euo pipefail

: "${PQS_ROOT:=/scratch/huterer_root/huterer0/jiamingp/pqs}"
: "${MODEL:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${OUTPUT_DIR:=$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo}"
: "${MAX_STEPS:=5}"
: "${DATASET_LIMIT:=10}"
: "${NUM_GENERATIONS:=4}"
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=4}"
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
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
