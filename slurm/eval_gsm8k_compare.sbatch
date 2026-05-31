#!/usr/bin/env bash
# Compare base Qwen2.5 Instruct against the GRPO checkpoint in FP16/W4/AWQ.
#
# Submit from repo root, for example:
#   sbatch --account=cavestru0 --time=00:30:00 \
#     --export=ALL,EVAL_SPLIT=test,EVAL_LIMIT=50,MODEL_TAG=qwen2_5_0_5b,PRECISIONS=fp16,w4 \
#     slurm/eval_gsm8k_compare.sbatch

#SBATCH --job-name=pqs-gsm8k-eval
#SBATCH --account=huterer0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/activate_great_lakes.sh

python scripts/cluster_check.py

MODEL_TAG="${MODEL_TAG:-qwen2_5_0_5b}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"
PRECISIONS="${PRECISIONS:-fp16,w4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MAX_REFERENCE_TOKENS="${MAX_REFERENCE_TOKENS:-512}"
PRECISION_SLUG="${PRECISIONS//,/_}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PQS_ROOT/evals/gsm8k_compare_${EVAL_SPLIT}${EVAL_LIMIT}_${MODEL_TAG}_g8_dr100_${PRECISION_SLUG}}"

ppl_args=()
if [[ "${SKIP_REFERENCE_PPL:-0}" == "1" ]]; then
  ppl_args+=(--skip_reference_ppl)
fi

model_args=(--model_tag "$MODEL_TAG")
if [[ -n "${BASE_MODEL:-}" ]]; then
  model_args+=(--base_model "$BASE_MODEL")
fi
if [[ -n "${TRAINED_MODEL:-}" ]]; then
  model_args+=(--trained_model "$TRAINED_MODEL")
fi
if [[ -n "${BASE_AWQ_MODEL:-}" ]]; then
  model_args+=(--base_awq_model "$BASE_AWQ_MODEL")
fi
if [[ -n "${TRAINED_AWQ_MODEL:-}" ]]; then
  model_args+=(--trained_awq_model "$TRAINED_AWQ_MODEL")
fi

python scripts/eval_gsm8k_compare.py \
  "${model_args[@]}" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --split "$EVAL_SPLIT" \
  --limit "$EVAL_LIMIT" \
  --precisions "$PRECISIONS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_reference_tokens "$MAX_REFERENCE_TOKENS" \
  "${ppl_args[@]}"
