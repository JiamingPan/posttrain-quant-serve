#!/usr/bin/env bash
# Compare base Qwen2.5-0.5B-Instruct against the GRPO-trained checkpoint.
#
# Submit from repo root, for example:
#   sbatch --account=cavestru0 --time=00:30:00 \
#     --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10 \
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

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TRAINED_MODEL="${TRAINED_MODEL:-$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo_100step}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
MAX_REFERENCE_TOKENS="${MAX_REFERENCE_TOKENS:-512}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PQS_ROOT/evals/gsm8k_compare_${EVAL_SPLIT}_${EVAL_LIMIT}}"

ppl_args=()
if [[ "${SKIP_REFERENCE_PPL:-0}" == "1" ]]; then
  ppl_args+=(--skip_reference_ppl)
fi

python scripts/eval_gsm8k_compare.py \
  --base_model "$BASE_MODEL" \
  --trained_model "$TRAINED_MODEL" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --split "$EVAL_SPLIT" \
  --limit "$EVAL_LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_reference_tokens "$MAX_REFERENCE_TOKENS" \
  "${ppl_args[@]}"
