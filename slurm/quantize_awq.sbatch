#!/usr/bin/env bash
# Quantize one checkpoint to AWQ W4G128 with AutoAWQ.
#
# Example:
#   MODEL_NAME_OR_PATH=Qwen/Qwen2.5-0.5B-Instruct \
#   AWQ_OUTPUT_DIR=$PQS_ROOT/ckpts_awq/qwen2_5_0_5b_base_awq_w4g128 \
#   sbatch --account=cavestru0 --time=00:45:00 slurm/quantize_awq.sbatch

#SBATCH --job-name=pqs-awq
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/activate_great_lakes.sh

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
AWQ_OUTPUT_DIR="${AWQ_OUTPUT_DIR:-$PQS_ROOT/ckpts_awq/qwen2_5_0_5b_base_awq_w4g128}"
AWQ_CALIB_SPLIT="${AWQ_CALIB_SPLIT:-train}"
AWQ_CALIB_LIMIT="${AWQ_CALIB_LIMIT:-128}"
AWQ_MAX_CALIB_SEQ_LEN="${AWQ_MAX_CALIB_SEQ_LEN:-512}"
AWQ_PARALLEL_CALIB="${AWQ_PARALLEL_CALIB:-8}"
AWQ_GROUP_SIZE="${AWQ_GROUP_SIZE:-128}"
AWQ_W_BIT="${AWQ_W_BIT:-4}"
AWQ_ZERO_POINT="${AWQ_ZERO_POINT:-true}"
AWQ_VERSION="${AWQ_VERSION:-GEMM}"

python scripts/cluster_check.py

echo "MODEL_NAME_OR_PATH=$MODEL_NAME_OR_PATH"
echo "AWQ_OUTPUT_DIR=$AWQ_OUTPUT_DIR"
echo "AWQ_CALIB_LIMIT=$AWQ_CALIB_LIMIT"
echo "AWQ_GROUP_SIZE=$AWQ_GROUP_SIZE"

python scripts/quantize_awq.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$AWQ_OUTPUT_DIR" \
  --calib_split "$AWQ_CALIB_SPLIT" \
  --calib_limit "$AWQ_CALIB_LIMIT" \
  --max_calib_seq_len "$AWQ_MAX_CALIB_SEQ_LEN" \
  --n_parallel_calib_samples "$AWQ_PARALLEL_CALIB" \
  --q_group_size "$AWQ_GROUP_SIZE" \
  --w_bit "$AWQ_W_BIT" \
  --zero_point "$AWQ_ZERO_POINT" \
  --version "$AWQ_VERSION"
