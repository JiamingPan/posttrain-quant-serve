#!/usr/bin/env bash
#SBATCH --job-name=pqs-grpo-smoke
#SBATCH --account=huterer0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/activate_great_lakes.sh

python scripts/cluster_check.py

MAX_STEPS="${MAX_STEPS:-5}" \
DATASET_LIMIT="${DATASET_LIMIT:-10}" \
OUTPUT_DIR="${OUTPUT_DIR:-$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo}" \
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}" \
bash scripts/run_grpo_smoke.sh
