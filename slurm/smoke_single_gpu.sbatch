#!/usr/bin/env bash
#SBATCH --job-name=pqs-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load the cluster's CUDA/Python modules here if needed.
# module load cuda/...

source .venv/bin/activate

python scripts/cluster_check.py
bash scripts/prepare_torchtune_config.sh

CONFIG=configs/smoke_qwen2_5_0_5b.yaml \
MAX_STEPS="${MAX_STEPS:-100}" \
bash scripts/finetune.sh
