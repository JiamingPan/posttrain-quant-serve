#!/usr/bin/env bash
# Offline weight/outlier diagnostics for base vs GRPO checkpoints.
#
# Submit from repo root, for example:
#   sbatch --account=cavestru0 --partition=spgpu --time=00:30:00 \
#     slurm/weight_outliers.sbatch

#SBATCH --job-name=pqs-weight-outliers
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/activate_great_lakes.sh

OUTPUT_DIR="${OUTPUT_DIR:-$PQS_ROOT/diagnostics/weight_outliers_0p5b_g8_dr100}"
MODEL_SPECS="${MODEL_SPECS:-base=Qwen/Qwen2.5-0.5B-Instruct g8_dr100=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_g8_dr100}"
DTYPE="${DTYPE:-auto}"
GROUP_SIZE="${GROUP_SIZE:-128}"
OUTLIER_MULTIPLIER="${OUTLIER_MULTIPLIER:-6.0}"
NO_QUANT_ERROR="${NO_QUANT_ERROR:-0}"
INCLUDE_EMBEDDINGS="${INCLUDE_EMBEDDINGS:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

args=(
  --output_dir "$OUTPUT_DIR"
  --dtype "$DTYPE"
  --group_size "$GROUP_SIZE"
  --outlier_multiplier "$OUTLIER_MULTIPLIER"
)

read -r -a model_specs <<< "$MODEL_SPECS"
for model_spec in "${model_specs[@]}"; do
  args+=(--model "$model_spec")
done

if [[ "$NO_QUANT_ERROR" == "1" ]]; then
  args+=(--no_quant_error)
fi

if [[ "$INCLUDE_EMBEDDINGS" == "1" ]]; then
  args+=(--include_embeddings)
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  args+=(--local_files_only)
fi

python scripts/weight_outlier_diagnostics.py "${args[@]}"
