#!/usr/bin/env bash
# Source this file from a Great Lakes shell:
#   source scripts/activate_great_lakes.sh

module purge
module load python/3.11.5
module load cuda/12.8.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PQS_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

export PQS_ROOT="${PQS_ROOT:-$DEFAULT_PQS_ROOT}"
export HF_HOME="${HF_HOME:-$PQS_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PQS_ROOT/hf_datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PQS_ROOT/pip-cache}"
export PYTHONNOUSERSITE=1

source "$PQS_ROOT/envs/posttrain-quant-serve/bin/activate"

echo "PQS_ROOT=$PQS_ROOT"
echo "HF_HOME=$HF_HOME"
which python
