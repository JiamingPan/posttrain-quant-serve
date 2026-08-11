#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/launch_slurm.sh --stage STAGE --gpus {1,2,4,8} [options] [-- STAGE_ARGS...]

Single-node Great Lakes launcher for the direct PyTorch FSDP2 track.

Stages:
  sft             train.fsdp_sft under torchrun
  grpo            train.fsdp_grpo under torchrun
  correctness     bench.correctness under torchrun
  consolidate     scripts.consolidate_dcp under torchrun
  scaling-worker  one bench.scaling worker under torchrun
  scaling         the 1/2/4/8 bench.scaling controller (requires 8 GPUs)

Options:
  --stage NAME
  --gpus COUNT
  --distributed-timeout-seconds SECONDS  Default: 600
  --dry-run                              Validate and print without loading modules
  -h, --help
EOF
}

fail() {
  printf 'launch_slurm.sh: %s\n' "$*" >&2
  exit 2
}

STAGE=""
GPUS=""
DRY_RUN=0
DIST_TIMEOUT_SECONDS="${PQS_DISTRIBUTED_TIMEOUT_SECONDS:-600}"
STAGE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ $# -ge 2 ]] || fail "--stage requires a value"
      STAGE="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || fail "--gpus requires a value"
      GPUS="$2"
      shift 2
      ;;
    --distributed-timeout-seconds)
      [[ $# -ge 2 ]] || fail "--distributed-timeout-seconds requires a value"
      DIST_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      STAGE_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown launcher argument: $1 (put stage arguments after --)"
      ;;
  esac
done

[[ -n "$STAGE" ]] || fail "--stage is required"
[[ -n "$GPUS" ]] || fail "--gpus is required"
[[ "$GPUS" =~ ^(1|2|4|8)$ ]] || fail "--gpus must be one of 1, 2, 4, or 8"
[[ "$DIST_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "distributed timeout must be a positive integer"

SLURM_NODE_COUNT="${SLURM_NNODES:-1}"
[[ "$SLURM_NODE_COUNT" == "1" ]] || fail "this track requires a single Slurm node"

MODULE=""
CONTROLLER=0
case "$STAGE" in
  sft)
    MODULE="train.fsdp_sft"
    ;;
  grpo)
    MODULE="train.fsdp_grpo"
    ;;
  correctness)
    MODULE="bench.correctness"
    ;;
  consolidate)
    MODULE="scripts.consolidate_dcp"
    ;;
  scaling-worker)
    MODULE="bench.scaling"
    ;;
  scaling)
    CONTROLLER=1
    [[ "$GPUS" == "8" ]] || fail "the complete scaling controller requires --gpus 8"
    ;;
  *)
    fail "unknown stage: $STAGE"
    ;;
esac

visible_gpu_list() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s' "$CUDA_VISIBLE_DEVICES"
  elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
    printf '%s' "$SLURM_STEP_GPUS"
  elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    printf '%s' "$SLURM_JOB_GPUS"
  fi
}

VISIBLE_GPUS="$(visible_gpu_list)"
if [[ -n "$VISIBLE_GPUS" ]]; then
  IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$VISIBLE_GPUS"
  [[ "${#VISIBLE_GPU_IDS[@]}" -eq "$GPUS" ]] || fail \
    "visible GPU count ${#VISIBLE_GPU_IDS[@]} does not match --gpus $GPUS ($VISIBLE_GPUS)"
fi

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export PQS_DISTRIBUTED_TIMEOUT_SECONDS="$DIST_TIMEOUT_SECONDS"

has_stage_argument() {
  local requested="$1"
  local argument
  for argument in "${STAGE_ARGS[@]}"; do
    if [[ "$argument" == "$requested" || "$argument" == "$requested="* ]]; then
      return 0
    fi
  done
  return 1
}

case "$STAGE" in
  sft|grpo|correctness|consolidate|scaling-worker|scaling)
    if ! has_stage_argument "--timeout_seconds"; then
      STAGE_ARGS+=(--timeout_seconds "$DIST_TIMEOUT_SECONDS")
    fi
    ;;
esac

if [[ "$CONTROLLER" -eq 1 ]]; then
  COMMAND=(python -m bench.scaling "${STAGE_ARGS[@]}")
else
  COMMAND=(
    torchrun
    --standalone
    --nnodes=1
    "--nproc-per-node=$GPUS"
    --module
    "$MODULE"
  )
  if [[ "$STAGE" == "scaling-worker" ]]; then
    COMMAND+=(--worker)
  fi
  COMMAND+=("${STAGE_ARGS[@]}")
fi

printf 'stage=%s\n' "$STAGE"
printf 'gpus=%s\n' "$GPUS"
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-}"
printf 'SLURM_NNODES=%s\n' "$SLURM_NODE_COUNT"
printf 'visible_gpus=%s\n' "${VISIBLE_GPUS:-unrestricted}"
printf 'NCCL_DEBUG=%s\n' "$NCCL_DEBUG"
printf 'TORCH_NCCL_ASYNC_ERROR_HANDLING=%s\n' "$TORCH_NCCL_ASYNC_ERROR_HANDLING"
printf 'TORCH_NCCL_BLOCKING_WAIT=%s\n' "$TORCH_NCCL_BLOCKING_WAIT"
printf 'PQS_DISTRIBUTED_TIMEOUT_SECONDS=%s\n' "$PQS_DISTRIBUTED_TIMEOUT_SECONDS"
for index in "${!STAGE_ARGS[@]}"; do
  printf 'stage_arg[%s]=%s\n' "$index" "${STAGE_ARGS[$index]}"
done
printf 'command='
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "run this launcher inside a Slurm allocation"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/activate_great_lakes.sh
source "$SCRIPT_DIR/activate_great_lakes.sh"
python scripts/cluster_check.py

if [[ "$STAGE" == "sft" || "$STAGE" == "grpo" ]]; then
  python -m train.memory_model \
    --preflight-stage "$STAGE" \
    --world-size "$GPUS" \
    -- "${STAGE_ARGS[@]}"
fi

exec "${COMMAND[@]}"
