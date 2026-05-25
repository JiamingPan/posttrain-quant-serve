#!/usr/bin/env bash
set -euo pipefail

: "${CONFIG:=configs/smoke_qwen2_5_0_5b.yaml}"
: "${MAX_STEPS:=100}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG"
  echo "Run: bash scripts/prepare_torchtune_config.sh"
  exit 1
fi

tune run full_finetune_single_device \
  --config "$CONFIG" \
  max_steps_per_epoch="$MAX_STEPS"
