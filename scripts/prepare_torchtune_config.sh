#!/usr/bin/env bash
set -euo pipefail

mkdir -p configs

TARGET="configs/smoke_qwen2_5_0_5b.yaml"

if [[ -f "$TARGET" ]]; then
  echo "Config already exists: $TARGET"
  exit 0
fi

echo "Available torchtune full_finetune_single_device configs containing qwen:"
tune ls full_finetune_single_device | grep -i qwen || true

for candidate in \
  "qwen2_5/0.5B_full_single_device" \
  "qwen2/0.5B_full_single_device" \
  "qwen2_5/1.5B_full_single_device" \
  "qwen2/1.5B_full_single_device"
do
  if tune cp "$candidate" "$TARGET"; then
    echo "Copied $candidate to $TARGET"
    exit 0
  fi
done

echo "Could not find a matching built-in torchtune Qwen full-finetune config."
echo "Run 'tune ls full_finetune_single_device' and copy the closest Qwen config manually."
exit 1
