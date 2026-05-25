# Configs

Use torchtune's built-in configs as the source of truth, then commit only the small edited configs needed to reproduce a run.

For the Day 0 smoke test:

```bash
bash scripts/prepare_torchtune_config.sh
```

This copies the closest official Qwen full-finetune single-device config into:

```text
configs/smoke_qwen2_5_0_5b.yaml
```

After copying, keep edits small and record any important config changes in `notes/study-log.md`.
