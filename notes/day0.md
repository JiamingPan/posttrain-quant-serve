# Day 0 Checklist

Goal: run the smallest GRPO smoke test. Do not touch larger models yet.

## Tasks

- [x] Read `notes/project-overview.md`.
- [x] Read `notes/grpo.md`.
- [x] Create or verify the Great Lakes scratch environment.
- [x] Run `scripts/cluster_check.py` on a GPU node.
- [x] Confirm TRL imports and `GRPOTrainer` is available.
- [x] Download/cache Qwen2.5-0.5B-Instruct and GSM8K under the scratch HF cache.
- [x] Launch a 5-step single-GPU GRPO shape check.
- [x] Launch a 100-step single-GPU GRPO smoke run.
- [x] Relaunch from the saved checkpoint for a short resume check.
- [x] Record exact commands, GPU, peak memory, reward movement, KL, checkpoint path, and reload result in `notes/study-log.md`.

## Stop Condition

Day 0 is done when the GRPO smoke run has a checkpoint that can be reloaded. If setup breaks before that, record the exact failure and make that the next fix.
