# Reading Queue

Start with `notes/project-overview.md`. Then use this reading order.

## Day 0

1. TRL GRPOTrainer docs
   - Extract: reward function signature, logged metrics, config knobs.
2. DeepSeekMath GRPO section
   - Extract: why GRPO avoids a critic and how group-relative rewards work.
3. `notes/grpo.md`
   - Extract: local definitions and what to watch in the smoke logs.

## Day 1

1. DeepSeek-R1
   - Extract: RL post-training stages, verifiable rewards, KL, and reasoning behavior.
2. PyTorch FSDP / ZeRO refresher
   - Extract: how to scale the policy model once single-GPU GRPO works.

## Later

1. vLLM/PagedAttention
2. AWQ or GPTQ
