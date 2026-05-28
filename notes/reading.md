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

## Day 2

1. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
   - URL: https://arxiv.org/abs/2402.03300
   - Extract: GRPO objective, group-relative advantages, KL penalty, and math reward setup.
2. Training Verifiers to Solve Math Word Problems
   - URL: https://arxiv.org/abs/2110.14168
   - Extract: what GSM8K measures and why exact final-answer checking is useful.
3. TRL GRPOTrainer docs
   - URL: https://huggingface.co/docs/trl/grpo_trainer
   - Extract: logged metrics, completion logging, reward function contract, generation settings.

## Later

1. vLLM/PagedAttention
2. AWQ or GPTQ
