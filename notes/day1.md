# Day 1 Checklist

Goal: make the GRPO smoke path reliable, not bigger.

## Tasks

- [x] Review the 5-step shape-check log.
- [x] Fix only Day 0 blockers: environment, model download, dataset loading, reward parser, checkpoint path.
- [x] Run the 100-step GRPO smoke job.
- [x] Run a short resume check from the latest `checkpoint-*`.
- [x] Record reward, reward std, KL, completion length, GPU, peak memory, and checkpoint path in `notes/study-log.md`.
- [x] Commit the smoke-run result.

## Do Not Do Yet

- Do not start Qwen3-1.7B.
- Do not install vLLM/AWQ.
- Do not rebuild the training stack unless the current stack is broken.
- Do not spend time improving GSM8K accuracy.

## Stop Condition

Day 1 is done when the 100-step Qwen2.5-0.5B-Instruct GRPO smoke run has a checkpoint that resumes.

## Current Status

Done:

- 5-step GRPO shape check completed.
- Resume check from `checkpoint-5` completed and continued to step 10.
- 100-step GRPO smoke run completed and saved `checkpoint-100`.
- Completion notebook loaded 400 logged completions.
- Reward signal was nonzero: mean exact-match reward `0.235`, max reward `1.0`.
- Parser extracted an answer from all logged completions.

Conclusion:

- Day 1 is complete. The single-GPU GRPO smoke path is reliable enough to move into Day 2
  evaluation/reward-cleanup work.

Next:

- Run base-vs-trained evaluation to measure whether `checkpoint-100` is better than the base model.
- Reduce prompt leakage before increasing dataset size or steps.
- Do not scale model size until the reward/completion behavior is understood.
