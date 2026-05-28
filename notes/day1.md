# Day 1 Checklist

Goal: make the GRPO smoke path reliable, not bigger.

## Tasks

- [ ] Review the 5-step shape-check log.
- [ ] Fix only Day 0 blockers: environment, model download, dataset loading, reward parser, checkpoint path.
- [ ] Run the 100-step GRPO smoke job.
- [ ] Run a short resume check from the latest `checkpoint-*`.
- [ ] Record reward, reward std, KL, completion length, GPU, peak memory, and checkpoint path in `notes/study-log.md`.
- [ ] Commit the smoke-run result.

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

Next:

- Inspect the saved completion logs, not just the printed Slurm tail.
- Decide whether the reward signal is usable before increasing dataset size or steps.
- Do not scale model size until the reward/completion behavior is understood.
