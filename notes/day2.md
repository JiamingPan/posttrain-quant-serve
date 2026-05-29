# Day 2 Checklist

Goal: inspect the completed 100-step GRPO smoke run and make the reward signal reliable before
running anything larger.

## Tasks

- [x] Pull the latest repo on Great Lakes so `slurm/jupyter_gpu.sbatch` is available.
- [x] Launch the Jupyter Slurm job and open `notebooks/grpo_smoke_analysis.ipynb`.
- [x] Inspect the full saved `completions/` artifacts for job `50954114`.
- [x] Record whether any completions receive reward `1.0`.
- [x] Check for prompt leakage, repeated `Human:` text, clipped completions, and final-answer
  formatting failures.
- [x] Record reward mean/std, KL if available, runtime, checkpoint path, and warnings in
  `notes/study-log.md`.
- [x] Compare base Qwen2.5-0.5B-Instruct vs the 100-step GRPO checkpoint with
  `slurm/eval_gsm8k_compare.sbatch`.
- [ ] Inspect the worsened paired-comparison example from the train-10 eval.
- [x] Rerun a 5-step GRPO check with the prompt-leak reward penalty and confirm
  the new reward values appear. Do not expect leakage rate to move meaningfully in only 5 steps.
- [x] Run a longer follow-up to `checkpoint-300` with the leak penalty.
- [ ] Use the notebook to compare `prompt_leak_rate` before vs after the penalty.
- [ ] If rewards are nonzero sometimes, run the next smoke job with more examples, not a bigger model.
- [ ] If rewards are still effectively zero, improve prompt/reward behavior before more training.

## Decision Rule

The next training run should be one of:

- reward signal looks usable but eval worsens: inspect paired outputs, then run `MAX_STEPS=300`, `DATASET_LIMIT=50`,
  same Qwen2.5-0.5B-Instruct model with the leak penalty.
- reward signal is broken: patch prompt/reward/parser and rerun a 5-step check.

Do not move to Qwen3-1.7B, vLLM, or AWQ until this is resolved.
