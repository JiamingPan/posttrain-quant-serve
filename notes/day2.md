# Day 2 Checklist

Goal: inspect the completed 100-step GRPO smoke run and make the reward signal reliable before
running anything larger.

## Tasks

- [ ] Pull the latest repo on Great Lakes so `slurm/jupyter_gpu.sbatch` is available.
- [ ] Launch the Jupyter Slurm job and open `notebooks/grpo_smoke_analysis.ipynb`.
- [ ] Inspect the full saved `completions/` artifacts for job `50954114`.
- [ ] Record whether any completions receive reward `1.0`.
- [ ] Check for prompt leakage, repeated `Human:` text, clipped completions, and final-answer
  formatting failures.
- [ ] Record reward mean/std, KL if available, runtime, checkpoint path, and warnings in
  `notes/study-log.md`.
- [ ] Compare base Qwen2.5-0.5B-Instruct vs the 100-step GRPO checkpoint with
  `slurm/eval_gsm8k_compare.sbatch`.
- [ ] If rewards are nonzero sometimes, run the next smoke job with more examples, not a bigger model.
- [ ] If rewards are still effectively zero, improve prompt/reward behavior before more training.

## Decision Rule

The next training run should be one of:

- reward signal looks usable: `MAX_STEPS=300`, `DATASET_LIMIT=50`, same Qwen2.5-0.5B-Instruct model.
- reward signal is broken: patch prompt/reward/parser and rerun a 5-step check.

Do not move to Qwen3-1.7B, vLLM, or AWQ until this is resolved.
