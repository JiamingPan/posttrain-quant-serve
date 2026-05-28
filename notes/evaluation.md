# Evaluation Notes

Training metrics answer: did the optimizer receive a useful signal?

Before/after evaluation answers: did the updated checkpoint behave better than the original model?

For this project, both are needed.

## What The 100-Step Run Means

The GRPO smoke run updated model weights for 100 optimizer steps and saved the result under:

```text
/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/smoke_qwen2_5_0_5b_grpo_100step
```

The notebook showed:

- `completion_rows = 400`
- `gsm8k_exact_match_reward_mean = 0.235`
- `gsm8k_exact_match_reward_max = 1.0`
- `answer_extraction_rate = 1.0`
- `prompt_leak_rate = 0.2025`

Interpretation:

- The pipeline works.
- The reward function is not dead; some generated completions got exact-match reward.
- The model still rambles or leaks into prompt-like text too often.
- This does not yet prove the checkpoint is better than the base model.

## Correct Before/After Comparison

Compare:

1. Base model: `Qwen/Qwen2.5-0.5B-Instruct`
2. Trained model: the GRPO output directory above

Run both on the same GSM8K questions with the same prompt and deterministic decoding. Record:

- exact-match accuracy
- answer parse rate
- prompt leakage rate
- mean completion length
- paired changes: improved / worsened / unchanged

First run the training-set sanity check:

```bash
sbatch --account=cavestru0 --time=00:30:00 \
  --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10,EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_train10_100step \
  slurm/eval_gsm8k_compare.sbatch
```

Then run a small held-out check:

```bash
sbatch --account=cavestru0 --time=01:00:00 \
  --export=ALL,EVAL_SPLIT=test,EVAL_LIMIT=50,EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_test50_100step \
  slurm/eval_gsm8k_compare.sbatch
```

Expected result for a tiny 100-step run: maybe little or no test improvement. That is fine. The goal
is to build the measurement harness before doing a larger run.
