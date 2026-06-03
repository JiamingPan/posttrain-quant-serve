# Evaluation Notes

Training metrics answer: did the optimizer receive a useful signal?

Before/after evaluation answers: did the updated checkpoint behave better than the original model?

For this project, both are needed.

## What The 100-Step Run Means

The GRPO smoke run updated model weights for 100 optimizer steps and saved the result under:

```text
$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo_100step
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
- reference-solution NLL / perplexity
- paired changes: improved / worsened / unchanged

Accuracy and PPL answer different questions:

- Accuracy asks whether the generated final answer is correct.
- PPL asks whether the model assigns high likelihood to the reference GSM8K solution under teacher
  forcing. Lower PPL is better, but lower PPL does not guarantee higher generated-answer accuracy.

First run the training-set sanity check:

```bash
sbatch --account=cavestru0 --time=00:30:00 \
  --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10,EVAL_OUTPUT_DIR=$PQS_ROOT/evals/gsm8k_compare_train10_100step \
  slurm/eval_gsm8k_compare.sbatch
```

Then run a small held-out check:

```bash
sbatch --account=cavestru0 --time=01:00:00 \
  --export=ALL,EVAL_SPLIT=test,EVAL_LIMIT=50,EVAL_OUTPUT_DIR=$PQS_ROOT/evals/gsm8k_compare_test50_100step \
  slurm/eval_gsm8k_compare.sbatch
```

Expected result for a tiny 100-step run: maybe little or no test improvement. That is fine. The goal
is to build the measurement harness before doing a larger run.

## First Sanity Result

The first train-10 comparison produced:

```text
base:    6/10 = 0.60
trained: 5/10 = 0.50
delta:  -0.10
paired: 0 improved, 1 worsened, 9 unchanged
```

Interpretation:

- The 100-step checkpoint is not better than the base model on this tiny sanity set.
- This does not mean GRPO cannot work. It means this specific run is only a smoke artifact.
- The next debugging target is not bigger training; it is inspecting the worsened example and
  improving the reward/prompt/generation setup.

Inspect:

```bash
cat $PQS_ROOT/evals/gsm8k_compare_train10_100step/paired_comparison.csv
```

## Leak-Penalty 300-Step Eval

After `checkpoint-300` exists, compare it against the base model:

```bash
sbatch --account=cavestru0 --time=00:45:00 \
  --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10,TRAINED_MODEL=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k,EVAL_OUTPUT_DIR=$PQS_ROOT/evals/gsm8k_compare_train10_leak300 \
  slurm/eval_gsm8k_compare.sbatch
```

This writes:

- `summary.json`: accuracy, PPL, parse rate, leakage rate.
- `paired_comparison.csv`: per-question base vs trained changes.
- `base_predictions.jsonl` and `trained_predictions.jsonl`: full generated completions.
