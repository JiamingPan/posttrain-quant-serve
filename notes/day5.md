# Day 5 — Data1000 GRPO and Final Quantization Matrix

Day 5 starts from a clean pipeline:

- train/eval/AWQ calibration now use Qwen chat templates,
- eval stopping is fixed (`<|im_end|>` / EOS, `max_new_tokens=512`),
- reward parsing and reward variance were smoke-tested after the format change,
- the old raw-prompt matrices are documented as stopping/truncation diagnostics,
  not final claims.

The Day 5 goal is to finish the GSM8K data1000 run and ship the quantization
comparison. Do not switch to MATH, larger models, vLLM, FSDP, or multi-GPU unless
explicitly redirected.

## Current Active Run

Real data1000 GRPO job:

- job: `51190676`
- state at launch: `RUNNING`
- output checkpoint:
  `/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_1_5b_grpo_data1000_chat`
- wrapper commit: `5c5f9f2 Add GRPO data1000 training wrapper`
- wrapper: `slurm/train_grpo.sbatch`

Recipe:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- split: `train`
- `DATASET_LIMIT=1000`
- `MAX_STEPS=250`
- `PER_DEVICE_TRAIN_BATCH_SIZE=1`
- `GRAD_ACCUM=8`
- effective prompts/step: `8`
- approximate epochs: `250 * 8 / 1000 = 2`
- `NUM_GENERATIONS=8`
- `LEARNING_RATE=1e-6`
- `LOSS_TYPE=dr_grpo`
- `SCALE_REWARDS=none`
- `BETA=0.0`
- `TEMPERATURE=1.0`
- `MAX_COMPLETION_LENGTH=512`

Preflight passed before launch:

- job: `51190636`
- state: `COMPLETED 0:0`
- config confirmed: `DATASET_LIMIT=16`, `MAX_STEPS=4`, `NUM_GENERATIONS=8`,
  `GRAD_ACCUM=8`, `MAX_COMPLETION_LENGTH=512`
- reward diagnostics had nonzero `reward_std` on calls 1 and 4.

Check the real job:

```bash
squeue -j 51190676

sacct -j 51190676 --format=JobID,JobName%25,State,ExitCode,Elapsed

grep -n "=== GRPO train config ===" -A20 logs/pqs-grpo-data1000-51190676.out
grep -n "reward_diag" logs/pqs-grpo-data1000-51190676.out | tail -n 20

ls -lh /scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_1_5b_grpo_data1000_chat
```

Guardrail: if `[reward_diag]` collapses early to `reward_std=0.0000` and
`zero_reward_variance_group_frac=1.0000` on every call, stop and inspect before
continuing. Some zero-variance calls are fine; all calls collapsing is not.

## Step 1 — Held-Out FP16 Eval

Run this after job `51190676` completes cleanly:

```bash
MODEL_TAG=qwen2_5_1_5b \
PRECISIONS=fp16 \
EVAL_SPLIT=test \
EVAL_LIMIT=100 \
DENSE_DTYPE=fp16 \
SKIP_REFERENCE_PPL=1 \
TRAINED_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_1_5b_grpo_data1000_chat \
EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_chat_test100_qwen2_5_1_5b_data1000_fp16 \
sbatch --job-name=pqs-eval-data1000-fp16 \
  --account=cavestru0 \
  --time=00:45:00 \
  --export=ALL \
  slurm/eval_gsm8k_compare.sbatch
```

Read it:

```bash
OUT=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_chat_test100_qwen2_5_1_5b_data1000_fp16
cat "$OUT/results_matrix.csv"
```

Metrics to record:

- `base_fp16` accuracy
- `g8_dr100_fp16` accuracy for the data1000 checkpoint
- `delta_fp16 = g8_dr100_fp16 - base_fp16`
- `hit_max_new_tokens_rate` should stay near `0`
- `ended_with_eos_rate` should stay near `1`
- `prompt_leak_rate` should stay near `0`

Proceed to quantization regardless of whether `delta_fp16` is positive, flat, or
negative. The project question is whether RL post-training changes quantization
behavior, not whether this one run produces a large GSM8K gain.

## Step 2 — AWQ Quantization

bnb NF4 W4 is load-time quantization, so it does not need a saved checkpoint. AWQ
does need saved W4G128 checkpoints.

Base AWQ with chat calibration:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-1.5B-Instruct \
AWQ_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts_awq/qwen2_5_1_5b_base_awq_w4g128_chatcalib \
sbatch --job-name=pqs-awq-base-chat \
  --account=cavestru0 \
  --time=01:00:00 \
  --export=ALL \
  slurm/quantize_awq.sbatch
```

Data1000 GRPO AWQ:

```bash
MODEL_NAME_OR_PATH=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_1_5b_grpo_data1000_chat \
AWQ_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts_awq/qwen2_5_1_5b_data1000_chat_awq_w4g128 \
sbatch --job-name=pqs-awq-data1000 \
  --account=cavestru0 \
  --time=01:00:00 \
  --export=ALL \
  slurm/quantize_awq.sbatch
```

## Step 3 — Full 6-Row Matrix

Run after both AWQ jobs complete:

```bash
MODEL_TAG=qwen2_5_1_5b \
PRECISIONS=fp16,w4,awq \
EVAL_SPLIT=test \
EVAL_LIMIT=100 \
DENSE_DTYPE=fp16 \
SKIP_REFERENCE_PPL=1 \
TRAINED_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_1_5b_grpo_data1000_chat \
BASE_AWQ_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts_awq/qwen2_5_1_5b_base_awq_w4g128_chatcalib \
TRAINED_AWQ_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts_awq/qwen2_5_1_5b_data1000_chat_awq_w4g128 \
EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_chat_test100_qwen2_5_1_5b_data1000_fp16_w4_awq \
sbatch --job-name=pqs-eval-data1000-full \
  --account=cavestru0 \
  --time=01:30:00 \
  --export=ALL \
  slurm/eval_gsm8k_compare.sbatch
```

Final table should include:

- `base_fp16`
- `g8_dr100_fp16`
- `base_w4`
- `g8_dr100_w4`
- `base_awq`
- `g8_dr100_awq`

Headline metrics:

- `delta_fp16`
- `delta_w4`
- `delta_awq`
- `gain_survival_w4 = delta_w4 - delta_fp16`
- `gain_survival_awq = delta_awq - delta_fp16`

## Interpretation Branches

If data1000 GRPO has a held-out FP16 gain:

> GRPO produced a held-out GSM8K gain; report whether that gain survives bnb-NF4
> W4 and AWQ using `gain_survival_w4` and `gain_survival_awq`.

If data1000 GRPO is flat or worse:

> On an already-strong Qwen2.5-1.5B-Instruct GSM8K setup, this GRPO recipe did not
> produce a clear held-out gain. Still report the quant matrix as the answer to
> whether the RL-updated checkpoint became harder/easier to quantize, paired with
> the weight-outlier diagnostics showing near-identical global quantizability.

Either way, ship the GSM8K result. MATH/harder-task headroom is future work.
