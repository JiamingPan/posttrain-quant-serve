# Day 3 Checklist

Goal: fix the bad GRPO behavior from Day 2 before running any larger job.

Starting evidence:

- 100-step checkpoint was slightly worse than base on train-10: `0.60 -> 0.50`.
- 300-step leak-penalty checkpoint was clearly worse: `0.60 -> 0.20`.
- Reference PPL got worse: `1.878 -> 1.934`.
- Eval prompt leakage stayed flat at `0.50` for both base and trained.
- Therefore the next step is diagnosis and controlled objective fixes, not scaling.

Updated evidence:

- 100-step format-fix checkpoint improved train-10 exact match under the stricter eval prompt:
  base `0.10` -> trained `0.20`, but reference PPL still got slightly worse.
- 100-step objective-fix checkpoint `g8_dr100` improved train-10 exact match:
  base `0.10` -> trained `0.30`, with `2` improved, `0` worsened, and `8` unchanged examples.
  Reference PPL still got slightly worse: `1.878` -> `1.904`.
- This is a useful positive smoke signal, but train-10 is too small to trust. Run held-out evals
  before any 300-step rerun.

## Definitions

- **GRPO training step** means one optimizer update to the model during `scripts/train_grpo_gsm8k.py`.
  It is controlled by `MAX_STEPS`. Example: `MAX_STEPS=100` means 100 GRPO optimizer updates.
- **Training examples** means GSM8K problems used to update the model. It is controlled by
  `DATASET_LIMIT` in the training job. Example: `DATASET_LIMIT=10` means the smoke run trains on
  10 GSM8K train-split problems.
- **Eval examples** means GSM8K problems used only for measurement. Eval does not update model
  weights. It is controlled by `EVAL_SPLIT` and `EVAL_LIMIT`.
- **`train10` eval** means `EVAL_SPLIT=train,EVAL_LIMIT=10`: evaluate on 10 examples from the
  GSM8K train split. This is fast, but it is not held out and is too small to trust by itself.
- **`test50` eval** means `EVAL_SPLIT=test,EVAL_LIMIT=50`: evaluate on 50 held-out GSM8K test
  examples. This is the next check before deciding whether to spend compute on a 300-step rerun.
- **Base model** means the original `Qwen/Qwen2.5-0.5B-Instruct`.
- **Trained model** means the checkpoint produced by one of these GRPO smoke runs.

## Run Ledger

| Name in notebook | What ran | Training config | Eval shown so far | Meaning |
| --- | --- | --- | --- | --- |
| `train10_100step` | Original Day 1/2 smoke checkpoint | `MAX_STEPS=100`, `DATASET_LIMIT=10`, `NUM_GENERATIONS=4`, old/default loss, `scale_rewards=group/default`, `BETA=0.02` | train10: base `0.60` -> trained `0.50` | Original setup got slightly worse. |
| `train10_leak300` | Old 300-step leak-penalty checkpoint | `MAX_STEPS=300`, `DATASET_LIMIT=50`, `NUM_GENERATIONS=4`, old/default loss, `scale_rewards=group/default`, `BETA=0.02` | train10: base `0.60` -> trained `0.20` | This is **not** the objective-fix run. It shows that more steps with the old setup made things worse. |
| `train10_format100` | Prompt/output-format cleanup checkpoint | `MAX_STEPS=100`, `DATASET_LIMIT=10`, `NUM_GENERATIONS=4`, old/default loss, `scale_rewards=group/default`, `BETA=0.02`, stricter prompt and shorter generation | train10: base `0.10` -> trained `0.20` | Tests whether cleaner prompt/generation helps. It is not a GRPO objective fix. |
| `train10_g8_dr100` | Objective-pathology fix checkpoint | `MAX_STEPS=100`, `DATASET_LIMIT=10`, `NUM_GENERATIONS=8`, `LOSS_TYPE=dr_grpo`, `SCALE_REWARDS=none`, `BETA=0.0` | train10: base `0.10` -> trained `0.30` | Best tiny eval so far. This is the checkpoint to test on held-out `test50` next. |
| `test50_g8_dr100` | Pending held-out eval of the objective-fix checkpoint | No training; eval only | pending | Uses 50 GSM8K test questions to check whether the train10 improvement is real. |

Important: `train10_g8_dr100` used 100 GRPO training steps. It has **not** been run for 300 GRPO
steps. The old 300-step row is `train10_leak300`, which used the old/default objective.

## Tasks

- [ ] Pull latest repo on Great Lakes.
- [ ] Rerun `notebooks/grpo_smoke_analysis.ipynb` so the negative Day 2 result is visible in the
  notebook scorecard.
- [ ] Inspect all four worsened examples from
  `/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_train10_leak300/paired_comparison.csv`.
- [x] Tighten the train/eval prompt so the model is told to stop after the `#### <answer>` line.
- [x] Reduce default training completion length from `256` to `128`.
- [x] Reduce default eval generation length from `192` to `96`.
- [x] Run and inspect the submitted 100-step format-fix smoke job `51123578`.
- [x] Evaluate the format-fix checkpoint before any 300-step rerun.
- [x] Confirm TRL 1.5.0 accepts `loss_type=dr_grpo`, `scale_rewards=none`, and the
  generation-sampling knobs exposed by `scripts/train_grpo_gsm8k.py`.
- [x] Submit the objective-pathology smoke run `pqs-grpo-g8-dr100`.
- [x] Evaluate the objective-pathology checkpoint with `pqs-eval-g8-dr100`.
- [ ] Run held-out test-50 evals for the promising Day 3 checkpoints, especially `g8_dr100`.
  Old checkpoint test-50 evals are still useful context, but the next decision depends most on the
  objective-fix result.
- [ ] Use the notebook dynamics plots to check reward-std / zero-advantage / KL collapse.
- [ ] Decide whether a 300-step rerun is justified or whether reward/generation still needs another patch.

## Why the Objective-Pathology Run Uses These Settings

The run `pqs-grpo-g8-dr100` is not a random hyperparameter sweep. It is a small combined
rescue test for two failure modes suggested by the Day 2 artifacts and the GRPO literature:

1. **Reward/advantage collapse.** GRPO compares completions within a group. If all completions
   for a prompt receive the same reward, their within-group reward standard deviation is zero
   or effectively zero. Then the normalized advantages are zero or unstable, so the policy-gradient
   signal is dead and the optimizer can drift through auxiliary terms instead of learning better
   answers.
2. **Length and reward-std normalization bias.** Standard GRPO-style losses can average a
   completion's contribution over its own length. That makes long wrong answers cheaper per token
   than short wrong answers, which matches the observed rambling, wrong completions. Reward std
   normalization can also over-weight low-variance prompt groups, which are the groups with the
   least useful learning signal.

The chosen smoke config is:

| Setting | Value | Why |
| --- | --- | --- |
| `MAX_STEPS` | `100` | Keep this as a cheap diagnostic. The old 300-step run got worse, so more steps are not justified until the objective behaves better. |
| `DATASET_LIMIT` | `10` | Keep the same tiny GSM8K subset as previous smoke runs so the result is comparable to `100step`, `leak300`, and `format100`. |
| `NUM_GENERATIONS` | `8` | Sample more completions per prompt. This increases the chance that at least one completion differs in reward, giving nonzero within-group variance and nonzero advantages. |
| `GRAD_ACCUM` | `8` | Keep the effective batch compatible with `NUM_GENERATIONS=8`; TRL requires the effective train batch to divide cleanly by the number of generations. |
| `LOSS_TYPE` | `dr_grpo` | Test the Dr. GRPO objective, which removes the response-length normalization bias that can make long wrong answers weakly penalized. |
| `SCALE_REWARDS` | `none` | Disable group reward std scaling for this smoke test so low-variance groups are not amplified just because their reward std is tiny. |
| `BETA` | `0.0` | Disable the KL penalty for this diagnostic. If advantages collapse to zero, a nonzero KL term can become the only live gradient and move the model for the wrong reason. |
| `TEMPERATURE` | `1.0` | Keep ordinary sampling while using the larger group size to improve reward diversity. This avoids changing exploration too aggressively in the same smoke run. |

This combined run will not identify which knob helped. It answers the first practical question:
whether the paper-grounded fixes can stop the obvious regression. If it improves or at least stops
the damage, run follow-up ablations. If it still gets worse, do not spend compute on 300 steps.

## Suggested Commands

Run from the Great Lakes repo root:

```bash
cd /scratch/huterer_root/huterer0/jiamingp/pqs/repos/posttrain-quant-serve
git pull --ff-only
```

Inspect worsened eval examples:

```bash
python - <<'PY'
import pandas as pd
p="/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_train10_leak300/paired_comparison.csv"
df=pd.read_csv(p)
print(df[df["change"]!="unchanged"].to_string(index=False))
PY
```

Run a short format-fix smoke:

```bash
sbatch --job-name=pqs-grpo-format5 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:20:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=5,DATASET_LIMIT=10,OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_0_5b_grpo_format_fix_5step \
  slurm/smoke_single_gpu.sbatch
```

If the 5-step run is healthy, run only a small 100-step follow-up, not another 300-step job yet.

Run the objective-pathology smoke after confirming `dr_grpo` is accepted by the installed TRL:

```bash
sbatch --job-name=pqs-grpo-g8-dr100 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:45:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=100,DATASET_LIMIT=10,NUM_GENERATIONS=8,GRAD_ACCUM=8,TEMPERATURE=1.0,LOSS_TYPE=dr_grpo,SCALE_REWARDS=none,BETA=0.0,OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_0_5b_grpo_g8_dr100 \
  slurm/smoke_single_gpu.sbatch
```

Evaluate it with:

```bash
sbatch --job-name=pqs-eval-g8-dr100 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:10:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10,TRAINED_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_0_5b_grpo_g8_dr100,EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_train10_g8_dr100 \
  slurm/eval_gsm8k_compare.sbatch
```

Run the held-out test-50 eval for the objective-fix checkpoint:

```bash
sbatch --account=cavestru0 --time=00:30:00 \
  --export=ALL,EVAL_SPLIT=test,EVAL_LIMIT=50,TRAINED_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_0_5b_grpo_g8_dr100,EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_test50_g8_dr100 \
  slurm/eval_gsm8k_compare.sbatch
```

## Stop Condition

Day 3 is done when the project has either:

- an objective-fix checkpoint whose small eval is not catastrophically worse than base and whose
  notebook diagnostics do not show reward-std / advantage collapse, or
- a clear diagnosis showing that the reward/prompt/objective setup still needs another patch before
  more GRPO.

Do not move to Qwen3-1.7B, vLLM, AWQ, or quantization diagnostics today.
