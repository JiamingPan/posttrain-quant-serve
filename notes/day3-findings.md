# Day 3 Findings

Status: diagnostic changes are in place; held-out eval and direct dynamics conclusions still need Great
Lakes artifacts. Do not treat this as permission to run a larger model.

## Current Evidence

The train-10 evals show a real warning sign:

- 100-step checkpoint: base `0.60` -> trained `0.50`.
- 300-step leak-penalty checkpoint: base `0.60` -> trained `0.20`.
- 300-step reference PPL: base `1.878` -> trained `1.934`.
- Final training loss was essentially zero.

This supports the advantage/reward-std-collapse hypothesis indirectly, but it does not prove it yet.
The notebook now has the plots needed to check reward standard deviation, zero-reward-variance prompt
groups, advantage magnitude, and KL when those artifacts are present.

The first objective-fix smoke has a better train-10 result:

- `g8_dr100` checkpoint: base `0.10` -> trained `0.30`.
- Paired changes: `2` improved, `0` worsened, `8` unchanged.
- Prompt leak stayed low but nonzero for trained: base `0.00` -> trained `0.10`.
- Reference PPL still worsened slightly: base `1.878` -> trained `1.904`.

Interpretation: the objective-fix direction is promising enough to evaluate on held-out data, but it is
not proven. Do not run 300 steps until test-50 and notebook dynamics support it.

Held-out `test50_g8_dr100` is positive but still weak in absolute terms:

- Base `0.06` -> trained `0.22`.
- Paired changes: `9` improved, `1` worsened, `40` unchanged.
- Prompt leakage stayed at `0.00`.

Interpretation: the objective-fix settings improved this tiny model on a 50-question held-out sample,
but the trained model is still only at `22%` exact match. The project should treat this as evidence
that the old objective was harmful and the new direction is worth diagnosing further, not as a strong
model-quality result.

Also note that the notebook's detailed dynamics section currently defaults to `RUN_DIR`, which is the
old 300-step leak-penalty run. Its reward/std/KL plots do not describe `g8_dr100` unless `RUN_DIR` is
changed to `OBJECTIVE_FIX_RUN_DIR` and `NUM_GENERATIONS` is changed to `8`.

The KL/loss spike visible in the old 300-step run should not be ignored. The notebook now prints
trainer-history rows above the 99th percentile and shows both raw and robust y-axis plots, so a single
logged spike does not flatten the rest of the curve. Treat that spike as a diagnostic clue, not as a
plotting artifact to silently delete.

## TRL 1.5.0 Knobs Exposed

The trainer now exposes these optional knobs while preserving current defaults unless flags are set:

- `--num_generations`
- `--temperature`
- `--top_p`, `--top_k`, `--min_p`, `--repetition_penalty`
- `--scale_rewards` with `group`, `batch`, or `none`
- `--loss_type`
- `--epsilon_high`
- `--mask_truncated_completions`
- `--beta`

`build_grpo_config` still filters unsupported fields, so the script remains tolerant of TRL version
mismatches. Run `python scripts/inspect_grpo_config.py` on Great Lakes to print the exact installed
field list.

## Held-Out Eval Commands

Run these before interpreting the train-10 regression as final. They compare base vs trained on the
same held-out test examples with the current eval prompt and generation defaults.

100-step checkpoint:

```bash
sbatch --job-name=pqs-eval-100-test50 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,EVAL_SPLIT=test,EVAL_LIMIT=50,TRAINED_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/smoke_qwen2_5_0_5b_grpo_100step,EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_test50_100step \
  slurm/eval_gsm8k_compare.sbatch
```

300-step leak-penalty checkpoint:

```bash
sbatch --job-name=pqs-eval-leak300-test50 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,EVAL_SPLIT=test,EVAL_LIMIT=50,TRAINED_MODEL=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k,EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_test50_leak300 \
  slurm/eval_gsm8k_compare.sbatch
```

## Recommended Next Run

If the notebook confirms high zero-reward-variance groups or near-zero advantage, do not increase
steps first. The next smoke should increase group diversity, reduce reward scaling bias, and remove
KL as a possible fallback gradient during the diagnostic:

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

Rationale: `NUM_GENERATIONS=8` gives each prompt more chances to produce reward disagreement, and
`LOSS_TYPE=dr_grpo` / `SCALE_REWARDS=none` tests the Dr. GRPO-style fix if the installed TRL accepts
those fields. `BETA=0.0` is intentional for this diagnostic: if rewards and advantages collapse, a
nonzero KL penalty can become the only meaningful gradient, so this run asks whether the reward signal
can train the model without KL masking the failure.

If TRL rejects `dr_grpo` or `scale_rewards=none`, use `python scripts/inspect_grpo_config.py` and the
printed `Dropping unsupported GRPOConfig fields` output to choose the closest supported option. Do not
move to Qwen3-1.7B, vLLM, AWQ, or quantization until the 0.5B GRPO signal is sane.
