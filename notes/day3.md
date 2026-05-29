# Day 3 Checklist

Goal: fix the bad GRPO behavior from Day 2 before running any larger job.

Starting evidence:

- 100-step checkpoint was slightly worse than base on train-10: `0.60 -> 0.50`.
- 300-step leak-penalty checkpoint was clearly worse: `0.60 -> 0.20`.
- Reference PPL got worse: `1.878 -> 1.934`.
- Eval prompt leakage stayed flat at `0.50` for both base and trained.
- Therefore the next step is diagnosis and format/generation cleanup, not scaling.

## Tasks

- [ ] Pull latest repo on Great Lakes.
- [ ] Rerun `notebooks/grpo_smoke_analysis.ipynb` so the negative Day 2 result is visible in the
  notebook scorecard.
- [ ] Inspect all four worsened examples from
  `/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_train10_leak300/paired_comparison.csv`.
- [x] Tighten the train/eval prompt so the model is told to stop after the `#### <answer>` line.
- [x] Reduce default training completion length from `256` to `128`.
- [x] Reduce default eval generation length from `192` to `96`.
- [ ] Run or inspect the submitted 100-step format-fix smoke job `51123578`.
- [ ] Evaluate the format-fix checkpoint before any 300-step rerun.
- [ ] Run held-out test-50 evals for both old checkpoints to check whether the regression survives beyond train-10 noise.
- [ ] Use the notebook dynamics plots to check reward-std / zero-advantage / KL collapse.
- [ ] Decide whether another GRPO run is justified or whether reward/generation needs another patch.

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

## Stop Condition

Day 3 is done when the project has either:

- a format-fix checkpoint whose small eval is not catastrophically worse than base, or
- a clear diagnosis showing that the reward/prompt setup still needs another patch before more GRPO.

Do not move to Qwen3-1.7B, vLLM, AWQ, or quantization diagnostics today.
