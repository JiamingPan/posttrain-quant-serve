# Progress And Plan Review

This is an audit note, not the chronological study log. It summarizes what the repo has actually proven so far and what should happen next before the project moves toward serving or quantization.

## Current Read

The project is still in the Qwen2.5-0.5B GRPO smoke-test phase. The training pipeline now runs on Great Lakes, saves checkpoints, resumes from checkpoints, logs completions, and supports a base-vs-trained GSM8K evaluation script. That is real progress.

The model-quality result is negative so far. The 100-step checkpoint was slightly worse than base on train-10, and the 300-step leak-penalty checkpoint was clearly worse: accuracy `0.60 -> 0.20`, reference PPL `1.878 -> 1.934`, and `0` improved / `4` worsened / `6` unchanged.

## Milestone Status

| README milestone | Status | Evidence |
| --- | --- | --- |
| Run a tiny single-GPU GRPO smoke test on 10 GSM8K problems | Done | Shape job `50951560` completed on one A40; 100-step job `50954114` completed and saved `checkpoint-100`. |
| Confirm reward, KL, and completion-length logs are produced | Partial | Completion logs and reward columns exist; notebook analysis loaded 400 rows for the 100-step run. KL is not yet clearly captured as a plotted per-step artifact. |
| Save and reload a GRPO checkpoint cleanly | Done | Resume job `50952454` resumed from `checkpoint-5` and completed to step 10. The 300-step run also resumed from `checkpoint-290` to `checkpoint-300` in job `51089453`. |
| Run a short scaled GRPO job and record memory, reward, KL, and wall-clock | Partial | 300-step / 50-example leak-penalty run completed via jobs `51062978` and `51089453`; wall-clock and reward behavior are present. Peak GPU memory and KL are still missing. |
| Serve base and GRPO-trained checkpoints with vLLM | Not started | No vLLM serving path has been exercised yet. `scripts/serve.py` is still a placeholder-level project file. |
| Quantize base and GRPO-trained checkpoints | Not started | `scripts/quantize.py` exists as a project slot, but no AWQ or W4 quantization result has been run or recorded. |
| Compare FP16-base vs FP16-GRPO vs W4-base vs W4-GRPO | Not started | Requires serving/quantization outputs that do not exist yet. |
| Publish results table, plots, and reproducibility notes | Partial | Notes and notebook scaffolding exist, but the actual comparison table for the research question does not. |

## What The Results Actually Show

The GRPO toolchain works end to end for the small model. CUDA is visible inside Slurm jobs, TRL `GRPOTrainer` runs, GSM8K rewards execute, checkpoints are written, and resume works.

The original 100-step checkpoint is a smoke artifact, not an improved model. The train-10 evaluation showed base accuracy `0.60`, trained accuracy `0.50`, delta `-0.10`, with `0` improved, `1` worsened, and `9` unchanged. That is not a reason to panic, but it is a reason not to scale model size yet.

The prompt-leak penalty was a reasonable next experiment. The 5-step check verified that the modified reward path executes, and the 300-step / 50-example run reached `checkpoint-300` after a timeout and resume. But eval quality got worse and eval prompt leakage stayed flat, so this checkpoint should be rejected as an improvement candidate.

## Gaps And Risks

Peak GPU memory has not been recorded. Without it, the project cannot make a defensible estimate for scaling to Qwen3-1.7B or larger jobs.

KL and entropy are not yet first-class metrics in the notebook. GRPO can appear to train while drifting in bad ways; KL is the main guardrail against that drift.

The evaluation set is tiny and not held out. Train-10 is a sanity check for the harness, not evidence of model improvement. A small test split eval is needed before making any model-quality claim.

The current generated outputs still show reasoning mistakes and occasional prompt-like continuation. The leak penalty helps a formatting failure mode, but it does not automatically improve math reasoning.

The quantization half of the research question has not started. There is no W4/AWQ comparison, no outlier diagnostic table, and no vLLM latency/throughput measurement yet.

The main strategic risk is over-investing in the GRPO smoke loop. The project needs enough GRPO quality to produce a meaningful checkpoint, but the final question is quantization behavior after post-training, not maximizing GSM8K accuracy.

## Recommended Next Steps

1. Inspect the four worsened examples from the 300-step leak-penalty eval.

   Rationale: accuracy and PPL both worsened, so the next move is diagnosis, not a longer run. Identify whether failures are arithmetic errors, format drift, excessive continuation, or parser mismatch.

   Command:

   ```bash
      python - <<'PY'
   import pandas as pd
   p="$PQS_ROOT/evals/gsm8k_compare_train10_leak300/paired_comparison.csv"
   df=pd.read_csv(p)
   print(df[df["change"]!="unchanged"].to_string(index=False))
   PY
   ```

2. Run only a small format-fix smoke after tightening prompt and generation length.

   Rationale: the current checkpoint is bad. The next run should test whether stricter stopping behavior fixes leakage without making math worse. Start with 5 steps, then at most 100 steps if the shape check is healthy.

   Example command:

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
     --export=ALL,MAX_STEPS=5,DATASET_LIMIT=10,OUTPUT_DIR=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_format_fix_5step \
     slurm/smoke_single_gpu.sbatch
   ```

3. Add explicit peak-memory and KL capture before any larger run.

   Rationale: memory and KL are required to decide whether scaling is safe. Without them, the next model-size jump would be guesswork.

   Likely files: `scripts/train_grpo_gsm8k.py`, `slurm/smoke_single_gpu.sbatch`, and the analysis notebook.

4. Freeze the 0.5B GRPO smoke decision.

   Rationale: after the 300-step eval and one held-out check, either keep the checkpoint as the first GRPO artifact or do one targeted reward/generation fix. Do not keep looping indefinitely.

5. Start the quantization/serving skeleton only after the eval harness is trustworthy.

   Rationale: the research question needs FP16-base, FP16-GRPO, W4-base, and W4-GRPO. Once the 0.5B checkpoint is good enough as a post-training artifact, the next value comes from quantization diagnostics, not more small reward tweaks.

## Readiness Call

The project is on track operationally but not yet on track scientifically. The training and evaluation harness is close to usable, but the evidence so far does not show a better trained model, and the quantization study has not begun.

The right posture is: reject the current 300-step checkpoint as an improvement candidate, run a small format-fix diagnostic, add missing memory/KL logging, and only then decide whether the 0.5B GRPO artifact is good enough to carry into the first FP16-vs-W4 comparison. Do not move to a bigger model until the 0.5B evidence is clean enough to justify the cost.
