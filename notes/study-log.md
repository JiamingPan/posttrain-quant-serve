# Study Log

## 2026-05-24

Started the repo and project scaffold.

Immediate next block:

1. Read the PyTorch FSDP tutorial and skim ZeRO.
2. Pick the smallest smoke-test model and dataset.
3. Create a minimal single-GPU fine-tune command before touching Qwen2.5-7B.
4. Record every working command, failure mode, GPU memory number, and throughput number here.

Decision log:

- Repo name: `posttrain-quant-serve`
- Repo visibility: public
- Main research question: whether fine-tuning shifts quantizability and outlier structure

## 2026-05-25

Day 0 kickoff scope:

- Understand the FSDP memory model before touching 7B.
- Use torchtune as the fine-tuning entry point.
- Use Qwen2.5-0.5B for the first single-GPU smoke test.
- Keep Qwen2.5-7B as the main target once the smoke path works.
- Use Alpaca-style instruction data first; do not spend time shopping for datasets.

Day 0 success condition:

- `scripts/cluster_check.py` reports CUDA and the expected GPU.
- A torchtune single-GPU run starts, consumes data, logs loss, and writes a checkpoint.
- The exact command, config path, GPU type, memory usage, and failure/fix notes are recorded here.

## 2026-05-26

Project pivot: SFT fine-tuning path replaced by GRPO RL post-training.

Reason:

- Updated five-month plan prioritizes RL/post-training.
- GSM8K gives verifiable rewards, so the smoke test can use rule-based reward checking rather
  than a learned reward model.
- The research question is now: does RL post-training change quantization behavior?

Great Lakes environment notes:

- Repo should live under `$PQS_ROOT/repos/posttrain-quant-serve`,
  not `$HOME`, because `$HOME` filled during package installs.
- Environment path:
  `$PQS_ROOT/envs/posttrain-quant-serve`.
- HF cache path:
  `$PQS_ROOT/hf_cache`.
- Working package stack found during setup:
  `torch==2.8.0+cu128`, `torchao==0.14.1`, `kagglehub==1.0.0`,
  `kagglesdk==0.1.24`.

Day 0 / Day 1 GRPO smoke result:

- Shape-check job:

```bash
sbatch \
  --job-name=pqs-grpo-shape \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:45:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=5,DATASET_LIMIT=10 \
  slurm/smoke_single_gpu.sbatch
```

- Shape-check result: job `50951560`, completed in `00:03:57`, exit code `0:0`.
- GPU from shape-check log: NVIDIA A40, 44.4 GiB.
- Shape-check checkpoint path:
  `$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo/checkpoint-5`.

- Resume check: job `50952454`, completed in `00:03:58`, exit code `0:0`; resumed from
  `checkpoint-5` and continued to step 10. Nonfatal warning observed:
  missing checkpoint key `lm_head.weight`.

- Parser fix check: job `50952893`, completed in `00:03:48`, exit code `0:0`; output path:
  `$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo_parser_fix`.

- 100-step smoke command:

```bash
sbatch \
  --job-name=pqs-grpo-100 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:40:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=100,DATASET_LIMIT=10,OUTPUT_DIR=$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo_100step \
  slurm/smoke_single_gpu.sbatch
```

- 100-step result: job `50954114`, completed in `00:18:46`, exit code `0:0`.
- Training progress: reached `100/100`; logged runtime `971.4` seconds and `0.103` steps/second.
- Checkpoint path:
  `$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo_100step/checkpoint-100`.
- Other saved checkpoint: `checkpoint-95`.
- Peak GPU memory: not captured in the pasted log; add explicit memory logging before using this as
  a scaling estimate.
- Reward/KL status: the two printed completions at step 100 had `gsm8k_exact_match_reward=0.00`
  and `Advantage=-0.50`. This does not prove every completion had zero reward because TRL only
  prints two completions. Inspect the saved `completions/` files before deciding whether the reward
  signal is broken.
- Warnings:
  unauthenticated Hugging Face requests; Great Lakes kernel below Accelerate's recommended version;
  tokenizer PAD/BOS/EOS alignment warning. None stopped the run.
- Current conclusion: the GRPO toolchain works end-to-end for Qwen2.5-0.5B-Instruct on one A40.
  The next blocker is understanding reward/completion quality, not cluster setup.

## 2026-05-28

Completion analysis for job `50954114`:

- Notebook loaded `400` completion rows from:
  `$PQS_ROOT/ckpts/smoke_qwen2_5_0_5b_grpo_100step/completions`.
- Columns found: `step`, `prompt`, `completion`, `gsm8k_exact_match_reward`, `advantage`,
  `source_file`.
- Reward signal:
  - `gsm8k_exact_match_reward_mean = 0.235`
  - max reward `1.0`
  - reward std about `0.4245`
- Answer extraction rate: `1.0`.
- Prompt leakage rate: `0.2025`.
- Interpretation: the smoke run is valid and has a live reward signal, but output cleanliness needs
  work before scaling. `checkpoint-100` should now be evaluated against the base model.

Prompt-leak reward penalty:

- Added a small penalty after the parsed answer:

```text
clean correct: 1.0
leaky correct: 0.75
clean wrong: 0.0
leaky wrong: -0.25
```

- Parser test passed locally with the new reward behavior.

5-step leak-penalty check:

```bash
sbatch --job-name=pqs-grpo-leakfix \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:20:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=5,DATASET_LIMIT=10,OUTPUT_DIR=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_5step \
  slurm/smoke_single_gpu.sbatch
```

- Job `51051394`, `pqs-grpo-leakfix`, completed in `00:08:42`, exit code `0:0`.
- Checkpoint path:
  `$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_5step`.
- Log tail showed sampled completions still rambling, which is expected for only 5 steps. This run
  verifies the modified reward path executes; it does not prove leakage has improved.

Next:

1. Run `slurm/eval_gsm8k_compare.sbatch` for base vs `checkpoint-100`.
2. If evaluation is sane, run a longer leak-penalty follow-up and compare prompt leakage before/after.

Base-vs-trained sanity evaluation:

```bash
sbatch --account=cavestru0 --time=00:30:00 \
  --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10,EVAL_OUTPUT_DIR=$PQS_ROOT/evals/gsm8k_compare_train10_100step \
  slurm/eval_gsm8k_compare.sbatch
```

- Job `51062503`, `pqs-gsm8k-eval`, completed in `00:07:23`, exit code `0:0`.
- Output path:
  `$PQS_ROOT/evals/gsm8k_compare_train10_100step`.
- Base `Qwen/Qwen2.5-0.5B-Instruct` on first 10 GSM8K train examples:
  - accuracy `0.6`
  - correct `6/10`
  - parse rate `1.0`
  - prompt leak rate `0.4`
- Trained `checkpoint-100`:
  - accuracy `0.5`
  - correct `5/10`
  - parse rate `1.0`
  - prompt leak rate `0.4`
- Delta accuracy: `-0.1`.
- Paired comparison: `0` improved, `1` worsened, `9` unchanged.

Interpretation:

- This is a negative result for the trained checkpoint, but only by one question on a tiny 10-example
  sanity set.
- The 100-step checkpoint should be treated as a pipeline smoke artifact, not as a useful improved
  model.
- Do not scale model size from this checkpoint. First inspect the worsened example and run a longer
  leak-penalty experiment or improve reward/prompt settings.

Longer leak-penalty follow-up:

```bash
sbatch --job-name=pqs-grpo-leak300 \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=01:00:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=300,DATASET_LIMIT=50,OUTPUT_DIR=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k \
  slurm/smoke_single_gpu.sbatch
```

- Job `51062978`, `pqs-grpo-leak300`, timed out at `01:00:15` near `294/300`.
- It saved `checkpoint-285` and `checkpoint-290`.
- Resume command:

```bash
sbatch --job-name=pqs-grpo-leak300-resume \
  --account=cavestru0 \
  --partition=spgpu \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=32G \
  --time=00:30:00 \
  --output=logs/%x-%j.out \
  --error=logs/%x-%j.err \
  --export=ALL,MAX_STEPS=300,DATASET_LIMIT=50,OUTPUT_DIR=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k,RESUME_FROM_CHECKPOINT=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k/checkpoint-290 \
  slurm/smoke_single_gpu.sbatch
```

- Job `51089453`, `pqs-grpo-leak300-resume`, completed in `00:06:53`, exit code `0:0`.
- Final checkpoint exists:
  `$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k/checkpoint-300`.
- Step 300 log showed the leak penalty is active: a leaky wrong completion received reward `-0.25`.
- The notebook now defaults to this 300-step run and compares it against the original 100-step run.

Next:

1. Run `notebooks/grpo_smoke_analysis.ipynb` on Great Lakes.
2. Compare `prompt_leak_rate` for `baseline_100step` vs `leak_penalty_300step`.
3. If leakage improves, run base-vs-trained eval for `checkpoint-300`.


## 2026-05-29

Day 2 closeout and Day 3 start.

300-step leak-penalty base-vs-trained eval:

```bash
sbatch --account=cavestru0 --time=00:45:00 \
  --export=ALL,EVAL_SPLIT=train,EVAL_LIMIT=10,TRAINED_MODEL=$PQS_ROOT/ckpts/qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k,EVAL_OUTPUT_DIR=$PQS_ROOT/evals/gsm8k_compare_train10_leak300 \
  slurm/eval_gsm8k_compare.sbatch
```

- Job `51091160`, `pqs-gsm8k-eval`, completed in `00:03:56`, exit code `0:0`.
- Base `Qwen/Qwen2.5-0.5B-Instruct` on first 10 GSM8K train examples:
  - accuracy `0.6` (`6/10`)
  - parse rate `1.0`
  - prompt leak rate `0.5`
  - reference PPL `1.8783`
- Trained `qwen2_5_0_5b_grpo_leak_penalty_300step_50gsm8k`:
  - accuracy `0.2` (`2/10`)
  - parse rate `1.0`
  - prompt leak rate `0.5`
  - reference PPL `1.9337`
- Delta accuracy: `-0.4`.
- Reference PPL ratio: `1.0295`, worse for the trained checkpoint.
- Paired comparison: `0` improved, `4` worsened, `6` unchanged.

Interpretation:

- The 300-step leak-penalty checkpoint is not an improved model.
- Training-completion leakage improved in the notebook, but deterministic eval leakage did not improve
  against the original base model.
- Do not scale this checkpoint and do not start quantization from it except as a known-bad control.
- Day 3 should diagnose worsened examples and tighten prompt/generation stopping behavior.

Day 3 repo changes:

- Tightened the train/eval prompt to require one final `#### <answer>` line and then stop.
- Reduced default training `max_completion_length` from `256` to `128`.
- Reduced default eval `max_new_tokens` from `192` to `96`.
- Added `notes/day3.md` with the diagnosis checklist and small format-fix smoke command.
