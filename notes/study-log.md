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

- Repo should live under `/scratch/huterer_root/huterer0/jiamingp/pqs/repos/posttrain-quant-serve`,
  not `$HOME`, because `$HOME` filled during package installs.
- Environment path:
  `/scratch/huterer_root/huterer0/jiamingp/pqs/envs/posttrain-quant-serve`.
- HF cache path:
  `/scratch/huterer_root/huterer0/jiamingp/pqs/hf_cache`.
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
  `/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/smoke_qwen2_5_0_5b_grpo/checkpoint-5`.

- Resume check: job `50952454`, completed in `00:03:58`, exit code `0:0`; resumed from
  `checkpoint-5` and continued to step 10. Nonfatal warning observed:
  missing checkpoint key `lm_head.weight`.

- Parser fix check: job `50952893`, completed in `00:03:48`, exit code `0:0`; output path:
  `/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/smoke_qwen2_5_0_5b_grpo_parser_fix`.

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
  --export=ALL,MAX_STEPS=100,DATASET_LIMIT=10,OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/smoke_qwen2_5_0_5b_grpo_100step \
  slurm/smoke_single_gpu.sbatch
```

- 100-step result: job `50954114`, completed in `00:18:46`, exit code `0:0`.
- Training progress: reached `100/100`; logged runtime `971.4` seconds and `0.103` steps/second.
- Checkpoint path:
  `/scratch/huterer_root/huterer0/jiamingp/pqs/ckpts/smoke_qwen2_5_0_5b_grpo_100step/checkpoint-100`.
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
