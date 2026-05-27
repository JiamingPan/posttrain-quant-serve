# Project Overview

This is the "start here" document. The repo is now centered on GRPO RL post-training, not SFT.

## One-Sentence Version

Run GRPO on GSM8K with a small Qwen model, save the checkpoint, serve and quantize it later, then measure whether RL post-training changed quantization behavior.

## The Research Question

The project question is:

> Does RL post-training change the weight/outlier structure in a way that changes quantization robustness?

The comparison will be:

| Variant | Purpose |
| --- | --- |
| FP16 base | Original model quality, latency, memory, and outlier structure |
| FP16 GRPO | What changed after RL post-training |
| W4 base | Quantization behavior before RL |
| W4 GRPO | Quantization behavior after RL |

The end product should let you say something evidence-based, such as:

- GRPO improved GSM8K reward but introduced larger layer outliers that made W4 quantization worse.
- GRPO improved GSM8K reward and did not materially change AWQ scale/outlier behavior.
- GRPO improved reward but the quantized trained model lost more accuracy than the quantized base.

## Why GRPO Instead Of SFT

SFT learns from labeled target completions. It is useful, but it is not the post-training story most aligned with your updated plan.

GRPO is RL post-training:

1. Sample several completions for the same prompt.
2. Score each completion with a reward.
3. Normalize rewards within the group.
4. Increase probability of better completions.
5. Penalize moving too far from the reference model with KL.

For GSM8K, the reward is verifiable: extract the final numeric answer and compare it to the ground truth. No reward model is needed.

## Day 0 Goal

Day 0 is only a smoke test:

```text
Qwen2.5-0.5B-Instruct + 10 GSM8K problems + 1 GPU + 5 then 100 GRPO steps
```

Success means:

- CUDA works inside a Slurm GPU job.
- TRL `GRPOTrainer` can load the model and GSM8K.
- The reward function runs.
- Reward/KL/completion-length logs appear.
- A checkpoint is saved.
- A reload/resume check works.

It does **not** mean the model is good.

## Pipeline

```text
Qwen2.5-0.5B-Instruct
        |
        v
GRPO smoke on 10 GSM8K problems
        |
        v
checkpoint + reload check
        |
        v
larger GRPO run once smoke path is stable
        |
        +-------------------------+
        |                         |
        v                         v
serve with vLLM             quantize with AWQ
        |                         |
        +-----------+-------------+
                    v
compare FP16-base, FP16-GRPO,
W4-base, and W4-GRPO
```

## Vocabulary You Need

| Term | Meaning |
| --- | --- |
| Policy model | The model being updated by RL. |
| Reference model | Frozen copy of the starting model used for KL regularization. |
| Rollout | A generated completion sampled from the policy model. |
| Reward | Scalar score for a rollout; for GSM8K, final answer correct = 1, wrong = 0. |
| Group | Multiple completions sampled for the same prompt. |
| Advantage | Relative score saying which completions in the group were better than the group average. |
| KL penalty | Cost for moving too far away from the reference model. |
| Entropy | Roughly, how much randomness/diversity the policy keeps. |
| Checkpoint | Saved model/trainer state that can be resumed or loaded. |

## What Each Tool Is For

| Tool | Role |
| --- | --- |
| PyTorch | Low-level tensor/training framework. |
| TRL | Hugging Face post-training library; provides `GRPOTrainer`. |
| Accelerate | Launches training in a way that can scale beyond one GPU later. |
| GSM8K | Math dataset with verifiable numeric answers. |
| Slurm | Great Lakes job scheduler. |
| vLLM | Later serving/throughput benchmark tool. |
| AWQ | Later W4 quantization path. |

## What Was Added For GRPO

| Path | Purpose |
| --- | --- |
| `scripts/train_grpo_gsm8k.py` | Main GRPO smoke trainer with GSM8K exact-match reward. |
| `scripts/run_grpo_smoke.sh` | Small shell wrapper for shape/real/reload runs. |
| `slurm/smoke_single_gpu.sbatch` | Great Lakes single-GPU GRPO smoke job. |
| `notes/grpo.md` | Concept notes and debugging guide. |
| `notes/day0.md` | Current Day 0 checklist. |

## Immediate Great Lakes Flow

Run from the scratch repo:

```bash
cd /scratch/huterer_root/huterer0/jiamingp/pqs/repos/posttrain-quant-serve
git pull
```

Use the existing scratch environment:

```bash
module purge
module load python/3.11.5
module load cuda/12.8.1
source /scratch/huterer_root/huterer0/jiamingp/pqs/envs/posttrain-quant-serve/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/huterer_root/huterer0/jiamingp/pqs/hf_cache
```

Then submit the shape check:

```bash
sbatch --export=ALL,MAX_STEPS=5,DATASET_LIMIT=10 slurm/smoke_single_gpu.sbatch
```

If that succeeds, submit the real smoke:

```bash
sbatch --export=ALL,MAX_STEPS=100,DATASET_LIMIT=10 slurm/smoke_single_gpu.sbatch
```

Do not run larger models until this path is clean.
