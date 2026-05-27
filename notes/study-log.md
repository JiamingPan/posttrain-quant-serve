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

Day 0 GRPO smoke status:

```text
# Fill in after smoke run completes:
# - shape-check command:
# - smoke command:
# - reload command:
# - GPU type:
# - peak GPU memory:
# - steps:
# - reward start -> end:
# - KL start -> end:
# - checkpoint path:
# - reload check:
# - errors / fixes:
```
