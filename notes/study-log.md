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
