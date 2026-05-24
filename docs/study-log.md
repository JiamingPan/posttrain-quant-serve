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
- Initial visibility: private until the project has results and a cleaned README
- Main research question: whether fine-tuning shifts quantizability and outlier structure
