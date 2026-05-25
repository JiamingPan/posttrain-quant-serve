# Day 0 Checklist

Goal: understand FSDP well enough to start and run a tiny smoke fine-tune. Do not touch Qwen2.5-7B yet.

## Tasks

- [ ] Read the PyTorch FSDP getting-started tutorial.
- [ ] Read ZeRO paper section 3.
- [ ] Fill gaps in `notes/fsdp.md`.
- [ ] Create the cluster environment.
- [ ] Run `scripts/cluster_check.py` on a GPU node.
- [ ] Log in with `huggingface-cli login` on the cluster.
- [ ] Download Qwen2.5-0.5B for the smoke test.
- [ ] Generate or copy the torchtune smoke-test config.
- [ ] Launch a 50-100 step single-GPU smoke fine-tune.
- [ ] Record exact command, GPU, peak memory, loss movement, checkpoint path, and reload result in `notes/study-log.md`.

## Stop Condition

Day 0 is done when the smoke run has a checkpoint that can be reloaded. If setup breaks before that, record the exact failure and make that the first Day 1 fix.
