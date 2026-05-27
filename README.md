# posttrain-quant-serve

Study repo for RL post-training a small Qwen model with GRPO, serving it with vLLM, quantizing the trained checkpoint, and measuring whether RL post-training changes quantization behavior.

## Research question

Does RL post-training change how cleanly a model quantizes?

Concretely, this repo will compare:

- Base model
- GRPO-trained model
- Quantized base model
- Quantized GRPO-trained model

Metrics will include perplexity or small eval accuracy, latency, throughput, memory use, and weight/outlier diagnostics.

## Scope

Target model:

- Day 0 smoke: Qwen2.5-0.5B-Instruct on GSM8K
- Scale target: Qwen3-1.7B or similar small reasoning model
- Stretch: larger multi-GPU GRPO only after the smoke path is stable

Core stack:

- TRL `GRPOTrainer` for RL post-training
- GSM8K answer checker for verifiable rewards
- PyTorch FSDP / Accelerate for later distributed scaling
- vLLM for serving and throughput/latency benchmarking
- AWQ or local quantization diagnostics for the quantization study
- Slurm for Michigan GPU cluster runs

## Milestones

- [ ] Run a tiny single-GPU GRPO smoke test on 10 GSM8K problems
- [ ] Confirm reward, KL, and completion-length logs are produced
- [ ] Save and reload a GRPO checkpoint cleanly
- [ ] Run a short scaled GRPO job and record memory, reward, KL, and wall-clock
- [ ] Serve base and GRPO-trained checkpoints with vLLM
- [ ] Quantize base and GRPO-trained checkpoints
- [ ] Compare FP16-base vs FP16-GRPO vs W4-base vs W4-GRPO
- [ ] Publish results table, plots, and reproducibility notes

## Layout

```text
configs/      Training, serving, and quantization configs
notes/        Study log, reading notes, and understanding checks
scripts/      Runnable training, serving, benchmark, and analysis scripts
slurm/        sbatch templates for cluster runs
src/          Reusable project code
results/      Curated result tables and plots
```

Large local artifacts such as datasets, checkpoints, raw logs, and model outputs are ignored by git.

## Day 0 Target

Do not touch larger models yet. First prove the smallest GRPO path works:

1. Create and verify the cluster Python/CUDA environment.
2. Download the smoke-test model and GSM8K data.
3. Read enough GRPO to understand rollouts, rewards, advantages, and KL.
4. Run a tiny single-GPU Qwen2.5-0.5B-Instruct GRPO run for 5 then 100 steps.
5. Confirm the run saves a checkpoint and can reload it.

Start with `notes/project-overview.md` for the plain-English explanation of the project and learning path.
