# posttrain-quant-serve

Study repo for fine-tuning a 7B model with FSDP, serving it with vLLM, quantizing the fine-tuned checkpoint, and measuring whether post-training changes quantizability.

## Research question

Does supervised fine-tuning change how cleanly a model quantizes?

Concretely, this repo will compare:

- Base model
- FSDP fine-tuned model
- Fine-tuned plus quantized model

Metrics will include perplexity or small eval accuracy, latency, throughput, memory use, and weight/outlier diagnostics.

## Scope

Target model:

- Primary: Qwen2.5-7B
- Fallback: smaller Qwen smoke-test model, then FSDP + LoRA if full fine-tuning slips

Core stack:

- PyTorch FSDP or torchtune for distributed fine-tuning
- vLLM for serving and throughput/latency benchmarking
- AWQ, GPTQ, or local quantization diagnostics for quantization study
- Slurm for Michigan GPU cluster runs

## Milestones

- [ ] Run a tiny single-GPU fine-tuning smoke test end-to-end
- [ ] Run short 4-GPU FSDP job and save/reload checkpoint
- [ ] Run real fine-tune and record memory, throughput, and wall-clock
- [ ] Serve base and fine-tuned checkpoints with vLLM
- [ ] Quantize fine-tuned checkpoint and serve quantized model
- [ ] Compare base vs fine-tuned vs fine-tuned+quantized
- [ ] Publish results table, plots, and reproducibility notes

## Layout

```text
configs/      Training, serving, and quantization configs
docs/         Study log and paper notes
scripts/      Runnable training, serving, benchmark, and analysis scripts
src/          Reusable project code
results/      Curated result tables and plots
```

Large local artifacts such as datasets, checkpoints, raw logs, and model outputs are ignored by git.
