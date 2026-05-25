# Project Overview

This is the "start here" document. Its job is to explain what this repo is doing, why each part exists, and what you need to understand before scaling to the 7B run.

## One-Sentence Version

Fine-tune a language model, serve it, quantize it, serve it again, and measure whether fine-tuning made the model easier or harder to quantize.

## The Real Research Question

You already have experience with quantization and outlier analysis. The new question is:

> Does supervised fine-tuning change the weight or activation structure in a way that changes quantization behavior?

That means this is not just a tutorial repo. The tutorial part is the pipeline:

1. Fine-tune a model.
2. Save and reload the checkpoint.
3. Serve the model behind an inference engine.
4. Quantize the checkpoint.
5. Evaluate speed, memory, and quality.

The research part is the comparison:

| Model Variant | What It Answers |
| --- | --- |
| Base model | What is the original quality, latency, throughput, memory, and outlier structure? |
| Fine-tuned model | What changed after training? |
| Fine-tuned + quantized model | Did quantization preserve quality and improve serving efficiency? |

The final result should be a table and a few plots that let you say something concrete, such as: "After fine-tuning, layer X developed larger outliers, and 4-bit quantization degraded more there," or "Fine-tuning did not meaningfully change quantizability under this setup."

## Why Day 0 Exists

Day 0 is not about training the real 7B model. Day 0 is about proving that the basic machinery works on a small model.

The reason is simple: a 7B FSDP run has many possible failure modes:

- CUDA or PyTorch version mismatch
- wrong torchtune config
- dataset formatting issue
- Hugging Face auth/cache issue
- Slurm launch issue
- out-of-memory
- checkpoint format confusion
- bad resume/reload path

If you start directly with the 7B model, every one of those failures is expensive and confusing. A smoke test isolates the boring failures first.

Day 0 success means:

1. The cluster can see a GPU.
2. PyTorch can allocate a CUDA tensor.
3. Hugging Face model download works.
4. torchtune can read a config and dataset.
5. A tiny fine-tune starts and logs loss.
6. A checkpoint is written.
7. That checkpoint can be reloaded.

Once those are true, the remaining work is real FSDP scaling work rather than basic plumbing.

## What The Pipeline Will Look Like

```text
Qwen2.5 base model
        |
        v
single-GPU smoke fine-tune on small Qwen
        |
        v
multi-GPU FSDP fine-tune on Qwen2.5-7B
        |
        v
fine-tuned checkpoint
        |
        +--------------------------+
        |                          |
        v                          v
serve with vLLM             quantize checkpoint
        |                          |
        v                          v
benchmark base/fine-tuned   serve quantized model
        |                          |
        +------------+-------------+
                     v
          compare quality, latency,
          throughput, memory, and
          outlier/weight structure
```

## What Each Tool Is For

| Tool | Why It Exists In This Project |
| --- | --- |
| PyTorch | The underlying training framework. FSDP is a PyTorch distributed training feature. |
| FSDP | Lets a full-parameter fine-tune fit by sharding model state across GPUs. |
| torchtune | Provides maintained LLM fine-tuning recipes and configs so you do not hand-roll the training loop first. |
| Slurm | Schedules jobs on the Michigan GPU cluster. |
| Hugging Face Hub | Stores model weights/tokenizers and handles model downloads. |
| vLLM | Serves base and fine-tuned models efficiently for latency/throughput measurement. |
| AWQ/GPTQ/local quant code | Produces the quantized checkpoint and helps measure quantization behavior. |

## The Core Concept: Why FSDP Matters

Full fine-tuning means updating all model weights. For a 7B or 8B model, the model weights are not the only memory cost.

With Adam, training stores:

- model parameters
- gradients
- fp32 master weights
- Adam first moment
- Adam second moment
- activations
- temporary buffers

The optimizer state is usually the largest piece. So the problem is not just "the model is 16 GB in bf16." The training state can be much larger than the model itself.

DDP does not solve this when the training state is too large because DDP replicates the whole training state on every GPU. DDP gives you more throughput, not more per-GPU memory headroom for the model state.

FSDP solves a different problem: it shards the training state across GPUs. If you use four GPUs, each GPU owns roughly one fourth of the parameters, gradients, and optimizer state. During computation, FSDP temporarily gathers the full parameters for the layer or block it is working on, computes, then frees what it does not own.

That is the trade:

- You save memory.
- You pay more communication.

This is why the first serious training milestone is not "high quality model." It is:

> Can I run a short FSDP job, save a checkpoint, reload it, and explain what got sharded?

## What You Need To Learn

### Level 1: Before Running Anything

You should be able to explain these in plain language:

- What is full-parameter fine-tuning?
- What is a checkpoint?
- What does an optimizer store?
- Why does Adam use much more memory than just the model weights?
- What does DDP replicate?
- What does FSDP shard?
- What is the difference between ZeRO-1, ZeRO-2, and ZeRO-3?
- Why does activation checkpointing save memory but cost compute?

Use `notes/fsdp.md` for this.

### Level 2: Before Scaling To 7B

You should be able to answer:

- What model was used for the smoke test?
- What dataset was used?
- What exact command launched training?
- Where did the checkpoint get saved?
- Did loss move at all?
- What was peak GPU memory?
- Can the checkpoint reload?
- Which torchtune config fields matter most?

Use `notes/study-log.md` for this.

### Level 3: Before Serving

You should understand:

- What vLLM does differently from a normal Hugging Face `generate` loop.
- What the KV cache is.
- Why batching matters for throughput.
- What p50 and p99 latency mean.
- Why throughput and latency can move in opposite directions.

Serving is not Day 0. This comes after a fine-tuned checkpoint exists.

### Level 4: Before Quantization Analysis

You should understand:

- What weight-only quantization is.
- What activation-aware quantization means.
- Why outlier channels can hurt low-bit quantization.
- Which metric you will use for quality.
- Which metrics you will use for serving efficiency.

This is where the project connects back to your quantization strength.

## What I Added To The Repo

| Path | What It Is |
| --- | --- |
| `README.md` | Public-facing project summary and milestones. |
| `notes/project-overview.md` | This guide. Read this first when confused. |
| `notes/day0.md` | Checklist for the first smoke-test day. |
| `notes/fsdp.md` | FSDP and ZeRO mental model notes. |
| `notes/reading.md` | Reading order. |
| `notes/cluster-setup.md` | Commands to run on the cluster. |
| `notes/study-log.md` | Running log for commands, failures, fixes, and measurements. |
| `requirements.txt` | Training/smoke-test Python dependencies, excluding PyTorch. |
| `requirements-serving.txt` | Later serving/quantization dependencies. |
| `scripts/cluster_check.py` | Verifies PyTorch can see CUDA on a GPU node. |
| `scripts/download_models.sh` | Downloads Qwen2.5-0.5B and Qwen2.5-7B through Hugging Face. |
| `scripts/prepare_torchtune_config.sh` | Copies an official torchtune config into `configs/`. |
| `scripts/finetune.sh` | Starts the torchtune single-GPU smoke fine-tune. |
| `slurm/smoke_single_gpu.sbatch` | Slurm job template for the smoke test. |
| `scripts/serve.py` | Placeholder for the later vLLM serving phase. |
| `scripts/benchmark.py` | Placeholder for later latency/throughput/eval measurement. |
| `scripts/quantize.py` | Placeholder for later quantization work. |

## What You Should Do Next

Run this on the cluster, not on your laptop:

```bash
git clone git@github.com:JiamingPan/posttrain-quant-serve.git
cd posttrain-quant-serve
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the cluster-compatible PyTorch build first. The exact command depends on the cluster CUDA module. After PyTorch:

```bash
python -m pip install -r requirements.txt
```

Then request or enter a GPU node and run:

```bash
python scripts/cluster_check.py
```

If CUDA works, log into Hugging Face:

```bash
huggingface-cli login
bash scripts/download_models.sh
```

Then prepare the torchtune config:

```bash
bash scripts/prepare_torchtune_config.sh
```

Then launch the smoke run:

```bash
CONFIG=configs/smoke_qwen2_5_0_5b.yaml MAX_STEPS=100 bash scripts/finetune.sh
```

Or through Slurm:

```bash
sbatch slurm/smoke_single_gpu.sbatch
```

## How To Know You Are Making Progress

You are not trying to get a good model yet. You are trying to remove uncertainty.

Good Day 0 outcomes:

- `cluster_check.py` prints the GPU name and CUDA availability.
- Model download succeeds.
- torchtune config is copied.
- Fine-tuning starts.
- Loss logs for at least a few steps.
- A checkpoint appears.
- You can describe what failed if it does not work.

Bad Day 0 pattern:

- Jumping to the 7B model before the small model runs.
- Spending hours choosing a dataset.
- Installing vLLM/AWQ before the training path works.
- Reading serving papers before you can explain FSDP.

## The Explanation You Should Eventually Be Able To Give In An Interview

"I built a pipeline that fine-tunes Qwen2.5 with PyTorch FSDP, then serves and quantizes the fine-tuned checkpoint. I started with a single-GPU Qwen2.5-0.5B smoke test to validate the environment, data path, checkpointing, and torchtune config before scaling. For the 7B run, FSDP was necessary because DDP would replicate the full parameters, gradients, and Adam states on every GPU, while FSDP shards those states across ranks and all-gathers parameters just in time for each wrapped transformer block. After training, I compared base, fine-tuned, and fine-tuned-plus-quantized models on quality, latency, throughput, memory, and outlier structure to test whether fine-tuning changed quantizability."

If you can say that and defend each sentence, you understand the project.
