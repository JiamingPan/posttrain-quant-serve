# Day 6 - Notebook Interpretation Cleanup

Day 6 was not another GPU-run day. The main work was tightening the interpretation
of the Day 5 notebook after reviewing the clean 1.5B data1000 matrix.

## Starting Point

The clean Day 5 result remains:

| Label | Accuracy | Correct |
| --- | ---: | ---: |
| `base_fp16` | 0.68 | 68/100 |
| `g8_dr100_fp16` | 0.72 | 72/100 |
| `base_w4` | 0.65 | 65/100 |
| `g8_dr100_w4` | 0.68 | 68/100 |
| `base_awq` | 0.58 | 58/100 |
| `g8_dr100_awq` | 0.67 | 67/100 |

Deltas:

- `delta_fp16 = +0.04`
- `delta_w4 = +0.03`
- `delta_awq = +0.09`
- `gain_survival_w4 = -0.01`
- `gain_survival_awq = +0.05`

Stopping metrics were clean, so this matrix is still the active result.

## Clarification 1 - bnb-W4 vs AWQ

In this test100 slice, bnb-NF4 W4 has higher absolute accuracy than AWQ:

| Scheme | Base Accuracy | GRPO Accuracy | GRPO Delta |
| --- | ---: | ---: | ---: |
| FP16 | 0.68 | 0.72 | +0.04 |
| bnb-NF4 W4 | 0.65 | 0.68 | +0.03 |
| AWQ W4G128 | 0.58 | 0.67 | +0.09 |

This does not mean "bnb is globally better than AWQ." It means this exact
implementation, calibration set, checkpoint pair, and 100-example slice produced
higher absolute bnb-W4 scores. AWQ is calibration-aware, but it is not guaranteed
to beat bnb-NF4 on every downstream exact-match eval. The calibration text,
group size, AutoAWQ implementation details, and finite-sample noise all matter.

For the research question, the more important quantity is not the absolute
bnb-vs-AWQ ranking. The important quantity is within-scheme base-vs-GRPO:

- bnb-W4: GRPO still beats base by `+0.03`, close to the FP16 `+0.04`.
- AWQ: GRPO beats base by `+0.09`; AWQ hurt the base checkpoint more than the
  GRPO checkpoint on this slice.

So the honest claim is: the GRPO gain survived both W4 schemes tested here. Do
not write "AWQ is worse than bnb" as a general conclusion.

## Clarification 2 - Weight Metrics vs Behavior

The old notebook sentence was too strong:

> no global metric moved by more than 5%. GRPO did not obviously reshape the
> weight distribution at this scale.

That can be read as "GRPO did not really change the weights," which is wrong.
GRPO must change weights to change model behavior.

The correct interpretation is narrower:

> The coarse global quantization metrics did not show a large shift. That does
> not mean the weights were unchanged. It means the behaviorally useful GRPO
> update was not visible as a large change in marginal weight scale, outlier
> fraction, or simple W4 reconstruction-error summaries.

Accuracy can improve through small coordinated directional changes across many
weights, changes in high-leverage attention/MLP paths, or logit shifts on the
answer tokens. Those changes can matter functionally while leaving global
histograms, max-abs summaries, outlier fractions, and W4 proxy MSE nearly
unchanged.

Also, the existing 1.5B weight diagnostic was run on the earlier `g8_dr100`
checkpoint, not the final data1000 checkpoint. It supports the broad observation
that this GRPO recipe family did not obviously create global quantization
pathology, but a direct data1000 weight diagnostic would be cleaner for the
final writeup.

## Day 6 Status

Day 6 is done as an interpretation/documentation cleanup day:

- notebook wording clarified for bnb-W4 vs AWQ;
- notebook wording clarified for global weight diagnostics;
- no new GPU jobs are required for the main Day 5 result.

Optional future cleanup:

- run direct weight diagnostics on
  `$PQS_ROOT/ckpts/qwen2_5_1_5b_grpo_data1000_chat`
  if the final writeup needs an exact data1000 weight-space claim.

## Project Scope Decision

The GRPO-to-quantization research question is done: we have the trained 1.5B
checkpoint, the clean held-out FP16 eval, the AWQ checkpoints, and the six-row
FP16/bnb-W4/AWQ matrix. vLLM serving metrics are useful for the broader
`posttrain-quant-serve` portfolio story, but they are not required to support the
research claim. Treat vLLM as a packaging/stretch task, not as a blocker for the
Day 5 result.

## Data Counts and Epochs

Keep these counts separate. They answer different questions:

- training used `DATASET_LIMIT=1000` GSM8K train prompts;
- held-out accuracy used `EVAL_SPLIT=test` and `EVAL_LIMIT=100`;
- serving benchmark uses `NUM_PROMPTS=16` GSM8K-style prompts for speed and
  memory only;
- serving benchmark uses `MAX_NEW_TOKENS=128` as the generation-length cap, not
  as a data-count setting.

The `100` in the accuracy result does not mean the GRPO model was trained on
100 prompts. It means the held-out test evaluation was limited to 100 examples
for a controlled, cheap comparison. GSM8K has more test examples, but the active
matrix is the test100 slice because every row uses the same slice and the same
clean stopping setup.

The final GRPO checkpoint is:

```text
$PQS_ROOT/ckpts/qwen2_5_1_5b_grpo_data1000_chat
```

It used the 1.5B Qwen2.5-Instruct base model, GRPO post-training, 1000 GSM8K
train prompts, `MAX_STEPS=250`, batch size 1, and gradient accumulation 8. That
means one optimizer step effectively uses 8 prompts. Across 250 steps, the run
sees about `250 * 8 = 2000` prompt exposures. With a 1000-prompt training slice,
that is roughly 2 epochs.

"2 epochs over 1000 prompts" means the optimizer got about two passes worth of
training signal over that 1000-prompt slice. It does not mean the model memorized
only 100 prompts, and it does not mean every prompt appeared exactly twice in a
simple fixed order. It is the practical exposure count: about 2000 training
prompt uses divided by 1000 unique training prompts.

## What Is Being Quantized

There are two separate axes in the final matrix.

The training axis is:

- `base`: the original `Qwen2.5-1.5B-Instruct` checkpoint;
- `GRPO`: the same architecture after GRPO post-training on 1000 GSM8K train
  prompts.

The quantization axis is:

- `fp16`: dense model loading, using 16-bit floating-point weights/compute;
- `bnb-W4`: load-time bitsandbytes NF4 4-bit quantization;
- `AWQ W4G128`: saved AutoAWQ 4-bit, group-size-128 checkpoint.

So the final matrix is not six unrelated models. It is two model states crossed
with three precision/quantization settings:

| Label | Meaning |
| --- | --- |
| `base_fp16` | original dense Qwen2.5-1.5B-Instruct |
| `g8_dr100_fp16` | GRPO-trained dense checkpoint |
| `base_w4` | original base checkpoint loaded with bnb NF4 W4 |
| `g8_dr100_w4` | GRPO checkpoint loaded with bnb NF4 W4 |
| `base_awq` | original base checkpoint converted to saved AWQ W4G128 |
| `g8_dr100_awq` | GRPO checkpoint converted to saved AWQ W4G128 |

The object being quantized is the model checkpoint's weights, especially the
large transformer linear layers. For Qwen-style blocks, the important target
layers are the attention projections and MLP projections:

- attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`;
- MLP: `gate_proj`, `up_proj`, `down_proj`.

These matrices dominate the parameter count and are what W4 quantization is
meant to compress. The quantization is not applied to the GSM8K prompts, answer
strings, reward function, tokenizer, parser, or evaluation metric. It also does
not mean every byte used during inference becomes 4-bit. Runtime memory still
includes KV cache, activations, temporary CUDA/vLLM buffers, scheduler overhead,
and metadata.

The two W4 methods in this repo differ operationally:

- `bnb-W4` is load-time quantization. `eval_gsm8k_compare.py` loads the dense
  checkpoint with `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
  bnb_4bit_use_double_quant=True)`. It does not write a separate quantized
  checkpoint to disk for this project.
- `AWQ W4G128` is an explicit saved checkpoint. `quantize_awq.py` uses AutoAWQ
  with `w_bit=4`, `q_group_size=128`, `zero_point=true`, and `version=GEMM`.
  It calibrates on chat-formatted GSM8K training text plus reference answers,
  then saves a quantized model directory under `ckpts_awq/`.

The research question is therefore precise: after GRPO changes the dense model
weights, do those changed weights still survive W4 quantization? That is why the
main comparison is not just `base` vs `AWQ`; it is the change in behavior from
FP16 to W4 for both the base checkpoint and the GRPO checkpoint.

## Serving Add-On Plan

The remaining repo gap is engineering, not science: `scripts/serve.py` and
`scripts/benchmark.py` should make the "serve" part of `posttrain-quant-serve`
runnable. The intended scope is single A40, single GPU, no FSDP, no new training,
no new quantization, and no accuracy reruns.

What this add-on measures:

- vLLM OpenAI-compatible serving for dense FP16 and AWQ W4G128 checkpoints;
- offline vLLM throughput in generated tokens per second;
- per-request end-to-end latency p50/p95;
- peak GPU memory;
- one CSV/JSONL row per checkpoint so the four variants can be assembled into a
  small serving table.

The honest framing is: the Day 5 matrix answers whether GRPO's accuracy gain
survives W4; the serving benchmark is a deployment sanity check on the same
checkpoint family. It should not be written as a proof that AWQ speeds up serving.

## Serving vs Benchmarking

There are two separate vLLM artifacts because they answer different questions.

`scripts/serve.py` is the serving artifact. In server mode it starts vLLM's
OpenAI-compatible API server for a chosen checkpoint. That is the production-like
path: a process stays alive, owns the GPU, listens on a port, and accepts requests
through endpoints such as `/v1/completions`. This proves the repo can actually
serve the dense FP16 and AWQ checkpoints in a standard API shape, rather than
only evaluating them inside a research script. It is the right artifact to show
if someone asks, "Can this checkpoint be deployed behind an inference endpoint?"

`scripts/benchmark.py` is the measurement artifact. It uses vLLM offline mode
inside one Python process instead of coordinating a separate HTTP server. This is
intentional: for a batch Slurm job, offline mode removes port allocation, server
startup synchronization, client retry logic, and HTTP overhead from the critical
path. The benchmark still uses vLLM's model loading, KV cache, scheduler, and
generation kernels, so it measures the inference engine rather than the old
Transformers eval loop. It reports the serving-relevant quantities: generated
tokens per second, p50/p95 request latency, and peak GPU memory.

In an interview, the clean explanation is: I implemented both the deployability
path and the reproducible measurement path. `serve.py` shows the model can run
as an OpenAI-compatible vLLM service. `benchmark.py` gives stable one-row
measurements for FP16 vs AWQ without needing a long-lived server in every Slurm
job. The former answers "can we serve it?"; the latter answers "what is the
throughput, latency, and memory cost?"

What counts as done for the vLLM part:

- `serve.py` can launch a real vLLM server for FP16 and AWQ checkpoints;
- `benchmark.py` can run vLLM offline and write CSV/JSONL rows;
- `slurm/serve_benchmark.sbatch` runs one benchmark row on a single A40;
- the README summarizes the server and benchmark paths without exposing raw
  cluster run details;
- `results/serving_benchmark.md` is filled from the four benchmark rows:
  base FP16, GRPO FP16, base AWQ, and GRPO AWQ.

The benchmark does not need to keep a server running forever. A persistent server
is useful for interactive demos and API integration, but the project result only
needs a reproducible serving measurement table.

## Serving Benchmark Result

The completed A40 vLLM offline benchmark used 16 GSM8K test prompts and
`max_new_tokens=128`.

| Variant | Quantization | Throughput tok/s | p50 latency s | p95 latency s | Peak device mem GB |
| --- | --- | ---: | ---: | ---: | ---: |
| Base FP16 | none | 138.6 | 0.879 | 1.045 | 40.31 |
| GRPO FP16 | none | 139.0 | 0.879 | 1.060 | 40.31 |
| Base AWQ W4G128 | awq | 64.4 | 1.956 | 2.025 | 40.36 |
| GRPO AWQ W4G128 | awq | 64.1 | 1.957 | 2.131 | 40.36 |

The honest read is that AWQ did not produce a speedup in this specific
small-model serving run. It was about half the FP16 throughput. That does not
change the main accuracy result; it says the serving payoff was not observed for
Qwen2.5-1.5B on one A40 with this vLLM/AWQ path.

The memory column is also not a clean weight-footprint comparison. The benchmark
sampled device memory with `nvidia-smi`, and vLLM was configured with
`gpu_memory_utilization=0.90`. vLLM therefore reserves most of the A40 for
KV-cache capacity in every row. A separate capacity sweep would be needed to
claim that AWQ supports more concurrent requests or a lower memory floor.

The first benchmark used sequential request mode: 16 prompts were measured one
after another, not sent as one concurrent batch. That is a valid low-concurrency
latency metric, but it is not the metric most favorable to vLLM or quantization.
The follow-up benchmark should use `REQUEST_MODE=batch` and `BATCH_SIZE=16/32/64`
so vLLM receives many prompts in one `llm.generate([...])` call. That tests
throughput under batch pressure, which is closer to the reason AWQ might help:
serving more work within the same memory budget.

The batch-pressure sweep was then run with batch sizes 16, 32, and 64. It
confirmed that batching helps vLLM throughput overall, but AWQ still stayed
slower than FP16:

| Batch size | Base FP16 tok/s | Base AWQ tok/s | GRPO FP16 tok/s | GRPO AWQ tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1569.1 | 793.4 | 1538.9 | 900.9 |
| 32 | 3060.9 | 1494.0 | 3090.0 | 1700.2 |
| 64 | 5377.1 | 3052.6 | 5687.7 | 3275.8 |

So the serving conclusion is now stronger and cleaner: the first sequential
benchmark was not hiding an AWQ speedup. Even under batched vLLM generation, AWQ
did not improve throughput or p50 latency for this 1.5B model on one A40.

The interview-safe version is: the project answers the accuracy and
quantizability question. The serving extension proves the checkpoints can be run
through vLLM and records an honest negative serving-speed result for this
small-model setup. I would not claim AWQ improves serving here; I would say the
next serving question is a memory capacity or larger-model sweep.

## Interview Talking Points

If asked "how can accuracy improve if global weight metrics barely move?", the
answer is: global metrics like mean scale, outlier fraction, and W4 proxy MSE are
coarse distribution summaries. They can detect whether GRPO made weights much
larger, spikier, or harder to approximate with int4, but they do not measure the
direction of the update or the function computed by the network. GRPO can make
many small coordinated weight changes that shift logits toward the correct final
answer while leaving global histograms almost unchanged. The result is therefore
not contradictory: behavior changed, but no coarse quantization-pathology metric
blew up.

If asked "why use both server mode and offline benchmark mode?", the answer is:
server mode demonstrates deployability, while offline benchmark mode gives a
controlled systems measurement. The API server is what a user or application
would call. The offline benchmark is easier to reproduce on Slurm because one job
can load the model, generate realistic GSM8K prompts, measure tokens/sec,
latency, and memory, and exit cleanly. Both use vLLM; they just remove different
sources of complexity.

If asked "what does 2 epochs over 1000 prompts mean?", the answer is: with batch
size 1 and gradient accumulation 8, each optimizer update collects gradients from
8 prompts. A 250-step GRPO run therefore uses about 2000 prompt exposures. Since
the training subset contains 1000 prompts, that is about two passes over the
training slice. The held-out test100 result is separate: those 100 examples were
used only for evaluation, not for training.

If asked "what exactly did you quantize?", the answer is: I quantized the model
weights, not the dataset or the reward function. The base checkpoint and the
GRPO checkpoint are two different weight states of the same Qwen2.5-1.5B
architecture. For each state I evaluated dense FP16, bitsandbytes NF4 W4
load-time quantization, and saved AutoAWQ W4G128. In practice, W4 targets the
large transformer linear layers, especially attention projections
(`q_proj/k_proj/v_proj/o_proj`) and MLP projections
(`gate_proj/up_proj/down_proj`). The question was whether the GRPO-shifted
weights still quantize cleanly compared with the original base weights.
