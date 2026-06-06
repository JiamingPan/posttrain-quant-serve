# Serving Benchmark

Single-GPU vLLM benchmark for the same 1.5B checkpoint family used in the GSM8K
quantization result. This is a deployment sanity check and a measured serving
result, not evidence that AWQ improves serving speed in this setting.

Sequential setup:

- hardware: one A40 GPU;
- engine: vLLM offline generation;
- model family: `Qwen2.5-1.5B-Instruct`;
- prompts: 16 GSM8K test prompts, chat-formatted;
- request mode: sequential, one prompt per `llm.generate` call;
- generation cap: `max_new_tokens=128`;
- dtype: `float16`;
- `max_model_len=2048`;
- `gpu_memory_utilization=0.90`;
- runner: `slurm/serve_benchmark.sbatch`;
- script: `scripts/benchmark.py`.

Batch-pressure setup:

- same model/checkpoint family and A40 hardware;
- `REQUEST_MODE=batch`;
- `BATCH_SIZE=16, 32, 64`;
- `NUM_PROMPTS=4 * BATCH_SIZE`;
- `max_new_tokens=128`.

## Sequential Results

| Variant | Quantization | Generated tokens | Throughput tok/s | Latency p50 s | Latency p95 s | Peak device mem GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base FP16 | none | 2004 | 138.6 | 0.879 | 1.045 | 40.31 |
| GRPO FP16 | none | 2048 | 139.0 | 0.879 | 1.060 | 40.31 |
| Base AWQ W4G128 | awq | 1909 | 64.4 | 1.956 | 2.025 | 40.36 |
| GRPO AWQ W4G128 | awq | 2048 | 64.1 | 1.957 | 2.131 | 40.36 |

## Batched Results

| Batch size | Variant | Quantization | Num prompts | Generated tokens | Throughput tok/s | Latency p50 s | Latency p95 s | Peak device mem GB |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Base FP16 | none | 64 | 8032 | 1569.1 | 0.968 | 2.220 | 39.96 |
| 16 | GRPO FP16 | none | 64 | 8169 | 1538.9 | 0.962 | 2.426 | 39.96 |
| 16 | Base AWQ W4G128 | awq | 64 | 7410 | 793.4 | 2.023 | 3.278 | 39.92 |
| 16 | GRPO AWQ W4G128 | awq | 64 | 8167 | 900.9 | 2.025 | 3.001 | 39.92 |
| 32 | Base FP16 | none | 128 | 15893 | 3060.9 | 1.107 | 1.951 | 40.05 |
| 32 | GRPO FP16 | none | 128 | 16346 | 3090.0 | 1.061 | 2.139 | 40.05 |
| 32 | Base AWQ W4G128 | awq | 128 | 14889 | 1494.0 | 2.208 | 3.452 | 40.01 |
| 32 | GRPO AWQ W4G128 | awq | 128 | 16289 | 1700.2 | 2.130 | 3.204 | 40.01 |
| 64 | Base FP16 | none | 256 | 31506 | 5377.1 | 1.226 | 2.200 | 40.27 |
| 64 | GRPO FP16 | none | 256 | 32690 | 5687.7 | 1.238 | 2.045 | 40.27 |
| 64 | Base AWQ W4G128 | awq | 256 | 30004 | 3052.6 | 2.307 | 2.953 | 40.23 |
| 64 | GRPO AWQ W4G128 | awq | 256 | 32577 | 3275.8 | 2.311 | 3.043 | 40.23 |

## Reading

On this small 1.5B / A40 / 16-prompt serving slice, AWQ W4G128 was slower than
dense FP16 under vLLM: about 64 tok/s versus 139 tok/s. This is a negative
serving result, not a failure of the accuracy/quantization claim. W4
quantization reduces stored weight size, but it does not guarantee higher
throughput for every model, kernel, batch size, and GPU. For this setup, AWQ
kernel and scheduling overhead appears to dominate the small-model serving path.

The batch-pressure sweep removed the main ambiguity in the first serving result.
Batching increased throughput substantially for all variants, which confirms
that vLLM is doing useful batched scheduling. However, AWQ still did not beat
FP16 at batch size 16, 32, or 64. Across the batched sweep, AWQ reached about
0.49x-0.57x of base FP16 throughput and about 0.55x-0.59x of GRPO FP16
throughput. The serving-speed conclusion is therefore stable: for this 1.5B
model on one A40, AWQ preserved accuracy but did not improve vLLM throughput or
latency.

The FP16 base and FP16 GRPO checkpoints have essentially identical throughput and
latency. The AWQ base and AWQ GRPO checkpoints are also essentially identical to
each other. This is consistent with the main study result: GRPO changed behavior
enough to improve GSM8K accuracy, but did not create an obvious serving-path
slowdown relative to the matching base checkpoint.

Peak device memory should be read carefully. The benchmark samples device memory
with `nvidia-smi`, and vLLM was run with `gpu_memory_utilization=0.90`. vLLM uses
that setting to reserve GPU memory for KV-cache capacity, so all four rows sit
near 40 GB on the A40. This column is therefore a vLLM allocation-at-this-cap
measurement, not a clean model-weight-footprint comparison and not evidence that
quantization failed to reduce checkpoint size. A separate disk-size,
memory-after-load, or capacity/concurrency sweep would be needed to make a
strong claim about serving more concurrent requests from AWQ.

## Interview Summary

The repo now contains both sides of the serving story. `scripts/serve.py` can
start a real OpenAI-compatible vLLM server for the dense and AWQ checkpoints.
`scripts/benchmark.py` gives a reproducible Slurm-friendly measurement of
tokens/sec, latency, and device memory in both sequential and batched modes. In
this single-A40 run, quantization preserved the GRPO accuracy gain but did not
improve vLLM throughput for the small 1.5B model, even under batch pressure. The
honest conclusion is that the deployment path works, while the serving-speed
payoff was not observed here and likely depends on model size, kernels, and
memory/capacity constraints.
