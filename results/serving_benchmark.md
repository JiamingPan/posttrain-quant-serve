# Serving Benchmark

Single-GPU vLLM serving benchmark for the same 1.5B checkpoint family used in
the GSM8K quantization result.

Setup:

- hardware: one A40 GPU;
- engine: vLLM offline generation;
- model family: `Qwen2.5-1.5B-Instruct`;
- prompts: 16 GSM8K test prompts, chat-formatted;
- generation cap: `max_new_tokens=128`;
- dtype: `float16`;
- `max_model_len=2048`;
- `gpu_memory_utilization=0.90`.

Raw rows were written by `scripts/benchmark.py` to:

```text
$PQS_ROOT/results/serving/qwen2_5_1_5b_memfix/serving_benchmark.csv
$PQS_ROOT/results/serving/qwen2_5_1_5b_memfix/serving_benchmark.jsonl
```

## Results

| Variant | Quantization | Generated tokens | Throughput tok/s | Latency p50 s | Latency p95 s | Peak device mem GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base FP16 | none | 2004 | 138.6 | 0.879 | 1.045 | 40.31 |
| GRPO FP16 | none | 2048 | 139.0 | 0.879 | 1.060 | 40.31 |
| Base AWQ W4G128 | awq | 1909 | 64.4 | 1.956 | 2.025 | 40.36 |
| GRPO AWQ W4G128 | awq | 2048 | 64.1 | 1.957 | 2.131 | 40.36 |

## Reading

On this small 1.5B / A40 / 16-prompt serving slice, AWQ W4G128 was slower than
dense FP16 under vLLM: about 64 tok/s versus 139 tok/s. This is an honest systems
result, not a failure of the accuracy/quantization claim. W4 quantization reduces
stored weight size, but it does not guarantee higher throughput for every model,
kernel, batch size, and GPU. For this setup, AWQ kernel and scheduling overhead
appears to dominate the small-model serving path.

The FP16 base and FP16 GRPO checkpoints have essentially identical throughput and
latency. The AWQ base and AWQ GRPO checkpoints are also essentially identical to
each other. This is consistent with the main study result: GRPO changed behavior
enough to improve GSM8K accuracy, but did not create an obvious serving-path
slowdown relative to the matching base checkpoint.

Peak device memory should be read carefully. The benchmark samples device memory
with `nvidia-smi`, and vLLM was run with `gpu_memory_utilization=0.90`. vLLM uses
that setting to reserve GPU memory for KV-cache capacity, so all four rows sit
near 40 GB on the A40. This column is therefore a vLLM allocation-at-this-cap
measurement, not a clean model-weight-footprint comparison. A separate memory-fit
or capacity sweep would be needed to make a strong claim about serving more
concurrent requests from AWQ.

## Interview Summary

The repo now contains both sides of the serving story. `scripts/serve.py` can
start a real OpenAI-compatible vLLM server for the dense and AWQ checkpoints.
`scripts/benchmark.py` gives a reproducible Slurm-friendly measurement of
tokens/sec, latency, and device memory. In this single-A40 run, quantization
preserved the GRPO accuracy gain but did not improve vLLM throughput for the
small 1.5B model; the honest conclusion is that the deployment path works, while
the speed payoff depends on model size, kernels, and serving configuration.
