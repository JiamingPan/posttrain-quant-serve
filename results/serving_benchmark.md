# Serving Benchmark Placeholder

This table is intentionally empty until the vLLM benchmark is run on a single A40
with `slurm/serve_benchmark.sbatch`. The benchmark writes raw rows to:

```text
/scratch/huterer_root/huterer0/jiamingp/pqs/results/serving/qwen2_5_1_5b/serving_benchmark.csv
/scratch/huterer_root/huterer0/jiamingp/pqs/results/serving/qwen2_5_1_5b/serving_benchmark.jsonl
```

## Planned A40 Rows

| Variant | Quantization | Throughput tok/s | Latency p50 s | Latency p95 s | Peak mem GB |
| --- | --- | ---: | ---: | ---: | ---: |
| Base FP16 | none | TBD | TBD | TBD | TBD |
| GRPO FP16 | none | TBD | TBD | TBD | TBD |
| Base AWQ W4G128 | awq | TBD | TBD | TBD | TBD |
| GRPO AWQ W4G128 | awq | TBD | TBD | TBD | TBD |
