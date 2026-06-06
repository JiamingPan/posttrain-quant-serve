# posttrain-quant-serve

RL post-training a small Qwen model with GRPO, quantizing the trained checkpoint to
4-bit, and measuring **whether RL post-training changes how cleanly the model quantizes.**
Single-GPU, end to end: GRPO -> quantize (bnb-NF4 and AWQ W4G128) -> eval -> serve.

## Research question

**Does RL post-training change how cleanly a model quantizes?**

Concretely, it compares four variants on held-out GSM8K, at FP16 and at 4-bit:

- Base model (`Qwen2.5-1.5B-Instruct`)
- GRPO-trained model
- Each, quantized to W4 (bitsandbytes NF4 *and* AWQ W4G128)

and asks whether the GRPO accuracy gain measured at FP16 **survives** quantization,
plus whether GRPO changed the model's weight-level quantizability (outlier structure,
W4 reconstruction error).

## Headline result

On `Qwen2.5-1.5B-Instruct`, chat-formatted GSM8K, held-out `test100`:

| Variant | FP16 | bnb-NF4 W4 | AWQ W4G128 |
| --- | ---: | ---: | ---: |
| Base | 0.68 | 0.65 | 0.58 |
| GRPO (data1000) | 0.72 | 0.68 | 0.67 |
| **Δ (GRPO − base)** | **+0.04** | **+0.03** | **+0.09** |

- `gain_survival_w4 = Δw4 − Δfp16 = −0.01` (the FP16 gain survives bnb-NF4 W4 intact)
- `gain_survival_awq = Δawq − Δfp16 = +0.05` (gain survives AWQ; AWQ appears *more*
  favorable to the GRPO checkpoint on this slice — flagged as a candidate, see caveats)

**Conclusion:** there is no evidence that this GRPO recipe makes the model harder to
quantize. The held-out FP16 gain survives both 4-bit schemes. This is corroborated by
the weight diagnostics: base vs GRPO have near-identical global outlier fraction and W4
reconstruction error, so the behavioral RL shift did not coincide with a global
quantizability change.

### Caveats (stated up front, on purpose)

- `n = 100`, so 1-sigma binomial stderr ≈ ±0.047. The FP16 gain (+0.04) is small
  relative to that; treat it as a clean-but-modest effect, not a large one.
- The "AWQ favors GRPO" gap (base 0.58 vs GRPO 0.67) is ~1.3σ — *suggestive, not
  significant*. Reported as a result candidate; larger-n confirmation is future work.
- Base `Qwen2.5-1.5B-Instruct` is already strong on GSM8K (0.68), so headroom for an RL
  gain is inherently limited; a harder benchmark (MATH) is the natural next step for a
  larger effect.

## What this repo establishes (and a measurement bug it caught)

The result above is only trustworthy because the eval was first debugged. An earlier
matrix showed wildly different numbers because the eval fed **raw-text prompts to a chat
model**, so generations never emitted the `<|im_end|>` stop token and were truncated
before reaching the answer line (`hit_max_new_tokens_rate = 1.0` on every row). Fixing
this — standardizing train *and* eval on the Qwen chat template, stopping on `<|im_end|>`,
and giving the model room to finish (`max_new_tokens=512`) — is what made the deltas
measurable. The final matrix is clean: parse rate 1.0, prompt-leak 0.0, ~0 max-token
hits, EOS ≈ 1.0. See `notes/day4.md` and `notes/day5.md` for the full diagnosis.

This study also documents a **quantization-floor finding** at 0.5B: both bnb-NF4 and AWQ
erased the gain there, but base sat at the accuracy floor regardless of training, so the
0.5B result is a confounded diagnostic, not an answer — which is why the headline result
is reported on 1.5B. See `notes/day4.md`.

## Scope (what this is, and isn't)

- **Single-GPU** (one A40). `Qwen2.5-1.5B-Instruct` fits on one GPU (~3 GB), so no model
  sharding is needed.
- **FSDP / multi-GPU is deliberately out of scope** — it solves a memory problem this
  model doesn't have at this scale. It's the path for ≥7B models; noted as future work,
  not faked here.
- GRPO recipe (`g8_dr100`): `num_generations=8`, `loss_type=dr_grpo`,
  `scale_rewards=none`, `beta=0.0`, `temperature=1.0`, trained on 1000 GSM8K prompts
  for ~2 epochs (250 steps). The Dr. GRPO objective and reward-std removal were adopted
  to fix an advantage-collapse failure mode diagnosed earlier (see `notes/day3.md`).
- Reward is **verifiable** (GSM8K answer-checker, no learned reward model) → this is RLVR.

## Stack

- TRL `GRPOTrainer` for RL post-training
- GSM8K answer-checker for verifiable rewards (`scripts/gsm8k_reward.py`)
- bitsandbytes NF4 (load-time W4) and AutoAWQ W4G128 (calibration-aware W4)
- vLLM OpenAI-compatible serving and offline latency/throughput benchmarking
- Slurm on the Michigan Great Lakes cluster (single A40)

## Reproduce

GRPO training, held-out eval, AWQ quantization, and the full 6-row matrix are all driven
by `slurm/*.sbatch` wrappers with env-var overrides. Exact commands, job IDs, and elapsed
times for every step of the final run are recorded in `notes/day5.md`. Key scripts:

```text
scripts/train_grpo_gsm8k.py     GRPO training (chat-formatted prompts)
scripts/gsm8k_reward.py         verifiable reward + answer extraction
scripts/quantize_awq.py         AWQ W4G128 quantization
scripts/eval_gsm8k_compare.py   FP16 / bnb-W4 / AWQ eval matrix
scripts/weight_outlier_diagnostics.py   per-layer outlier / W4 reconstruction stats
scripts/serve.py                vLLM OpenAI-compatible server / offline smoke
scripts/benchmark.py            vLLM offline throughput, latency, and memory benchmark
```

## Serving & benchmarking

The main result is accuracy-side: the GRPO gain survives W4 quantization. The
serving code is included as a deployment sanity check, not as a claim that AWQ is
faster in this small setting. `scripts/serve.py` can start an OpenAI-compatible
vLLM endpoint for the dense or AWQ checkpoints, and `scripts/benchmark.py`
measures offline vLLM throughput, latency, and device allocation on one GPU.

The completed single-A40 serving benchmark tested both sequential requests and
batched vLLM generation. It did **not** retrain, requantize, or rerun the
accuracy matrix.

Batch-64 summary:

| Variant | Quantization | Throughput tok/s | Latency p50 s | Latency p95 s | Peak mem GB |
| --- | --- | ---: | ---: | ---: | ---: |
| Base FP16 | none | 5377.1 | 1.226 | 2.200 | 40.27 |
| GRPO FP16 | none | 5687.7 | 1.238 | 2.045 | 40.27 |
| Base AWQ W4G128 | awq | 3052.6 | 2.307 | 2.953 | 40.23 |
| GRPO AWQ W4G128 | awq | 3275.8 | 2.311 | 3.043 | 40.23 |

In this small 1.5B / A40 run, AWQ was slower than FP16 under vLLM in both
sequential and batched modes. Quantization preserved the accuracy-side GRPO
result, but it did not produce a serving throughput or latency win here. The
memory column is also not a model-weight-footprint claim: vLLM was run with
`gpu_memory_utilization=0.90`, so it reserves most of the A40 for KV-cache
capacity in every row. Full sequential and batch-sweep tables are in
`results/serving_benchmark.md`.

## Layout

```text
configs/   Training, serving, and quantization configs
notes/     Study log (day0–day5), reading notes, diagnoses
scripts/   Training, quantization, eval, and diagnostic scripts
slurm/     sbatch templates for cluster runs
results/   Curated result tables and plots
```

Large local artifacts (datasets, checkpoints, raw logs) are git-ignored.

## Study log

The `notes/` directory is a dated log of the actual investigation, kept readable for
interview prep: `day3.md` (advantage-collapse diagnosis + fix), `day4.md` (quantization
pipeline, 0.5B floor, stopping bug), `day5.md` (final data1000 matrix and interpretation),
`day6.md` (serving/benchmark interpretation), plus `grpo-literature.md` (annotated
reading list: DeepSeekMath, Dr. GRPO, DAPO, GSPO, RLVR, quantization×RL).

## Future work

- Larger-n eval (test200+) to tighten the AWQ-survival estimate.
- MATH or another harder benchmark, where base headroom is larger.
- Serving memory-fit/capacity sweep if making a strong claim about AWQ concurrency.
- FSDP multi-GPU scaling to a ≥7B model, where sharding is actually required.
