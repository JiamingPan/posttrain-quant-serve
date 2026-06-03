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
survives W4; the serving benchmark checks the systems payoff of W4 on the same
checkpoint family.

## Interview Talking Point

If asked "how can accuracy improve if global weight metrics barely move?", the
answer is: global metrics like mean scale, outlier fraction, and W4 proxy MSE are
coarse distribution summaries. They can detect whether GRPO made weights much
larger, spikier, or harder to approximate with int4, but they do not measure the
direction of the update or the function computed by the network. GRPO can make
many small coordinated weight changes that shift logits toward the correct final
answer while leaving global histograms almost unchanged. The result is therefore
not contradictory: behavior changed, but no coarse quantization-pathology metric
blew up.
