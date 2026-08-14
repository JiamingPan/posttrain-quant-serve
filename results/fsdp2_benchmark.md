# FSDP2 scaling pilot

**What this establishes:** FSDP2 `FULL_SHARD` makes full-parameter Qwen3-8B
training steps run on two A40s. The analytical no-shard memory ledger predicts
that the same optimizer configuration cannot fit on one 44.42 GiB A40.

**What it does not establish:** anything about end-to-end SFT wall clock,
convergence, or final quality. This is a benchmark pilot measuring step
throughput and peak memory, nothing else.

## Results

### Qwen3-8B, 2 x A40, FULL_SHARD

| Metric | Value |
| --- | ---: |
| Throughput | 893 tokens/s |
| Time per measured step | ~18.35 s |
| Peak memory allocated | 38.48 GiB / GPU |
| Peak memory reserved | 43.82 GiB / GPU |
| Device capacity | 44.42 GiB |
| Headroom on reserved | **about 0.60 GiB** |
| MFU | 14.65% |
| Communication active / exposed | 96.42% / 23.17% |

Scaling efficiency for the 8B configuration is recorded as `null`, not as a
number. There is no valid single-GPU 8B baseline to divide by because the
analytical memory model predicts that unsharded full-parameter training does not
fit on one A40. Reporting an efficiency here would require inventing the
denominator.

### 0.5B control, 1 GPU -> 2 GPU

| Metric | 1 GPU | 2 GPU | Ratio |
| --- | ---: | ---: | ---: |
| Throughput (tokens/s) | 2,877 | 3,185 | 1.11x |
| Peak memory / GPU (GiB) | 5.42 | 3.36 | 0.62x |
| MFU | 5.70% | 3.15% | 0.55x |
| Communication active | 0.00% | 92.72% | n/a |
| Communication exposed | 0.00% | 62.56% | n/a |

### Cost

16:29 wall clock, approximately 0.55 A40 GPU-hours.

## Reading the numbers honestly

**The 1.11x is not the headline, and taken alone it looks bad.** That is 55%
strong-scaling efficiency on two GPUs. It was measured at a fixed global batch
of 8 and a short sequence length, a regime where per-step communication
dominates the compute it overlaps with. This is a deliberately communication-
bound pilot point, not a representative training configuration, and it should
not be extrapolated.

“Communication active” is the fraction of the profiled CUDA step window covered
by NCCL kernels. “Communication exposed” is the fraction covered by NCCL kernels
that did not overlap compute. The exposed fraction is the relevant evidence for
the strong-scaling bottleneck.

**The memory column is the result on the 0.5B control.** The reduction from 5.42
to 3.36 GiB per GPU is sharding doing its job. It is a 38% reduction rather than
50% because parameters, gradients, and optimizer state shard across the two
ranks, but activations and allocator buffers do not.

**The 8B row is the result overall.** 38.48 GiB allocated against a 44.42 GiB
card, with about 0.60 GiB of headroom on reserved memory, is the difference between
this configuration fitting and not fitting. The margin is thin enough that it
should be treated as configuration-specific rather than as a general claim
about 8B models on A40s.

## Caveats

- The schedule was 3 warmup steps, 10 measured steps, and 3 profiled steps.
  Throughput figures are single-run point estimates with **no error bar** and
  should not be compared against other runs at the resolution of a few percent.
- The 0.5B control used sequence length 512. The 8B feasibility run used sequence
  length 2048. The two model rows are not throughput comparisons.
- Both runs used fixed global batch 8, activation checkpointing, and
  reduce-scatter gradient accumulation on one two-A40 node.
- Peak memory is thin enough on the 8B run that small changes to sequence length,
  batch size, or allocator behavior could push it over.
- Qwen3-8B world sizes 1, 4, and 8 are **unmeasured**. No predicted value for
  those configurations is presented as a measurement.

## Provenance

The pilot ran as Great Lakes Slurm job `57342451` from git commit
`97c1c9957a31b5234d6ab856b4a86261c7d2be8e` on node `gl1511`, using two NVIDIA
A40 GPUs and PyTorch `2.11.0+cu130`. The model revisions were
`7ae557604adf67be50417f59c2c2f167def9a775` for Qwen2.5-0.5B-Instruct and
`b968826d9c46dd6066d109eabc6255188de91218` for Qwen3-8B. Every record is tagged
`pilot`; Qwen3-8B scaling efficiency is `null` by design.

Committed source records:

- [Qwen2.5-0.5B, 1 GPU](fsdp_scaling/final-a40-pilot-97c1c99/qwen2.5-0.5b/workers/w1/worker_result.json)
  (`scaling-20260813T210628Z-19124f115f`)
- [Qwen2.5-0.5B, 2 GPU](fsdp_scaling/final-a40-pilot-97c1c99/qwen2.5-0.5b/workers/w2/worker_result.json)
  (`scaling-20260813T211004Z-c795833634`)
- [Qwen3-8B, 2 GPU](fsdp_scaling/final-a40-pilot-97c1c99/qwen3-8b/workers/w2/worker_result.json)
  (`scaling-20260813T211335Z-e1493f549e`)

The surrounding `run_config.json`, JSONL, and CSV files preserve the controller
configuration and the repo's standard run-tracking formats.
