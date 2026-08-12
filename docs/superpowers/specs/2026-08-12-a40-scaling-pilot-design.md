# A40-Constrained FSDP2 Scaling Pilot Design

## Purpose

The production scaling benchmark targets Qwen3-8B at world sizes 1, 2, 4,
and 8 on one homogeneous eight-GPU node. The available Great Lakes resource
is limited to at most two NVIDIA A40 48 GiB GPUs. That hardware cannot run the
Qwen3-8B world-size-1 point: the analytical native-bf16 prediction is
66.03 GiB allocated and 71.31 GiB reserved on one GPU.

This pilot therefore has two narrower goals:

1. validate the existing scaling measurement and run-tracking machinery at
   world sizes 1 and 2 with a small model; and
2. obtain one real Qwen3-8B world-size-2 A40 measurement without presenting it
   as a complete scaling curve.

The pilot does not replace the production 1/2/4/8 sweep. It must not populate
unmeasured Qwen3-8B points with analytical estimates.

## Resource Ceiling

The complete pilot uses one single-node Slurm allocation with two A40 GPUs and
a wall-time limit of one hour. The maximum allocation is therefore two
GPU-hours. The three measurements run sequentially inside that allocation so
they do not compete for device memory or contaminate one another's allocator
and profiler state.

No Slurm job is submitted as part of implementation. Submission requires a
separate preview of the exact command, target account and partition, wall-time,
and side effects, followed by explicit `APPROVE RUN` authorization.

## Experiments

### Small-Model Harness Validation

Run `Qwen/Qwen2.5-0.5B-Instruct` first at world size 1 and then at world size 2
on the same two-A40 node. Both points use:

- fixed global batch size 8;
- local microbatch size 1;
- sequence length 512;
- activation checkpointing enabled;
- reduce-scatter gradient accumulation;
- identical model revision, dataset subset, packing, seed, precision, and
  attention backend;
- 3 optimizer-step warmups;
- 10 unprofiled measured optimizer steps; and
- 3 separately profiled optimizer steps.

The derived gradient-accumulation count is 8 at world size 1 and 4 at world
size 2. Only the unprofiled ten-step window contributes to throughput, step
dispersion, and MFU. Communication fractions come from the separate profiler
window so profiler overhead does not contaminate throughput.

Because world size 1 is present, the world-size-2 pilot scaling efficiency is
computed as `throughput_2 / (2 * throughput_1)`.

### Qwen3-8B Feasibility Measurement

Run `Qwen/Qwen3-8B` once at world size 2 with:

- fixed global batch size 8;
- local microbatch size 1;
- gradient accumulation 4;
- sequence length 2048;
- native bf16 resident state;
- activation checkpointing enabled;
- reduce-scatter gradient accumulation;
- 3 optimizer-step warmups;
- 10 unprofiled measured optimizer steps; and
- 3 separately profiled optimizer steps.

This point uses the production sequence length so its allocated and reserved
memory can be compared with the analytical world-size-2 ledger prediction of
38.22 GiB allocated and 41.28 GiB reserved per GPU. The prediction remains
labelled analytical until the committed pilot record supplies measured values.

The Qwen3-8B record reports tokens per second, MFU, step-time spread,
communication-active and communication-exposed fractions, and allocated and
reserved peak memory for both ranks. Its `scaling_efficiency` is `null`
because no valid Qwen3-8B world-size-1 A40 baseline exists.

## Interpretation Boundaries

Fixed global batch size 8 makes the small-model comparison a strong-scaling
experiment. At world size 2, each GPU processes only half of the global batch.
The compute available to hide collective communication therefore shrinks,
while the collective setup and synchronization costs remain. The resulting
pilot efficiency is expected to be a communication-heavy lower bound for this
small-batch regime. It is not representative of a production configuration
with a larger global batch and must not be extrapolated to 4 or 8 GPUs.

The small-model and Qwen3-8B results are separate experiments. Their sequence
lengths are 512 and 2048 respectively, and their parameter counts and
communication-to-compute ratios differ. Their throughput, MFU, memory, and
communication fractions must not be compared against one another.

## Pilot Mode

The existing full-sweep behavior remains the default and retains its strict
requirements: world sizes exactly 1, 2, 4, and 8; one homogeneous eight-GPU
allocation; complete rank records; and the production correctness gates.

An explicit `--pilot` mode adds only the constrained behavior needed here:

- accept an ordered unique subset of supported world sizes;
- require a homogeneous allocation whose visible GPU count equals the largest
  requested world size;
- label controller and worker records with `benchmark_mode: "pilot"`;
- compute scaling efficiency only when a world-size-1 baseline is present;
- leave scaling efficiency `null` for an isolated world-size-2 point; and
- validate the requested subset without weakening the full-sweep validator.

The small-model controller runs with `--pilot --world_sizes 1,2`. The
Qwen3-8B controller runs separately with `--pilot --world_sizes 2` so the two
configurations receive different comparison digests and output directories.

Qwen3-8B pilot execution additionally requires:

- a clean Git worktree;
- the committed, passing SFT correctness gate at HEAD;
- a pinned model revision; and
- a successful memory preflight before weight materialization.

The SFT scaling pilot does not require the GRPO correctness gate because it
does not execute GRPO. The later production SFT-to-GRPO artifact flow retains
its GRPO gate.

## Records and Provenance

Pilot results reuse the repository's append-only scaling record format. Every
record includes:

- pilot/full mode;
- Git commit and dirty state;
- Slurm job ID and hostname;
- GPU name, capacity, topology, and per-rank identity;
- model name and immutable resolved revision;
- dataset digest, seed, batch geometry, and sequence length;
- warmup, measurement, and profiling counts;
- metric definitions and the comparison-configuration digest; and
- a unique run ID.

The allocation writes two independent result trees under one job directory:

```text
results/raw/fsdp_scaling/a40-pilot-<commit>/
  qwen2.5-0.5b/
    workers/w1/
    workers/w2/
    scaling.jsonl
    ranks.jsonl
    run_config.json
  qwen3-8b/
    workers/w2/
    scaling.jsonl
    ranks.jsonl
    run_config.json
```

Raw profiler traces and logs remain uncommitted. Only compact validated JSON
records selected for documentation are curated under `results/fsdp_scaling/`.
Every empirical statement in `docs/memory_ledger.md` or `docs/fsdp_notes.md`
must cite one of those committed run IDs.

## Validation

Full-sweep validation remains unchanged. Pilot validation requires:

- record world sizes exactly match the requested pilot subset;
- all points use homogeneous GPU hardware;
- comparison digests match within a multi-point pilot;
- global batch size is exactly 8;
- throughput and MFU are finite and positive;
- communication fractions are finite and in `[0, 1]`;
- step-time mean is positive and the ten measured step durations are present;
- every requested rank has positive allocated memory and reserved memory no
  smaller than allocated memory; and
- scaling efficiency is positive only when a world-size-1 baseline exists,
  otherwise it is `null`.

Tests cover pilot subset parsing, visible-GPU requirements, mode labelling,
one-point null efficiency, two-point efficiency, strict full-sweep behavior,
Qwen3 SFT-gate enforcement, non-finite metrics, and incomplete rank records.

## Failure Handling

Each measurement is a fresh torchrun process. Process exit destroys its
distributed group and releases all GPU allocations before the next
measurement begins.

The job stops on the first failed measurement and preserves completed output
directories for diagnosis. Qwen3-8B preflight runs before model weights are
materialized. If its predicted reserved peak exceeds 95% of detected GPU
capacity, the 8B process exits without loading the model. CUDA OOM, NCCL
timeout, non-finite metrics, mismatched configuration digests, or missing rank
records fail validation rather than producing a partial success record.

No automatic retry is performed inside the paid allocation. A retry requires
inspection of the committed configuration and a separately approved Slurm
submission.

## Documentation Outcome

After a successful pilot:

- `docs/fsdp_notes.md` may report the 0.5B 1-to-2-GPU pilot efficiency with an
  explicit small-batch strong-scaling lower-bound warning;
- `docs/memory_ledger.md` may place the measured Qwen3-8B world-size-2 memory
  beside its analytical prediction;
- Qwen3-8B world sizes 1, 4, and 8 remain explicitly unmeasured; and
- no performance relationship is drawn between the 0.5B/512-token and
  8B/2048-token experiments.

The production scaling curve and activation-checkpointing ablation remain
future measurements requiring suitable homogeneous hardware.
