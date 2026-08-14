# FSDP2 Training Track Design

## Goal

Extend the existing single-GPU Qwen post-training pipeline with a direct
PyTorch FSDP2 track for full-parameter Qwen3-8B SFT followed by GRPO. The
final distributed checkpoint must be converted into a normal Hugging Face
checkpoint directory so the existing quantization, evaluation, and vLLM
serving scripts continue to work without FSDP-specific changes.

This track uses PyTorch's `fully_shard` DTensor API. It does not use the
legacy `FullyShardedDataParallel` wrapper, DeepSpeed, or an Accelerate layer
around the FSDP calls.

## Existing Repository Conventions

The implementation will extend these existing patterns:

- Training is script-driven with `argparse`, not YAML-driven.
- Slurm wrappers translate environment variables into explicit CLI flags and
  print the resolved configuration before launch.
- The existing GRPO behavior is defined by
  `scripts/train_grpo_gsm8k.py` and `scripts/gsm8k_reward.py`.
- Runs emit machine-readable JSON, JSONL, or CSV artifacts under an output
  directory, while curated Markdown records cite Slurm job IDs and artifact
  paths.
- Quantization and serving consume an ordinary Hugging Face model directory
  through `--model_name_or_path` or `--model`.
- Existing single-GPU scripts remain available as correctness oracles and are
  not replaced by the FSDP2 entry points.

The new code will reuse the current GSM8K prompt, answer parser, exact-match
reward, prompt-leak penalty, environment setup, and result-record style.

## Chosen Architecture

A direct PyTorch training loop is preferred over two alternatives:

1. Subclassing TRL's trainer would retain more trainer internals, but model
   preparation and checkpoint ownership would remain partly hidden behind
   Accelerate and could select a legacy FSDP path.
2. Adopting TorchTitan or torchtune recipes would provide good FSDP2 examples,
   but would create a second configuration and tracking system and would not
   preserve the existing GRPO workflow directly.

The direct loop makes sharding, gradient synchronization, rollout memory, and
checkpoint state explicit while still reusing the repository's task-specific
logic.

### Files

- `train/fsdp_utils.py`: shared distributed initialization, model loading,
  FSDP2 application, gradient handling, DCP state, and run-record helpers.
- `train/fsdp_sft.py`: GSM8K SFT dataset construction and causal-LM training.
- `train/fsdp_grpo.py`: rollout generation, verifiable rewards, Dr. GRPO loss,
  reference-policy handling, and policy updates.
- `scripts/consolidate_dcp.py`: model-only DCP-to-Hugging-Face conversion.
- `scripts/launch_slurm.sh`: validated single-node Slurm and torchrun launcher.
- `bench/scaling.py`: fixed-global-batch scaling measurement using the same SFT
  training step as the real trainer.
- `docs/memory_ledger.md`: analytical and measured memory accounting.
- `docs/fsdp_notes.md`: scaling, communication, and activation-checkpointing
  results.
- `tests/`: unit and distributed smoke coverage for data, loss, checkpoint,
  tracking, and parity behavior.

## Model Initialization and Sharding

Qwen3-8B has 36 decoder blocks, hidden size 4096, intermediate size 12288,
vocabulary size 151,936, and untied input/output embeddings. Those dimensions
give approximately 8,190,735,360 trainable parameters.

Each torchrun process will:

1. Initialize NCCL and bind `LOCAL_RANK` to one CUDA device.
2. Instantiate the Hugging Face model on the meta device.
3. Apply activation-checkpoint wrappers when enabled.
4. Apply `fully_shard` bottom-up using one-dimensional CUDA `DeviceMesh`:
   - `model.model.embed_tokens` as an explicit group;
   - each of the 36 decoder blocks as its own group;
   - `model.lm_head` as an explicit group;
   - the root model last, covering the final norm and remaining parameters.
5. Materialize only local DTensor shards on the current GPU.
6. Load Hugging Face safetensors or DCP state directly into the sharded model.
7. Create the optimizer only after `fully_shard`, so it owns DTensor
   parameters rather than pre-sharding tensor objects.

The initial Hugging Face load will use DCP's Hugging Face safetensor reader
where supported by the pinned PyTorch version. This avoids first materializing
the complete 8B model on every GPU. Loading a local DCP checkpoint uses the
standard DCP filesystem reader. Both paths target canonical Hugging Face
parameter names.

### Mixed Precision

The required policy is:

- `param_dtype=torch.bfloat16` for forward, backward, and parameter all-gather;
- `reduce_dtype=torch.float32` for gradient reduction;
- `output_dtype=torch.bfloat16`;
- `reshard_after_forward=True` for the embedding, decoder, head, and root
  groups during training.

The required 1/2/4/8 scaling sweep uses native bf16 resident parameters,
gradients, and AdamW moments. This is the only on-GPU profile that gives a
valid 1-GPU Qwen3-8B point on the stated hardware, and that point requires an
80 GiB A100. A separate fp32-resident option is retained for production runs
with at least four GPUs; it is not mixed into the scaling curve.

## SFT Stage

The repository has no existing SFT path. The SFT dataset will use GSM8K train
questions and reference solutions:

- The user message uses the same instruction text as the existing GRPO prompt.
- The assistant target is the reference reasoning and final `####` answer.
- The Qwen chat template renders the complete user/assistant conversation.
- Prompt and padding tokens receive label `-100`; only assistant tokens
  contribute to causal-LM loss.
- The scaling benchmark uses packed fixed-length sequences to keep useful
  token counts identical across world sizes.

The default scaling shape is one 2048-token sequence per GPU and a global
batch of eight sequences. Gradient accumulation is therefore 8, 4, 2, and 1
microbatches at world sizes 1, 2, 4, and 8. Production batch and sequence
settings remain explicit CLI flags.

## Activation Checkpointing

Activation checkpointing is controlled by paired CLI flags and enabled by
default. Each decoder block uses non-reentrant checkpointing with RNG state
preserved. Training disables the KV cache.

The initial implementation does not add a fused or chunked language-model
loss because that could change parity with the correctness oracle. If the
LM-head logits dominate measured memory, an optimization may be proposed
after the baseline gate and memory ledger are complete.

## Gradient Accumulation and Clipping

FSDP2 can defer gradient synchronization, but doing so retains unsharded
gradients until the final microbatch and can add about one full bf16 model to
peak memory. The default instead reduce-scatters every microbatch and
accumulates into sharded gradients. This costs more communication but
preserves the memory benefit that makes the 8B run feasible.

Loss scaling produces the global token-level mean rather than an average of
unequal per-rank means. After the last microbatch, every rank executes:

1. `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`;
2. the fused AdamW step;
3. the scheduler step;
4. `zero_grad(set_to_none=True)`.

The clipping call runs on every rank. DTensor dispatch combines local shard
norms into the whole-model norm. Clipping only on rank 0 or clipping each shard
independently is incorrect.

An explicit no-sync accumulation mode may be included for a memory/performance
ablation, but is never enabled automatically.

## GRPO Stage

The distributed loop preserves the repository's final recipe:

- `num_generations=8`;
- `loss_type=dr_grpo`;
- `scale_rewards=none`;
- `beta=0.0`;
- `temperature=1.0`;
- GSM8K exact-match reward with the prompt-leak penalty.

The policy is initialized from the SFT checkpoint and remains FSDP2-sharded.
Detached rollout log-probabilities act as the old-policy values, so a separate
old-policy model is not created.

### Reference Model

- With the repository default `beta=0`, no reference model is instantiated.
- With `beta>0`, a frozen bf16 reference copy of the SFT checkpoint is loaded
  and independently FSDP2-sharded.
- The reference has no gradients or optimizer state.
- Reference log-probabilities are computed in inference-mode teacher-forcing
  microbatches.
- The immutable reference weights are not duplicated in every GRPO DCP save.
  The checkpoint records their source path, model revision, and digest so
  resume can recreate the exact reference.

### Rollout Generation

Rollouts use the same policy rather than a colocated vLLM engine. A vLLM engine
would add another full model allocation, reserve KV-cache capacity, and require
weight synchronization after each policy update.

Generation handles one prompt group of eight completions at a time. Token IDs,
old log-probabilities, rewards, masks, and completion metadata move to CPU once
the group is complete. Only the current rollout or training microbatch remains
on GPU.

Autoregressive generation exposes a deliberate policy:

- `reshard` mode releases complete layers after each generated token, saving
  memory but performing layer all-gathers repeatedly.
- `keep_unsharded` mode pays one all-gather per layer and holds one replicated
  bf16 policy during the rollout phase, adding about 15.3 GiB per GPU but
  avoiding token-by-token parameter communication.

`keep_unsharded` is selected only when an analytical preflight says it fits.
At the end of generation, every FSDP module is explicitly resharded before
reference scoring or policy training.

All ranks execute the same number of generation calls and synchronized stopping
iterations. Local prompt groups are padded when necessary so variable
completion length or dataset exhaustion cannot make one rank leave a
collective early.

## Analytical Memory Model

### Assumptions

The primary SFT prediction assumes:

- 8,190,735,360 parameters;
- native bf16 parameters, gradients, and two AdamW moments;
- fp32 gradient reduction;
- fused AdamW;
- one 2048-token sequence per GPU;
- block activation checkpointing and Flash/SDPA attention;
- no reference model;
- 3.5 GiB of checkpointed activations;
- 2.7 GiB of overlapping unsharded parameter and full-gradient communication
  buffers for world sizes greater than one;
- 1.5 GiB of PyTorch workspaces and temporary tensors;
- reserved-memory budget equal to 1.08 times predicted allocated memory.

The largest explicit bf16 FSDP group is an embedding or LM head at about
1.16 GiB. One decoder block is about 0.36 GiB.

### Native bf16 Resident State

| GPUs | Params | Grads | Adam m/v | Activations | Collective | Other | Predicted allocated | Predicted reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.26 | 15.26 | 30.51 | 3.50 | 0.00 | 1.50 | 66.03 GiB | 71.3 GiB |
| 2 | 7.63 | 7.63 | 15.26 | 3.50 | 2.70 | 1.50 | 38.22 GiB | 41.3 GiB |
| 4 | 3.81 | 3.81 | 7.63 | 3.50 | 2.70 | 1.50 | 22.96 GiB | 24.8 GiB |
| 8 | 1.91 | 1.91 | 3.81 | 3.50 | 2.70 | 1.50 | 15.33 GiB | 16.6 GiB |

Expected capacity constraints are:

- world size 1 requires an 80 GiB A100;
- world size 2 fits a 48 GiB A40 but is unsafe on a 40 GiB A100;
- world sizes 4 and 8 fit A40 or A100 nodes.

### fp32 Resident State

Keeping original DTensor parameters, gradients, and AdamW moments in fp32
requires approximately 122.05 GiB of persistent state divided by world size.

| GPUs | Persistent state | Other peak allocations | Predicted allocated | Predicted reserved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 122.05 | 5.00 | 127.05 GiB | 137.2 GiB |
| 2 | 61.03 | 7.70 | 68.73 GiB | 74.2 GiB |
| 4 | 30.51 | 7.70 | 38.21 GiB | 41.3 GiB |
| 8 | 15.26 | 7.70 | 22.96 GiB | 24.8 GiB |

This profile cannot produce a 1-GPU Qwen3-8B point on an A40 or A100. CPU
offload would change the experiment into a host-memory and PCIe benchmark and
is excluded from the homogeneous scaling curve.

### Activation Checkpointing Off

The uncheckpointed activation estimate is 13.5 GiB.

| GPUs | Predicted allocated | Predicted reserved | Capacity consequence |
| ---: | ---: | ---: | --- |
| 1 | 76.03 GiB | 82.1 GiB | likely OOM on A100-80 |
| 2 | 48.22 GiB | 52.1 GiB | OOM on A40 |
| 4 | 32.96 GiB | 35.6 GiB | expected to fit A40/A100 |
| 8 | 25.33 GiB | 27.4 GiB | expected to fit A40/A100 |

The analytical saving is about 10 GiB per GPU. Throughput cost is measured in
matched world-size-4 runs rather than asserted in advance.

### GRPO Peak

With `beta=0`, a group of eight 1,536-token sequences uses about 1.69 GiB of
KV cache. Retaining a complete policy during rollout shifts the peak at larger
world sizes:

| GPUs | Training-phase peak | Rollout-phase peak | Predicted GRPO peak |
| ---: | ---: | ---: | ---: |
| 1 | 65.5 | 50.0 | 65.5 GiB |
| 2 | 37.7 | 42.3 | 42.3 GiB |
| 4 | 22.5 | 30.9 | 30.9 GiB |
| 8 | 14.8 | 25.2 | 25.2 GiB |

With `beta>0`, adding a frozen reference shard gives approximate peaks of
80.8, 50.0, 34.7, and 27.1 GiB. Beta-enabled GRPO is therefore excluded from
the 1-GPU target and is marginal on two A40s.

These figures are analytical predictions. `docs/memory_ledger.md` will place
measured allocated and reserved values, plus a committed run ID, next to every
prediction.

## Distributed Checkpointing

Training uses `torch.distributed.checkpoint` with canonical distributed state
dictionaries. Each checkpoint contains:

- the sharded policy model;
- the sharded optimizer;
- scheduler state;
- global optimizer step and consumed-token count;
- epoch and sampler cursor;
- Python, NumPy, CPU Torch, and per-rank CUDA RNG states;
- the fully resolved training configuration;
- model, SFT, and reference revision/digest metadata;
- world size, package versions, git commit, and Slurm job ID.

Checkpoints are written only after complete optimizer steps, never midway
through accumulation or rollout generation.

### Resume Ordering

1. Recreate the model on meta.
2. Apply the identical FSDP2 plan.
3. Materialize local shards.
4. Create the optimizer from DTensor parameters.
5. Load model and optimizer through DCP `set_state_dict`.
6. Restore scheduler, progress, sampler, and RNG state.
7. Validate optimizer-state key count, dtype, shape, and global step before
   allowing the next forward pass.

A world-size change may reshard model and optimizer tensors, but changes data
partitioning and RNG streams. Such a resume is supported as recovery but is
recorded as non-bitwise. Exact continuation requires the original world size.

### Publication Safety

Each save writes to a step-specific temporary directory. After DCP completes
on every rank, rank 0 writes a success marker and atomically publishes the
final directory. `--resume latest` considers only directories containing the
success marker, so a Slurm cancellation cannot promote a partial checkpoint.

## Hugging Face Consolidation

`scripts/consolidate_dcp.py` will:

1. Load only policy weights, never optimizer state.
2. Use FSDP2/DCP to gather a full bf16 state dictionary to CPU rank 0.
3. Write one Hugging Face checkpoint directory with safetensor files and an
   index.
4. Copy tokenizer, config, chat template, and generation configuration.
5. Reload the result through `AutoModelForCausalLM.from_pretrained`.
6. Compare parameter count, selected tensor hashes, and fixed-input logits
   against the distributed source model.

"Single HF-loadable checkpoint" means one model directory; it does not require
one monolithic 16 GiB safetensor file. The output directory is accepted by the
existing AWQ quantizer, evaluation scripts, and vLLM serving scripts without
changes.

The artifact flow is:

```text
HF Qwen3-8B
  -> SFT DCP
  -> GRPO policy/reference initialization
  -> GRPO DCP
  -> consolidated Hugging Face directory
  -> existing quantization, evaluation, and serving stages
```

## Slurm and torchrun

`scripts/launch_slurm.sh` validates GPU count against `1`, `2`, `4`, and `8`,
runs the existing Great Lakes environment activation and CUDA preflight, prints
the resolved configuration, and launches a single-node torchrun command such
as:

```text
torchrun --standalone --nnodes=1 --nproc-per-node=$GPUS --module train.fsdp_sft
```

The launcher records NCCL diagnostics and fails before model allocation if the
selected storage precision and world size exceed detected VRAM.

The valid 1/2/4/8 efficiency curve runs on one homogeneous 8xA100-80 node.
Mixing a 1xA100-80 result with multi-GPU A40 results is not considered a
scaling curve. The same launcher supports feasible A40 production jobs at
world sizes 2, 4, and 8 for native bf16 state.

## Scaling Benchmark

`bench/scaling.py` imports the real SFT training step rather than implementing
a synthetic parallel path. Each world size uses:

- the same model revision, packed dataset subset, seed, global batch, sequence
  length, and activation-checkpoint flag;
- three optimizer-step warmups;
- ten unprofiled measured steps for throughput and MFU;
- three separately profiled steps for communication;
- initialized optimizer state before peak-memory counters are reset.

The output includes:

- aggregate non-padding tokens per second;
- allocated and reserved peak GiB for every rank, with mean and maximum;
- MFU using useful training FLOPs `6 * parameters * tokens`, divided by
  aggregate dense bf16 peak FLOPs and wall time;
- communication-active fraction from NCCL CUDA kernel intervals;
- exposed communication fraction from NCCL intervals not overlapped by
  compute kernels;
- step-time mean and dispersion;
- scaling efficiency `throughput_N / (N * throughput_1)`.

Profiler iterations are separate so profiler overhead does not affect the
reported throughput.

## Run Tracking and Documentation Provenance

The benchmark extends the repository's existing append-only format:

- `scaling.jsonl` for complete records;
- `scaling.csv` for tabular inspection;
- a resolved `run_config.json`;
- rank-level JSONL when per-rank detail does not fit naturally in CSV;
- curated committed records under `results/fsdp_scaling/`.

Every record contains git commit and dirty state, Slurm job ID, GPU name and
VRAM, topology, world size, package/CUDA/NCCL versions, model revision, seed,
batch shape, checkpointing flag, and metric definitions.

Every empirical sentence in `docs/memory_ledger.md` and `docs/fsdp_notes.md`
must cite a run ID backed by a committed JSON record. Analytical claims cite
their formula and model dimensions. Documentation does not present a predicted
number as measured or fill a missing run with an estimate.

## Correctness Gates

No Qwen3 scaling run starts until the parity records are committed.

### SFT Gate

Because the repository has no existing SFT trainer, the oracle is an
unsharded execution of the same data and loss code:

- Qwen2.5-0.5B;
- fixed 16-row subset and tokenized batches;
- identical dtype, optimizer, scheduler, attention backend, and checkpoint
  setting;
- unsharded versus FSDP2 world size 1.

The hard acceptance checks are:

- both paths use the same immutable model, dataset, batches, and seed;
- first-batch loss is within the declared bf16 sanity tolerance, catching
  initialization, loading, and data mismatches;
- all three seeded 20-step runs complete without non-finite losses or
  gradients; and
- the three-seed loss curve remains inside the pooled statistical noise band.

Per-step pre-clip gradient-norm differences and selected post-step parameter
update cosine remain in the committed comparison record as diagnostics. They
do not independently fail SFT acceptance: fused CUDA kernels and bf16 update
rounding may produce small trajectory differences without changing the
training outcome. This keeps the gate focused on behavioral parity while
retaining enough evidence to investigate a real regression.

### GRPO Gate

`train/fsdp_grpo.py` at world size 1 is compared with
`scripts/train_grpo_gsm8k.py` using the same small model, fixed subset, prompt
template, eight generations, reward, Dr. GRPO normalization, beta, and
decoding settings.

A fixed-rollout fixture tests the objective without sampling noise. A separate
seeded end-to-end comparison tests rollout and reward behavior.

Acceptance requires:

- fixed-rollout loss and gradient norm within a declared bf16 tolerance;
- selected update tensors with cosine similarity above 0.999;
- end-to-end loss and reward curves within the pooled three-seed noise band;
- DCP save/resume producing the same next-step loss as uninterrupted training.

After both gates pass, the execution order is the 1/2/4/8 checkpointed SFT
sweep, the world-size-4 activation-checkpointing ablation, and then the full
Qwen3 SFT-to-GRPO artifact flow.

## Expected Failure Modes

| Failure | Mechanism | Prevention and recovery |
| --- | --- | --- |
| NCCL timeout during GRPO | One rank stops generation early or executes fewer groups | Synchronized generation, padded groups, identical collective order, and NCCL diagnostics |
| NCCL timeout at job end | Rank failure or uneven checkpoint I/O | Collective DCP error propagation, explicit timeout, and no independent rank exit |
| Startup OOM | Complete HF model materialized before sharding | Meta initialization and direct safetensors-to-DTensor loading |
| Consolidation OOM | Full model and optimizer gathered together or output accidentally fp32 | Model-only CPU gather, bf16 output, safetensor sharding, and host-RAM preflight |
| Missing optimizer state after resume | Optimizer created before sharding or state saved with raw parameter IDs | Create optimizer after `fully_shard`, use canonical-FQN DCP APIs, and validate state |
| Repeated data after resume | Model restored without sampler cursor or RNG | Checkpoint consumed samples, sampler position, and all RNG states |
| Non-identical world-size-change resume | Tensor state resharding works but data/RNG streams change | Label as non-bitwise recovery and require the original world size for exact continuation |
| Accumulation OOM | No-sync retains full gradients or complete weights | Reduce-scatter every microbatch by default and memory-gate no-sync mode |
| Incorrect global loss | Unequal local token counts averaged as equal rank means | Normalize with global token counts or the fixed Dr. GRPO denominator |
| LM-head peak exceeds estimate | Full vocabulary logits and cross-entropy workspace dominate | Measure first; optimize loss only after parity is established |
| Poor GRPO scaling | Per-token all-gathers or replicated rollout weights dominate | Select rollout parameter mode by memory, and report rollout/update timing separately |
| Misleading efficiency curve | Different GPU types, precision, or offload between points | Require homogeneous hardware and identical state precision |
| Partial checkpoint selected as latest | Slurm cancellation during write | Temporary directory, success marker, and atomic publication |

## Implementation Boundaries

- Do not refactor unrelated evaluation, quantization, serving, notebook, or
  note files.
- Do not introduce DeepSpeed or wrap FSDP2 setup in Accelerate.
- Do not claim measured memory, throughput, MFU, communication fraction, or
  scaling efficiency until a committed run record exists.
- Do not run the expensive scaling sweep before both world-size-1 correctness
  gates pass.
- Preserve the existing single-GPU GRPO entry point as the comparison oracle.
