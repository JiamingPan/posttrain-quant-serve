# FSDP Notes

## Why This Project Needs FSDP

A full-parameter fine-tune stores more than model weights. For an 8B-parameter model in bf16 with Adam:

| Component | Rough Size |
| --- | ---: |
| Parameters, bf16 | 16 GB |
| Gradients, bf16 | 16 GB |
| Adam state: fp32 master weights, first moment, second moment | 96 GB |
| Subtotal before activations | 128 GB |

The optimizer state dominates because Adam keeps three fp32 tensors at parameter scale. Activations, temporary buffers, and fragmentation add more pressure on top of this estimate.

## DDP vs FSDP

`DistributedDataParallel` splits the batch but replicates the whole model, gradients, and optimizer state on every GPU. If the full training state does not fit on one GPU, adding more DDP GPUs does not solve the memory problem.

`FullyShardedDataParallel` shards parameters, gradients, and optimizer state across data-parallel ranks. During forward and backward, FSDP all-gathers the full parameters for the current wrapped unit, computes with them, and then reshares/frees parameters the rank does not own. Gradients are reduce-scattered so each rank keeps only its shard.

The tradeoff is memory for communication: FSDP buys memory headroom by adding all-gather and reduce-scatter traffic.

## ZeRO Mapping

| Method | Optimizer State | Gradients | Parameters |
| --- | --- | --- | --- |
| ZeRO-1 | Sharded | Replicated | Replicated |
| ZeRO-2 | Sharded | Sharded | Replicated |
| ZeRO-3 | Sharded | Sharded | Sharded |

FSDP `FULL_SHARD` is the ZeRO-3-like setting. FSDP `SHARD_GRAD_OP` is closer to ZeRO-2.

## Knobs To Understand

- Sharding strategy: start with `FULL_SHARD`; consider `SHARD_GRAD_OP` only if memory is already fine and communication is the bottleneck.
- Mixed precision: use bf16 compute on modern NVIDIA GPUs; consider fp32 reduction only if stability becomes an issue.
- Activation checkpointing: saves activation memory by recomputing during backward; expect extra compute cost.
- Auto-wrap policy: controls FSDP granularity, usually wrapping each transformer block.
- CPU offload: fallback for VRAM pressure; usually slower and not the first choice.
- Checkpointing: understand full vs sharded state dicts before the first serious 7B run.

## Understanding Check

Why can't DDP train a model that does not fit on a single GPU, even with 8 GPUs?

DDP still keeps the whole model, gradient set, and optimizer state on every GPU. It only partitions the data batch, not the training state.

What exactly does FSDP all-gather, at what granularity, and when is it freed?

FSDP all-gathers parameters for a wrapped unit, commonly a transformer block, before that unit needs them in forward/backward. After computation, ranks keep their owned shard and free the gathered full parameters depending on the resharding policy.

ZeRO-2 vs ZeRO-3: what does each shard?

ZeRO-2 shards optimizer state and gradients while parameters remain replicated. ZeRO-3 also shards parameters. FSDP `FULL_SHARD` is ZeRO-3-like.

For an 8B Adam full fine-tune in bf16, where does most of the memory go?

The Adam optimizer state, because it keeps fp32 master weights plus two fp32 moment tensors.

What does activation checkpointing save, and what does it cost?

It saves activation memory by not storing every intermediate activation from forward. It costs extra compute because those activations are recomputed during backward.
