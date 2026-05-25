# Reading Queue

Start with `notes/project-overview.md`. Then use this reading order.

## Day 0

1. PyTorch FSDP getting-started tutorial
   - Extract: DDP vs FSDP mental model, all-gather/reduce-scatter flow, basic API names.
2. ZeRO paper, section 3
   - Extract: ZeRO-1/2/3, memory partitioning, why optimizer state dominates.

## Day 1

1. PyTorch FSDP paper or deeper API docs
   - Extract: wrapping policy, mixed precision, checkpoint/state dict behavior.
2. torchtune distributed full fine-tune recipe docs
   - Extract: official config names, launch command, checkpoint layout.

## Later

1. vLLM/PagedAttention
2. AWQ or GPTQ
