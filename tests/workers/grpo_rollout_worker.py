from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from tests.tiny_qwen import build_tiny_qwen3
from train.fsdp_utils import (
    FSDPSettings,
    destroy_distributed,
    fully_shard_qwen,
    init_distributed,
    rollout_parameter_state,
)
from train.grpo_core import RolloutConfig, generate_rollout_batch


class NumericTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": torch.tensor([[2, 4, 5] for _ in texts]),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }

    def batch_decode(self, token_ids, **kwargs):
        del kwargs
        return ["#### 1" for _ in token_ids]


def _all_sharded(model) -> bool:
    return all(isinstance(parameter, DTensor) for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    ctx = init_distributed(timeout_seconds=120)
    try:
        model = build_tiny_qwen3()
        fully_shard_qwen(model, ctx, FSDPSettings())
        with rollout_parameter_state(model, "keep_unsharded"):
            pass
        after_success = _all_sharded(model)
        try:
            with rollout_parameter_state(model, "keep_unsharded"):
                raise RuntimeError("injected rollout failure")
        except RuntimeError:
            pass
        after_failure = _all_sharded(model)

        torch.manual_seed(101)
        rollout = generate_rollout_batch(
            model,
            NumericTokenizer(),
            prompt_texts=["prompt"],
            answers=["work #### 1"],
            config=RolloutConfig(
                num_generations=2,
                max_prompt_length=8,
                max_completion_length=2,
                rollout_mode="reshard",
                top_k=16,
            ),
            ctx=ctx,
        )
        local_tokens = rollout.completion_input_ids.to(ctx.device)
        generated_tokens = [torch.zeros_like(local_tokens) for _ in range(ctx.world_size)]
        dist.all_gather(generated_tokens, local_tokens)
        payload = {
            "all_groups_sharded_after_success": after_success,
            "all_groups_sharded_after_forced_error": after_failure,
            "rollout_cpu_backed": all(
                tensor.device.type == "cpu"
                for tensor in (
                    rollout.prompt_input_ids,
                    rollout.completion_input_ids,
                    rollout.old_logps,
                    rollout.rewards,
                )
            ),
            "same_collective_schedule": all(
                torch.equal(generated_tokens[0], tokens)
                for tokens in generated_tokens[1:]
            ),
        }
    finally:
        destroy_distributed()
    if ctx.rank == 0:
        args.result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
