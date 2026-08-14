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
    clip_global_grad_norm_,
    destroy_distributed,
    fully_shard_qwen,
    init_distributed,
    rollout_parameter_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()

    ctx = init_distributed(timeout_seconds=120)
    process_group_destroyed = False
    try:
        model = build_tiny_qwen3()
        global_numel = sum(parameter.numel() for parameter in model.parameters())
        groups = fully_shard_qwen(model, ctx, FSDPSettings())
        input_ids = torch.tensor([[2, 4, 5, 3]], device=ctx.device)
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        all_dtensor = all(isinstance(parameter, DTensor) for parameter in model.parameters())

        local_resident_numel = torch.tensor(
            sum(parameter.to_local().numel() for parameter in model.parameters()),
            device=ctx.device,
            dtype=torch.int64,
        )
        dist.all_reduce(local_resident_numel)

        local_squared_norm = torch.zeros((), device=ctx.device, dtype=torch.float64)
        for parameter in model.parameters():
            if parameter.grad is not None:
                local_squared_norm += parameter.grad.to_local().double().square().sum()
        dist.all_reduce(local_squared_norm)
        expected_norm = local_squared_norm.sqrt().float()
        clip_norm = clip_global_grad_norm_(model, max_norm=0.5)
        if isinstance(clip_norm, DTensor):
            clip_norm = clip_norm.full_tensor()
        gathered_norms = [torch.zeros_like(clip_norm) for _ in range(ctx.world_size)]
        dist.all_gather(gathered_norms, clip_norm)
        with rollout_parameter_state(model, "keep_unsharded"):
            rollout_unsharded = all(
                not isinstance(parameter, DTensor) for parameter in model.parameters()
            )
        rollout_resharded = all(
            isinstance(parameter, DTensor) for parameter in model.parameters()
        )

        payload = {
            "groups": groups,
            "all_parameters_dtensor_after_forward": all_dtensor,
            "resident_numel_sum": int(local_resident_numel.item()),
            "global_numel": global_numel,
            "clip_norm_matches_global_shards": bool(
                torch.allclose(clip_norm.float(), expected_norm, atol=1e-5, rtol=1e-5)
            ),
            "clip_norm_equal_on_all_ranks": all(
                torch.equal(gathered_norms[0], rank_norm) for rank_norm in gathered_norms[1:]
            ),
            "rollout_parameters_are_unsharded_inside_context": rollout_unsharded,
            "rollout_parameters_are_resharded_after_context": rollout_resharded,
        }
    finally:
        destroy_distributed()
        process_group_destroyed = not dist.is_initialized()

    if ctx.rank == 0:
        payload["process_group_destroyed"] = process_group_destroyed
        args.result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
