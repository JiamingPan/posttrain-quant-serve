from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.consolidate_dcp import consolidate_checkpoint
from tests.tiny_qwen import build_tiny_qwen3, write_tiny_qwen3
from train.checkpointing import TrainProgress, save_dcp_checkpoint
from train.fsdp_utils import FSDPSettings, destroy_distributed, fully_shard_qwen, init_distributed
from train.gsm8k_data import CheckpointableDistributedSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ctx = init_distributed(timeout_seconds=180)
    try:
        source = args.output_dir / "source-hf"
        if ctx.rank == 0:
            write_tiny_qwen3(source)
        torch.distributed.barrier()

        model = build_tiny_qwen3().to(device=ctx.device, dtype=torch.bfloat16)
        fully_shard_qwen(model, ctx, FSDPSettings())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        sampler = CheckpointableDistributedSampler(
            8,
            rank=ctx.rank,
            world_size=ctx.world_size,
            seed=7,
        )
        input_ids = torch.tensor([[2, 4, 5, 3]], device=ctx.device)
        model(input_ids=input_ids, labels=input_ids, use_cache=False).loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        run_dir = args.output_dir / "training"
        checkpoint = save_dcp_checkpoint(
            run_dir / "checkpoints",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            progress=TrainProgress(
                global_step=1,
                consumed_tokens=4 * ctx.world_size,
                sampler_state=sampler.state_dict(),
                rng_states=[],
                config={"stage": "sft"},
                source_digests={"model": "tiny", "data": "fixed"},
            ),
        )
        if ctx.rank == 0:
            (run_dir / "run_config.json").write_text(
                json.dumps({"model": str(source), "resolved_revision": None}),
                encoding="utf-8",
            )
        torch.distributed.barrier()
        del optimizer, scheduler, model
        torch.cuda.empty_cache()

        destination = args.output_dir / "consolidated-hf"
        manifest = consolidate_checkpoint(
            checkpoint=checkpoint,
            output_dir=destination,
            ctx=ctx,
            max_shard_size="5GB",
        )
        payload = manifest if ctx.rank == 0 else {}
    finally:
        destroy_distributed()
    if ctx.rank == 0:
        args.result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
