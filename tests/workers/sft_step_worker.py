from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.tiny_qwen import build_tiny_qwen3
from train.fsdp_sft import assert_adamw_moment_dtype, sft_optimizer_step
from train.fsdp_utils import FSDPSettings, destroy_distributed, fully_shard_qwen, init_distributed
from train.gsm8k_data import TokenBatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()
    ctx = init_distributed(timeout_seconds=120)
    try:
        model = build_tiny_qwen3().to(torch.bfloat16)
        fully_shard_qwen(model, ctx, FSDPSettings())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        input_ids = torch.tensor([[2, 4 + ctx.rank, 5 + ctx.rank, 3]], device=ctx.device)
        labels = input_ids.clone()
        labels[:, 0] = -100
        if ctx.rank == 1:
            labels[:, 1] = -100
        batch = TokenBatch(
            input_ids=input_ids,
            labels=labels,
            attention_mask=torch.ones_like(input_ids),
            supervised_tokens=int(labels.ne(-100).sum().item()),
        )

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=batch.attention_mask).logits
            local_loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        local_tokens = labels[:, 1:].ne(-100).sum()
        expected = torch.stack([local_loss.double(), local_tokens.double()])
        dist.all_reduce(expected)
        expected_mean = expected[0] / expected[1]

        metrics = sft_optimizer_step(
            model,
            optimizer,
            scheduler,
            [batch],
            ctx=ctx,
            max_grad_norm=100.0,
            accumulation_sync="reduce_scatter",
        )
        assert_adamw_moment_dtype(optimizer, torch.bfloat16)
        metric_tensor = torch.tensor(
            [metrics.loss, float(metrics.global_step_tokens)],
            device=ctx.device,
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(metric_tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, metric_tensor)
        payload = {
            "reported_loss_matches_independent_global_mean": abs(
                metrics.loss - expected_mean.item()
            )
            < 1e-6,
            "metrics_equal_on_all_ranks": all(
                torch.equal(gathered[0], item) for item in gathered[1:]
            ),
            "adamw_moments_bfloat16": True,
        }
    finally:
        destroy_distributed()

    if ctx.rank == 0:
        args.result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
