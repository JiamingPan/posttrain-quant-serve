from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import get_state_dict

from tests.tiny_qwen import build_tiny_qwen3
from train.checkpointing import (
    TrainProgress,
    load_hf_weights_into_shards,
    load_dcp_checkpoint,
    resolve_resume_checkpoint,
    save_dcp_checkpoint,
)
from train.fsdp_utils import (
    FSDPSettings,
    destroy_distributed,
    fully_shard_qwen,
    init_distributed,
)
from train.gsm8k_data import CheckpointableDistributedSampler


def _build_training_state(ctx):
    model = build_tiny_qwen3()
    fully_shard_qwen(model, ctx, FSDPSettings())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.95**step)
    sampler = CheckpointableDistributedSampler(
        32,
        rank=ctx.rank,
        world_size=ctx.world_size,
        seed=23,
        shuffle=False,
    )
    return model, optimizer, scheduler, sampler


def _step(model, optimizer, scheduler, indices, device) -> float:
    input_ids = torch.tensor(
        [[2, 4 + (index % 20), 5 + (index % 20), 3] for index in indices],
        device=device,
    )
    loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().float().item())


def _optimizer_state_keys(model, optimizer) -> list[str]:
    _, optimizer_state = get_state_dict(model, optimizer)
    return sorted(optimizer_state["state"])


def _save_once(ctx, output_dir: Path):
    model, optimizer, scheduler, sampler = _build_training_state(ctx)
    _step(model, optimizer, scheduler, sampler.next_indices(2), ctx.device)
    progress = TrainProgress(
        global_step=1,
        consumed_tokens=8 * ctx.world_size,
        sampler_state=sampler.state_dict(),
        rng_states=[],
        config={"model": "tiny-qwen3", "seed": 23},
        source_digests={"model": "tiny-qwen3-seed-0", "data": "indices-0-31"},
    )
    checkpoint = save_dcp_checkpoint(
        output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        progress=progress,
    )
    return checkpoint, model, optimizer, scheduler, sampler, progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("exact", "save", "load-reshard", "hf-load"),
        required=True,
    )
    parser.add_argument("--hf-model-dir", type=Path)
    args = parser.parse_args()
    ctx = init_distributed(timeout_seconds=120)
    payload: dict[str, object]
    try:
        if args.mode in {"exact", "save"}:
            checkpoint, model, optimizer, scheduler, sampler, progress = _save_once(
                ctx,
                args.output_dir,
            )
            if args.mode == "save":
                payload = {"checkpoint": str(checkpoint)}
            else:
                optimizer_keys_before = _optimizer_state_keys(model, optimizer)
                sampler_cursor_before = sampler.global_cursor
                next_indices = sampler.next_indices(2)
                next_loss_uninterrupted = _step(
                    model,
                    optimizer,
                    scheduler,
                    next_indices,
                    ctx.device,
                )

                resumed_model, resumed_optimizer, resumed_scheduler, resumed_sampler = (
                    _build_training_state(ctx)
                )
                loaded = load_dcp_checkpoint(
                    checkpoint,
                    model=resumed_model,
                    optimizer=resumed_optimizer,
                    scheduler=resumed_scheduler,
                    sampler=resumed_sampler,
                    expected_config=progress.config,
                    expected_source_digests=progress.source_digests,
                )
                resumed_indices = resumed_sampler.next_indices(2)
                next_loss_resumed = _step(
                    resumed_model,
                    resumed_optimizer,
                    resumed_scheduler,
                    resumed_indices,
                    ctx.device,
                )
                partial = args.output_dir / "step-99999999"
                if ctx.rank == 0:
                    partial.mkdir(exist_ok=True)
                    (partial / "manifest.json").write_text("{}", encoding="utf-8")
                torch.distributed.barrier()
                selected = resolve_resume_checkpoint("latest", output_dir=args.output_dir)
                payload = {
                    "success_marker_present": (checkpoint / "_SUCCESS").is_file(),
                    "partial_directory_selected": selected == partial,
                    "next_loss_uninterrupted": next_loss_uninterrupted,
                    "next_loss_resumed": next_loss_resumed,
                    "optimizer_state_keys_before": optimizer_keys_before,
                    "optimizer_state_keys_after": _optimizer_state_keys(
                        resumed_model,
                        resumed_optimizer,
                    ),
                    "sampler_cursor_before": sampler_cursor_before,
                    "sampler_cursor_after": loaded.sampler_state["global_cursor"],
                    "bitwise_resume": loaded.bitwise_resume,
                }
        elif args.mode == "load-reshard":
            checkpoint = resolve_resume_checkpoint("latest", output_dir=args.output_dir)
            assert checkpoint is not None
            model, optimizer, scheduler, sampler = _build_training_state(ctx)
            loaded = load_dcp_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                expected_config={"model": "tiny-qwen3", "seed": 23},
                expected_source_digests={
                    "model": "tiny-qwen3-seed-0",
                    "data": "indices-0-31",
                },
                allow_world_size_change=True,
            )
            indices = sampler.next_indices(2)
            input_ids = torch.tensor(
                [[2, 4 + (index % 20), 5 + (index % 20), 3] for index in indices],
                device=ctx.device,
            )
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            payload = {
                "model_load_finite": bool(torch.isfinite(loss).item()),
                "optimizer_state_restored": bool(_optimizer_state_keys(model, optimizer)),
                "bitwise_resume": loaded.bitwise_resume,
            }
        else:
            if args.hf_model_dir is None:
                raise ValueError("--hf-model-dir is required for hf-load mode")
            from torch.distributed.tensor import DTensor
            from transformers import Qwen3ForCausalLM

            reference = build_tiny_qwen3()
            expected_sum = reference.model.embed_tokens.weight.float().sum().item()
            del reference
            from tests.tiny_qwen import tiny_qwen3_config

            with torch.device("meta"):
                model = Qwen3ForCausalLM(tiny_qwen3_config())
            fully_shard_qwen(model, ctx, FSDPSettings())
            source = load_hf_weights_into_shards(
                model,
                args.hf_model_dir,
                device=ctx.device,
            )
            all_dtensor = all(
                isinstance(parameter, DTensor) for parameter in model.parameters()
            )
            actual_sum = (
                model.model.embed_tokens.weight.full_tensor().float().sum().item()
            )
            payload = {
                "all_parameters_dtensor": all_dtensor,
                "loaded_weight_matches_source": abs(actual_sum - expected_sum) < 1e-5,
                "resolved_revision": source.revision,
            }
    finally:
        destroy_distributed()

    if ctx.rank == 0:
        args.result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
