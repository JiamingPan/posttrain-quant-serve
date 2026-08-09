"""Full-parameter Qwen SFT using direct PyTorch FSDP2 and sharded DCP."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Literal, Protocol, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from train.checkpointing import (
    HFCheckpointSource,
    TrainProgress,
    load_dcp_checkpoint,
    load_hf_weights_into_shards,
    resolve_hf_checkpoint_source,
    resolve_resume_checkpoint,
    save_dcp_checkpoint,
)
from train.fsdp_utils import (
    DistContext,
    FSDPSettings,
    apply_qwen_activation_checkpointing,
    clip_global_grad_norm_,
    destroy_distributed,
    fully_shard_qwen,
    init_distributed,
)
from train.gsm8k_data import (
    CheckpointableDistributedSampler,
    SFTFeature,
    TokenBatch,
    build_sft_feature,
    collate_token_batches,
    pack_sft_features,
)
from train.memory_model import assert_memory_fits, predict_sft_peak
from train.run_tracking import append_run_record, capture_run_identity, write_run_config


AccumulationSync = Literal["reduce_scatter", "no_sync"]
QWEN3_8B_PARAMETERS = 8_190_735_360


class Scheduler(Protocol):
    def step(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class StepContext(Protocol):
    world_size: int
    device: torch.device


@dataclass(frozen=True)
class SFTConfig:
    output_dir: str
    model: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    sharding: Literal["fsdp2", "none"] = "fsdp2"
    resident_precision: Literal["bf16", "fp32"] = "bf16"
    activation_checkpointing: bool = True
    accumulation_sync: AccumulationSync = "reduce_scatter"
    resume: str = "none"
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    dataset_limit: int | None = None
    packing: bool = True
    sequence_length: int = 1024
    local_microbatch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 100
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 50
    attention_backend: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        positive_ints = {
            "sequence_length": self.sequence_length,
            "local_microbatch_size": self.local_microbatch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "timeout_seconds": self.timeout_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dataset_limit is not None and self.dataset_limit <= 0:
            raise ValueError("dataset_limit must be positive when supplied")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")


@dataclass(frozen=True)
class SFTStepMetrics:
    loss: float
    global_loss_sum: float
    global_step_tokens: int
    preclip_grad_norm: float
    clipped: bool
    elapsed_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    accumulation_sync: AccumulationSync


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision")
    parser.add_argument("--sharding", choices=("fsdp2", "none"), default="fsdp2")
    parser.add_argument(
        "--resident_precision",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument(
        "--activation_checkpointing",
        dest="activation_checkpointing",
        action="store_true",
    )
    checkpointing.add_argument(
        "--no_activation_checkpointing",
        dest="activation_checkpointing",
        action="store_false",
    )
    parser.set_defaults(activation_checkpointing=True)
    parser.add_argument(
        "--accumulation_sync",
        choices=("reduce_scatter", "no_sync"),
        default="reduce_scatter",
    )
    parser.add_argument("--resume", default="none")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_limit", type=int)
    packing = parser.add_mutually_exclusive_group()
    packing.add_argument("--packing", dest="packing", action="store_true")
    packing.add_argument("--no_packing", dest="packing", action="store_false")
    parser.set_defaults(packing=True)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--local_microbatch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument(
        "--attention_backend",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--timeout_seconds", type=int, default=600)
    return parser


def parse_sft_args(argv: Sequence[str] | None = None) -> SFTConfig:
    return SFTConfig(**vars(_build_parser().parse_args(argv)))


def backward_scale(*, world_size: int, global_step_tokens: int) -> float:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if global_step_tokens <= 0:
        raise ValueError("global_step_tokens must be positive")
    return world_size / global_step_tokens


def prepare_step_batches(
    features: Sequence[SFTFeature],
    sampler: CheckpointableDistributedSampler,
    *,
    local_microbatch_size: int,
    gradient_accumulation_steps: int,
    pad_token_id: int,
    pad_to_length: int | None = None,
) -> list[TokenBatch]:
    if local_microbatch_size <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("microbatch size and accumulation steps must be positive")
    batches: list[TokenBatch] = []
    for _ in range(gradient_accumulation_steps):
        indices = sampler.next_indices(local_microbatch_size)
        if len(indices) != local_microbatch_size:
            raise RuntimeError("distributed sampler returned a partial local microbatch")
        batches.append(
            collate_token_batches(
                [features[index] for index in indices],
                pad_token_id=pad_token_id,
                pad_to_length=pad_to_length,
            )
        )
    return batches


def _distributed_sum(value: torch.Tensor, *, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("world_size > 1 requires an initialized process group")
    if dist.get_world_size() != world_size:
        raise RuntimeError("step context world size does not match the process group")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _scalar_tensor_value(value: torch.Tensor) -> float:
    full_tensor = getattr(value, "full_tensor", None)
    if callable(full_tensor):
        value = full_tensor()
    return float(value.detach().float().item())


def sft_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler,
    microbatches: Sequence[TokenBatch],
    *,
    ctx: StepContext,
    max_grad_norm: float,
    accumulation_sync: AccumulationSync,
) -> SFTStepMetrics:
    """Run one token-normalized optimizer step on sharded or plain parameters."""

    if not microbatches:
        raise ValueError("an optimizer step requires at least one microbatch")
    if accumulation_sync not in {"reduce_scatter", "no_sync"}:
        raise ValueError("accumulation_sync must be 'reduce_scatter' or 'no_sync'")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if accumulation_sync == "no_sync" and not hasattr(
        model,
        "set_requires_gradient_sync",
    ):
        raise TypeError("no_sync accumulation requires an FSDP2 model")

    local_tokens = sum(
        int(batch.labels[:, 1:].ne(-100).sum().item()) for batch in microbatches
    )
    global_tokens_tensor = torch.tensor(
        local_tokens,
        device=ctx.device,
        dtype=torch.int64,
    )
    _distributed_sum(global_tokens_tensor, world_size=ctx.world_size)
    global_tokens = int(global_tokens_tensor.item())
    scale = backward_scale(
        world_size=ctx.world_size,
        global_step_tokens=global_tokens,
    )

    is_cuda = ctx.device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(ctx.device)
        torch.cuda.reset_peak_memory_stats(ctx.device)
    started = time.perf_counter()
    local_loss_sum_for_record = torch.zeros(
        (),
        device=ctx.device,
        dtype=torch.float64,
    )
    model.train()
    sync_setter = getattr(model, "set_requires_gradient_sync", None)
    try:
        for index, batch in enumerate(microbatches):
            if accumulation_sync == "no_sync":
                sync_setter(index == len(microbatches) - 1)
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            ).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = batch.labels[:, 1:]
            local_loss_sum = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            local_loss_sum_for_record += local_loss_sum.detach().double()
            (local_loss_sum * scale).backward()
    finally:
        if accumulation_sync == "no_sync":
            sync_setter(True)

    preclip_norm = clip_global_grad_norm_(model, max_grad_norm)
    preclip_norm_value = _scalar_tensor_value(preclip_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    global_loss_sum_tensor = local_loss_sum_for_record.clone()
    _distributed_sum(global_loss_sum_tensor, world_size=ctx.world_size)
    if is_cuda:
        torch.cuda.synchronize(ctx.device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(ctx.device) if is_cuda else 0
    peak_reserved = torch.cuda.max_memory_reserved(ctx.device) if is_cuda else 0
    global_loss_sum = float(global_loss_sum_tensor.item())
    return SFTStepMetrics(
        loss=global_loss_sum / global_tokens,
        global_loss_sum=global_loss_sum,
        global_step_tokens=global_tokens,
        preclip_grad_norm=preclip_norm_value,
        clipped=preclip_norm_value > max_grad_norm,
        elapsed_seconds=elapsed,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        accumulation_sync=accumulation_sync,
    )


def assert_adamw_moment_dtype(
    optimizer: torch.optim.Optimizer,
    expected_dtype: torch.dtype,
) -> None:
    moment_count = 0
    for state in optimizer.state.values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = state.get(name)
            if value is None:
                continue
            moment_count += 1
            if value.dtype is not expected_dtype:
                raise RuntimeError(
                    f"AdamW {name} has dtype {value.dtype}; expected {expected_dtype}"
                )
    if moment_count == 0:
        raise RuntimeError("AdamW moment state was not initialized after the optimizer step")


def resolve_sft_source(
    model: str | Path,
    revision: str | None,
) -> HFCheckpointSource:
    """Resolve one immutable model source for both fresh load and resume."""

    return resolve_hf_checkpoint_source(model, revision=revision)


def _dataset_digest(rows: Sequence[MappingLike]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["question"]).encode())
        digest.update(b"\0")
        digest.update(str(row["answer"]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


class MappingLike(Protocol):
    def __getitem__(self, key: str) -> Any: ...


def _checkpoint_config(config: SFTConfig) -> dict[str, Any]:
    payload = asdict(config)
    for operational_key in (
        "output_dir",
        "resume",
        "max_steps",
        "logging_steps",
        "save_steps",
        "timeout_seconds",
    ):
        payload.pop(operational_key)
    return payload


def _lr_multiplier(step: int, *, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    decay_steps = max(max_steps - warmup_steps, 1)
    return max((max_steps - step) / decay_steps, 0.0)


def _broadcast_identity(config: dict[str, Any], ctx: DistContext) -> dict[str, Any]:
    value: list[dict[str, Any] | None] = [
        capture_run_identity("sft", config).as_dict() if ctx.rank == 0 else None
    ]
    if ctx.world_size > 1:
        dist.broadcast_object_list(value, src=0)
    if value[0] is None:
        raise RuntimeError("rank zero did not publish the run identity")
    return value[0]


def _load_sft_features(config: SFTConfig, tokenizer: Any) -> tuple[list[SFTFeature], str]:
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset,
        config.dataset_config,
        split=config.split,
    )
    if config.dataset_limit is not None:
        dataset = dataset.select(range(min(config.dataset_limit, len(dataset))))
    rows = [dataset[index] for index in range(len(dataset))]
    features = [
        build_sft_feature(
            tokenizer,
            str(row["question"]),
            str(row["answer"]),
            max_length=config.sequence_length,
        )
        for row in rows
    ]
    if config.packing:
        features = pack_sft_features(
            features,
            sequence_length=config.sequence_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            pad_final=True,
        )
    if not features:
        raise ValueError("the selected dataset produced no SFT features")
    return features, _dataset_digest(rows)


def _preflight_target_model(config: SFTConfig, ctx: DistContext) -> None:
    if config.model != "Qwen/Qwen3-8B":
        return
    prediction = predict_sft_peak(
        QWEN3_8B_PARAMETERS,
        ctx.world_size,
        checkpointing=config.activation_checkpointing,
    )
    capacity_gib = torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3
    assert_memory_fits(
        prediction,
        capacity_gib=capacity_gib,
        model_name=config.model,
        state_precision=config.resident_precision,
    )


def run_sft(config: SFTConfig) -> None:
    """Run full-parameter SFT; all heavyweight dependencies are loaded lazily."""

    ctx = init_distributed(timeout_seconds=config.timeout_seconds)
    try:
        if config.sharding == "none" and ctx.world_size != 1:
            raise ValueError("unsharded SFT is accepted only at world size one")
        if config.accumulation_sync == "no_sync" and config.sharding != "fsdp2":
            raise ValueError("no_sync accumulation is only available with FSDP2")
        _preflight_target_model(config, ctx)
        random.seed(config.seed + ctx.rank)
        np.random.seed(config.seed + ctx.rank)
        torch.manual_seed(config.seed + ctx.rank)
        torch.cuda.manual_seed(config.seed + ctx.rank)

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_source = resolve_sft_source(config.model, config.revision)
        snapshot = model_source.path
        resolved_revision = model_source.revision
        tokenizer = AutoTokenizer.from_pretrained(snapshot)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        features, data_digest = _load_sft_features(config, tokenizer)
        model_config = AutoConfig.from_pretrained(snapshot)
        model_config.use_cache = False
        resident_dtype = (
            torch.bfloat16 if config.resident_precision == "bf16" else torch.float32
        )
        checkpoint_root = Path(config.output_dir) / "checkpoints"
        resume_checkpoint = resolve_resume_checkpoint(
            config.resume,
            output_dir=checkpoint_root,
        )

        if config.sharding == "fsdp2":
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    model_config,
                    torch_dtype=resident_dtype,
                    attn_implementation=config.attention_backend,
                )
            apply_qwen_activation_checkpointing(
                model,
                enabled=config.activation_checkpointing,
            )
            fully_shard_qwen(model, ctx, FSDPSettings())
            if resume_checkpoint is None:
                source = load_hf_weights_into_shards(
                    model,
                    snapshot,
                    device=ctx.device,
                    revision=resolved_revision,
                )
                model_digest = source.revision
            else:
                model.to_empty(device=ctx.device)
                model_digest = model_source.revision
        else:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                torch_dtype=resident_dtype,
                attn_implementation=config.attention_backend,
            ).to(ctx.device)
            apply_qwen_activation_checkpointing(
                model,
                enabled=config.activation_checkpointing,
            )
            model_digest = model_source.revision

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
        warmup_steps = int(config.max_steps * config.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _lr_multiplier(
                step,
                max_steps=config.max_steps,
                warmup_steps=warmup_steps,
            ),
        )
        sampler = CheckpointableDistributedSampler(
            len(features),
            rank=ctx.rank,
            world_size=ctx.world_size,
            seed=config.seed,
            shuffle=True,
        )
        semantic_config = _checkpoint_config(config)
        source_digests = {"model": model_digest, "data": data_digest}
        progress = TrainProgress(
            global_step=0,
            consumed_tokens=0,
            sampler_state=sampler.state_dict(),
            rng_states=[],
            config=semantic_config,
            source_digests=source_digests,
        )
        if resume_checkpoint is not None:
            progress = load_dcp_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                expected_config=semantic_config,
                expected_source_digests=source_digests,
            )

        resolved_config = {
            **asdict(config),
            "resolved_revision": resolved_revision,
            "world_size": ctx.world_size,
            "global_batch_size": (
                config.local_microbatch_size
                * config.gradient_accumulation_steps
                * ctx.world_size
            ),
            "model_digest": model_digest,
            "data_digest": data_digest,
        }
        identity = _broadcast_identity(resolved_config, ctx)
        if ctx.rank == 0:
            print(json.dumps(resolved_config, sort_keys=True, indent=2), flush=True)
            write_run_config(
                config.output_dir,
                resolved_config,
                filename="run_config.json",
            )

        record_fields = (
            "run_id",
            "stage",
            "global_step",
            "world_size",
            "loss",
            "global_loss_sum",
            "global_step_tokens",
            "tokens_per_sec",
            "preclip_grad_norm",
            "clipped",
            "elapsed_seconds",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "accumulation_sync",
        )
        while progress.global_step < config.max_steps:
            batches = prepare_step_batches(
                features,
                sampler,
                local_microbatch_size=config.local_microbatch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                pad_token_id=tokenizer.pad_token_id,
                pad_to_length=config.sequence_length if config.packing else None,
            )
            batches = [
                TokenBatch(
                    input_ids=batch.input_ids.to(ctx.device),
                    labels=batch.labels.to(ctx.device),
                    attention_mask=batch.attention_mask.to(ctx.device),
                    supervised_tokens=batch.supervised_tokens,
                )
                for batch in batches
            ]
            metrics = sft_optimizer_step(
                model,
                optimizer,
                scheduler,
                batches,
                ctx=ctx,
                max_grad_norm=config.max_grad_norm,
                accumulation_sync=config.accumulation_sync,
            )
            next_step = progress.global_step + 1
            progress = TrainProgress(
                global_step=next_step,
                consumed_tokens=progress.consumed_tokens + metrics.global_step_tokens,
                sampler_state=sampler.state_dict(),
                rng_states=[],
                config=semantic_config,
                source_digests=source_digests,
            )
            if next_step == 1:
                assert_adamw_moment_dtype(optimizer, resident_dtype)
            record = {
                "run_id": identity["run_id"],
                "stage": "sft",
                "global_step": next_step,
                "world_size": ctx.world_size,
                "loss": metrics.loss,
                "global_loss_sum": metrics.global_loss_sum,
                "global_step_tokens": metrics.global_step_tokens,
                "tokens_per_sec": metrics.global_step_tokens / metrics.elapsed_seconds,
                "preclip_grad_norm": metrics.preclip_grad_norm,
                "clipped": metrics.clipped,
                "elapsed_seconds": metrics.elapsed_seconds,
                "peak_allocated_bytes": metrics.peak_allocated_bytes,
                "peak_reserved_bytes": metrics.peak_reserved_bytes,
                "accumulation_sync": metrics.accumulation_sync,
            }
            if ctx.rank == 0:
                append_run_record(
                    config.output_dir,
                    "train",
                    record,
                    record_fields,
                )
                if next_step % config.logging_steps == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
            if next_step % config.save_steps == 0 or next_step == config.max_steps:
                save_dcp_checkpoint(
                    checkpoint_root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    progress=progress,
                )
    finally:
        destroy_distributed()


def main(argv: Sequence[str] | None = None) -> None:
    run_sft(parse_sft_args(argv))


if __name__ == "__main__":
    main()
