"""Full-parameter Qwen GRPO using direct PyTorch FSDP2 and sharded DCP."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Literal, Protocol, Sequence

import torch
import torch.distributed as dist
from torch import nn
import numpy as np

from train.checkpointing import (
    TrainProgress,
    checkpoint_manifest,
    load_dcp_checkpoint,
    load_dcp_model_only,
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
from train.gsm8k_data import CheckpointableDistributedSampler
from train.grpo_core import (
    RolloutBatch,
    RolloutConfig,
    generate_rollout_batch,
    grpo_loss_sum,
    select_token_logps,
    teacher_forced_logps,
)
from train.memory_model import assert_memory_fits, predict_grpo_peak
from train.run_tracking import append_run_record, capture_run_identity, write_run_config
from train.fsdp_sft import QWEN3_8B_PARAMETERS, assert_adamw_moment_dtype
from scripts.gsm8k_reward import build_gsm8k_chat_text


AccumulationSync = Literal["reduce_scatter", "no_sync"]


class Scheduler(Protocol):
    def step(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class StepContext(Protocol):
    world_size: int
    device: torch.device


@dataclass(frozen=True)
class GRPOConfig:
    model: str
    output_dir: str
    revision: str | None = None
    resume: str = "none"
    resident_precision: Literal["bf16", "fp32"] = "bf16"
    activation_checkpointing: bool = True
    accumulation_sync: AccumulationSync = "reduce_scatter"
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    dataset_limit: int | None = None
    num_generations: int = 8
    gradient_accumulation_steps: int = 8
    policy_microbatch_size: int = 1
    max_steps: int = 100
    learning_rate: float = 1e-6
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    loss_type: Literal["dr_grpo"] = "dr_grpo"
    scale_rewards: Literal["none", "group", "batch"] = "none"
    beta: float = 0.0
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_prompt_length: int = 512
    max_completion_length: int = 128
    rollout_mode: Literal["auto", "reshard", "keep_unsharded"] = "auto"
    seed: int = 42
    logging_steps: int = 1
    save_steps: int = 50
    attention_backend: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
        positive_ints = {
            "num_generations": self.num_generations,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "policy_microbatch_size": self.policy_microbatch_size,
            "max_steps": self.max_steps,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "timeout_seconds": self.timeout_seconds,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least two")
        if self.dataset_limit is not None and self.dataset_limit <= 0:
            raise ValueError("dataset_limit must be positive when supplied")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if not 0 <= self.epsilon_low < 1 or self.epsilon_high < 0:
            raise ValueError("invalid GRPO clipping bounds")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("invalid rollout sampling values")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")


@dataclass(frozen=True)
class PhaseMetrics:
    elapsed_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


@dataclass(frozen=True)
class GRPOStepMetrics:
    loss: float
    global_loss_sum: float
    global_policy_loss_sum: float
    global_kl_sum: float
    clip_ratio: float
    global_completion_count: int
    global_valid_tokens: int
    normalizer: int
    preclip_grad_norm: float
    clipped: bool
    policy_phase: PhaseMetrics
    optimizer_phase: PhaseMetrics
    accumulation_sync: AccumulationSync


@dataclass(frozen=True)
class PolicySource:
    kind: Literal["hf", "dcp"]
    weights_path: Path
    architecture_path: Path
    model_revision: str
    checkpoint_digest: str


def _dcp_source_digest(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for filename in ("manifest.json", ".metadata"):
        path = checkpoint / filename
        if path.is_file():
            digest.update(filename.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_policy_source(
    source: str | Path,
    *,
    revision: str | None,
) -> PolicySource:
    """Resolve either an HF directory/repo or a published SFT DCP source."""

    candidate = Path(source)
    if candidate.is_dir() and (candidate / "_SUCCESS").is_file():
        checkpoint_manifest(candidate)
        run_config_path = candidate.parent.parent / "run_config.json"
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            base_model = str(run_config["model"])
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                f"SFT DCP source is missing a valid run_config.json: {candidate}"
            ) from error
        resolved_revision = run_config.get("resolved_revision")
        if revision is not None and resolved_revision not in {None, revision}:
            raise ValueError("requested revision disagrees with the SFT DCP source")
        architecture = resolve_hf_checkpoint_source(
            base_model,
            revision=resolved_revision or revision,
        )
        return PolicySource(
            kind="dcp",
            weights_path=candidate.resolve(),
            architecture_path=architecture.path,
            model_revision=architecture.revision,
            checkpoint_digest=_dcp_source_digest(candidate),
        )

    hf_source = resolve_hf_checkpoint_source(source, revision=revision)
    return PolicySource(
        kind="hf",
        weights_path=hf_source.path,
        architecture_path=hf_source.path,
        model_revision=hf_source.revision,
        checkpoint_digest=hf_source.revision,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--resume", default="none")
    parser.add_argument("--resident_precision", choices=("bf16", "fp32"), default="bf16")
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
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_limit", type=int)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--policy_microbatch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--loss_type", choices=("dr_grpo",), default="dr_grpo")
    parser.add_argument(
        "--scale_rewards",
        choices=("none", "group", "batch"),
        default="none",
    )
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument(
        "--rollout_mode",
        choices=("auto", "reshard", "keep_unsharded"),
        default="auto",
    )
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


def parse_grpo_args(argv: Sequence[str] | None = None) -> GRPOConfig:
    return GRPOConfig(**vars(_build_parser().parse_args(argv)))


def build_memory_layout_record(*, beta: float, rollout_mode: str) -> dict[str, Any]:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if rollout_mode not in {"auto", "reshard", "keep_unsharded"}:
        raise ValueError("invalid rollout mode")
    rollout_generation = (
        "same_policy_full_bf16_replica"
        if rollout_mode == "keep_unsharded"
        else "same_policy_layerwise_all_gather"
    )
    return {
        "policy": "fsdp2_sharded_gpu",
        "reference": (
            "independent_frozen_fsdp2_shard_gpu"
            if beta > 0
            else "absent_beta_zero"
        ),
        "rollout_records": "cpu_after_group",
        "rollout_generation": rollout_generation,
        "peak_memory_implications": {
            "keep_unsharded": "one_full_bf16_policy_per_gpu_during_rollout",
            "beta_positive": "one_additional_frozen_bf16_reference_shard_per_gpu",
        },
    }


def build_reference_model(
    *,
    source: str | PolicySource,
    ctx: Any,
    attention_backend: str = "sdpa",
) -> Any:
    """Build one independently sharded, frozen bf16 SFT reference policy."""

    from transformers import AutoConfig, AutoModelForCausalLM

    resolved = source if isinstance(source, PolicySource) else resolve_policy_source(
        source,
        revision=None,
    )
    model_config = AutoConfig.from_pretrained(resolved.architecture_path)
    model_config.use_cache = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_backend,
        )
    fully_shard_qwen(model, ctx, FSDPSettings())
    if resolved.kind == "dcp":
        model.to_empty(device=ctx.device)
        load_dcp_model_only(resolved.weights_path, model=model)
    else:
        load_hf_weights_into_shards(
            model,
            resolved.weights_path,
            device=ctx.device,
            revision=resolved.model_revision,
        )
    model.requires_grad_(False)
    model.eval()
    return model


def maybe_build_reference(
    *,
    beta: float,
    source: Any,
    ctx: Any,
    **build_kwargs: Any,
) -> Any | None:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if beta == 0:
        return None
    return build_reference_model(source=source, ctx=ctx, **build_kwargs)


def _distributed_sum(value: torch.Tensor, *, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("world_size > 1 requires an initialized process group")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _phase_start(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def _phase_finish(started: float, device: torch.device) -> PhaseMetrics:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    else:
        allocated = 0
        reserved = 0
    return PhaseMetrics(
        elapsed_seconds=time.perf_counter() - started,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
    )


def _scalar_tensor_value(value: torch.Tensor) -> float:
    full_tensor = getattr(value, "full_tensor", None)
    if callable(full_tensor):
        value = full_tensor()
    return float(value.detach().float().item())


def grpo_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Scheduler,
    rollouts: Sequence[RolloutBatch],
    *,
    ctx: StepContext,
    policy_microbatch_size: int,
    max_grad_norm: float,
    epsilon_low: float,
    epsilon_high: float,
    beta: float,
    max_completion_length: int,
    accumulation_sync: AccumulationSync,
) -> GRPOStepMetrics:
    """Apply one globally normalized GRPO policy update from CPU rollouts."""

    if not rollouts:
        raise ValueError("a GRPO optimizer step requires at least one rollout")
    if policy_microbatch_size <= 0 or max_grad_norm <= 0:
        raise ValueError("microbatch size and max_grad_norm must be positive")
    if accumulation_sync not in {"reduce_scatter", "no_sync"}:
        raise ValueError("invalid accumulation_sync mode")
    if accumulation_sync == "no_sync" and not hasattr(
        model,
        "set_requires_gradient_sync",
    ):
        raise TypeError("no_sync accumulation requires an FSDP2 model")

    local_completions = sum(rollout.batch_size for rollout in rollouts)
    completion_count = torch.tensor(
        local_completions,
        dtype=torch.int64,
        device=ctx.device,
    )
    _distributed_sum(completion_count, world_size=ctx.world_size)
    global_completions = int(completion_count.item())
    normalizer = global_completions * max_completion_length
    backward_scale = ctx.world_size / normalizer

    slices: list[tuple[RolloutBatch, int, int]] = []
    for rollout in rollouts:
        for start in range(0, rollout.batch_size, policy_microbatch_size):
            slices.append(
                (rollout, start, min(start + policy_microbatch_size, rollout.batch_size))
            )

    totals = torch.zeros(5, dtype=torch.float64, device=ctx.device)
    policy_started = _phase_start(ctx.device)
    model.train()
    sync_setter = getattr(model, "set_requires_gradient_sync", None)
    try:
        for index, (rollout, start, stop) in enumerate(slices):
            if accumulation_sync == "no_sync":
                sync_setter(index == len(slices) - 1)
            prompt_ids = rollout.prompt_input_ids[start:stop].to(ctx.device)
            completion_ids = rollout.completion_input_ids[start:stop].to(ctx.device)
            input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
            attention_mask = torch.cat(
                (
                    rollout.prompt_attention_mask[start:stop].to(ctx.device),
                    rollout.completion_mask[start:stop].to(
                        device=ctx.device,
                        dtype=rollout.prompt_attention_mask.dtype,
                    ),
                ),
                dim=1,
            )
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            current_logps = select_token_logps(
                logits,
                input_ids,
                completion_length=completion_ids.size(1),
            )
            result = grpo_loss_sum(
                current_logps=current_logps,
                old_logps=rollout.old_logps[start:stop].to(ctx.device),
                ref_logps=(
                    rollout.ref_logps[start:stop].to(ctx.device)
                    if rollout.ref_logps is not None
                    else None
                ),
                advantages=rollout.advantages[start:stop].to(ctx.device),
                completion_mask=rollout.completion_mask[start:stop].to(ctx.device),
                epsilon_low=epsilon_low,
                epsilon_high=epsilon_high,
                beta=beta,
                max_completion_length=max_completion_length,
            )
            (result.loss_sum * backward_scale).backward()
            totals[0] += result.loss_sum.detach().double()
            totals[1] += result.policy_loss_sum.detach().double()
            totals[2] += result.kl_sum.detach().double()
            totals[3] += result.clip_ratio.detach().double() * result.valid_tokens
            totals[4] += result.valid_tokens
    finally:
        if accumulation_sync == "no_sync":
            sync_setter(True)
    policy_phase = _phase_finish(policy_started, ctx.device)

    optimizer_started = _phase_start(ctx.device)
    preclip_norm = clip_global_grad_norm_(model, max_grad_norm)
    preclip_norm_value = _scalar_tensor_value(preclip_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_phase = _phase_finish(optimizer_started, ctx.device)

    _distributed_sum(totals, world_size=ctx.world_size)
    global_valid_tokens = int(totals[4].item())
    global_loss_sum = float(totals[0].item())
    return GRPOStepMetrics(
        loss=global_loss_sum / normalizer,
        global_loss_sum=global_loss_sum,
        global_policy_loss_sum=float(totals[1].item()),
        global_kl_sum=float(totals[2].item()),
        clip_ratio=float(totals[3].item() / global_valid_tokens),
        global_completion_count=global_completions,
        global_valid_tokens=global_valid_tokens,
        normalizer=normalizer,
        preclip_grad_norm=preclip_norm_value,
        clipped=preclip_norm_value > max_grad_norm,
        policy_phase=policy_phase,
        optimizer_phase=optimizer_phase,
        accumulation_sync=accumulation_sync,
    )


def _checkpoint_config(config: GRPOConfig) -> dict[str, Any]:
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


def _dataset_digest(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["question"]).encode())
        digest.update(b"\0")
        digest.update(str(row["answer"]).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _broadcast_identity(config: dict[str, Any], ctx: DistContext) -> dict[str, Any]:
    value: list[dict[str, Any] | None] = [
        capture_run_identity("grpo", config).as_dict() if ctx.rank == 0 else None
    ]
    if ctx.world_size > 1:
        dist.broadcast_object_list(value, src=0)
    if value[0] is None:
        raise RuntimeError("rank zero did not publish the run identity")
    return value[0]


def _combine_phases(phases: Sequence[PhaseMetrics]) -> PhaseMetrics:
    if not phases:
        return PhaseMetrics(0.0, 0, 0)
    return PhaseMetrics(
        elapsed_seconds=sum(phase.elapsed_seconds for phase in phases),
        peak_allocated_bytes=max(phase.peak_allocated_bytes for phase in phases),
        peak_reserved_bytes=max(phase.peak_reserved_bytes for phase in phases),
    )


def _measure_phase(device: torch.device, operation: Any) -> tuple[Any, PhaseMetrics]:
    started = _phase_start(device)
    result = operation()
    return result, _phase_finish(started, device)


def tensor_moments(values: torch.Tensor) -> torch.Tensor:
    """Return sum, squared sum, and count without leaving the tensor device."""

    if values.numel() == 0:
        raise ValueError("moment calculation requires at least one value")
    return torch.stack(
        (
            values.sum(),
            values.square().sum(),
            values.new_tensor(values.numel()),
        )
    )


def _reward_statistics(
    rollouts: Sequence[RolloutBatch],
    *,
    num_generations: int,
    ctx: DistContext,
) -> dict[str, float]:
    rewards = torch.cat([rollout.rewards for rollout in rollouts]).to(
        device=ctx.device,
        dtype=torch.float64,
    )
    values = tensor_moments(rewards)
    zero_variance = sum(
        int(group.unique().numel() == 1)
        for rollout in rollouts
        for group in rollout.rewards.view(-1, num_generations)
    )
    group_counts = torch.stack(
        (
            rewards.new_tensor(zero_variance),
            rewards.new_tensor(
                sum(rollout.batch_size for rollout in rollouts) / num_generations
            ),
        )
    )
    _distributed_sum(values, world_size=ctx.world_size)
    _distributed_sum(group_counts, world_size=ctx.world_size)
    count = values[2].item()
    mean = values[0].item() / count
    variance = max(values[1].item() / count - mean * mean, 0.0)
    return {
        "reward_mean": mean,
        "reward_std": variance**0.5,
        "zero_reward_variance_group_frac": (
            group_counts[0].item() / group_counts[1].item()
        ),
    }


def _apply_global_batch_reward_scaling(
    rollouts: Sequence[RolloutBatch],
    *,
    ctx: DistContext,
) -> list[RolloutBatch]:
    rewards = torch.cat([rollout.rewards for rollout in rollouts]).to(
        device=ctx.device,
        dtype=torch.float64,
    )
    moments = tensor_moments(rewards)
    _distributed_sum(moments, world_size=ctx.world_size)
    mean = moments[0] / moments[2]
    std = (moments[1] / moments[2] - mean.square()).clamp_min(0).sqrt()
    denominator = max(float(std.item()), 1e-4)
    return [
        replace(rollout, advantages=rollout.advantages / denominator)
        for rollout in rollouts
    ]


def _completion_statistics(
    rollouts: Sequence[RolloutBatch],
    *,
    ctx: DistContext,
) -> dict[str, float | int]:
    lengths = torch.cat(
        [rollout.completion_mask.sum(dim=1) for rollout in rollouts]
    ).to(device=ctx.device, dtype=torch.float64)
    values = torch.stack(
        (
            lengths.sum(),
            lengths.new_tensor(lengths.numel()),
        )
    )
    maximum = lengths.max().to(torch.int64)
    _distributed_sum(values, world_size=ctx.world_size)
    if ctx.world_size > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {
        "completion_length_mean": values[0].item() / values[1].item(),
        "completion_length_max": int(maximum.item()),
    }


def _rank_phase_records(
    phase_record: dict[str, Any],
    *,
    ctx: DistContext,
) -> list[dict[str, Any]] | None:
    local = {"rank": ctx.rank, **phase_record}
    if ctx.world_size == 1:
        return [local]
    gathered: list[dict[str, Any] | None] | None = (
        [None] * ctx.world_size if ctx.rank == 0 else None
    )
    dist.gather_object(local, gathered, dst=0)
    if gathered is None:
        return None
    return [value for value in gathered if value is not None]


def _phase_as_dict(phase: PhaseMetrics) -> dict[str, float | int]:
    return {
        "elapsed_seconds": phase.elapsed_seconds,
        "peak_allocated_bytes": phase.peak_allocated_bytes,
        "peak_reserved_bytes": phase.peak_reserved_bytes,
    }


def _load_dataset_rows(config: GRPOConfig) -> tuple[list[dict[str, Any]], str]:
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset,
        config.dataset_config,
        split=config.split,
    )
    if config.dataset_limit is not None:
        dataset = dataset.select(range(min(config.dataset_limit, len(dataset))))
    rows = [dict(dataset[index]) for index in range(len(dataset))]
    if not rows:
        raise ValueError("the selected dataset contains no GRPO prompts")
    return rows, _dataset_digest(rows)


def _build_sharded_policy(
    source: PolicySource,
    *,
    config: GRPOConfig,
    ctx: DistContext,
    resume_checkpoint: Path | None,
) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(source.architecture_path)
    model_config.use_cache = False
    resident_dtype = (
        torch.bfloat16 if config.resident_precision == "bf16" else torch.float32
    )
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
    if resume_checkpoint is not None:
        model.to_empty(device=ctx.device)
    elif source.kind == "dcp":
        model.to_empty(device=ctx.device)
        load_dcp_model_only(source.weights_path, model=model)
    else:
        load_hf_weights_into_shards(
            model,
            source.weights_path,
            device=ctx.device,
            revision=source.model_revision,
        )
    return model


def _rollout_preflight(
    config: GRPOConfig,
    source: PolicySource,
    ctx: DistContext,
) -> tuple[str, float | None, float]:
    from transformers import AutoConfig
    from train.grpo_core import choose_rollout_mode

    capacity_gib = torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3
    model_config = AutoConfig.from_pretrained(source.architecture_path)
    is_qwen3_8b = (
        getattr(model_config, "num_hidden_layers", None) == 36
        and getattr(model_config, "hidden_size", None) == 4096
    )
    if not is_qwen3_8b:
        mode = "reshard" if config.rollout_mode == "auto" else config.rollout_mode
        return mode, None, capacity_gib

    keep_prediction = predict_grpo_peak(
        QWEN3_8B_PARAMETERS,
        world_size=ctx.world_size,
        beta=config.beta,
        rollout_mode="keep_unsharded",
    )
    requested = config.rollout_mode
    mode = choose_rollout_mode(
        predicted_keep_unsharded_gib=keep_prediction.reserved_gib,
        capacity_gib=capacity_gib,
        requested=requested,
    )
    selected_prediction = predict_grpo_peak(
        QWEN3_8B_PARAMETERS,
        world_size=ctx.world_size,
        beta=config.beta,
        rollout_mode=mode,
    )
    assert_memory_fits(
        selected_prediction,
        capacity_gib=capacity_gib,
        model_name="Qwen/Qwen3-8B",
        state_precision=config.resident_precision,
    )
    return mode, keep_prediction.reserved_gib, capacity_gib


def run_grpo(config: GRPOConfig) -> None:
    """Run phase-separated FSDP2 GRPO from an immutable SFT source."""

    ctx = init_distributed(timeout_seconds=config.timeout_seconds)
    try:
        random.seed(config.seed + ctx.rank)
        np.random.seed(config.seed + ctx.rank)
        torch.manual_seed(config.seed + ctx.rank)
        torch.cuda.manual_seed(config.seed + ctx.rank)

        from transformers import AutoTokenizer

        source = resolve_policy_source(config.model, revision=config.revision)
        tokenizer = AutoTokenizer.from_pretrained(source.architecture_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        rows, data_digest = _load_dataset_rows(config)
        prompts = [
            build_gsm8k_chat_text(tokenizer, str(row["question"])) for row in rows
        ]
        answers = [str(row["answer"]) for row in rows]
        checkpoint_root = Path(config.output_dir) / "checkpoints"
        resume_checkpoint = resolve_resume_checkpoint(
            config.resume,
            output_dir=checkpoint_root,
        )
        resolved_rollout_mode, keep_peak_gib, capacity_gib = _rollout_preflight(
            config,
            source,
            ctx,
        )
        policy = _build_sharded_policy(
            source,
            config=config,
            ctx=ctx,
            resume_checkpoint=resume_checkpoint,
        )
        reference = maybe_build_reference(
            beta=config.beta,
            source=source,
            ctx=ctx,
            attention_backend=config.attention_backend,
        )
        optimizer = torch.optim.AdamW(
            policy.parameters(),
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
            len(rows),
            rank=ctx.rank,
            world_size=ctx.world_size,
            seed=config.seed,
            shuffle=True,
        )
        semantic_config = _checkpoint_config(config)
        source_digests = {
            "policy_init": source.checkpoint_digest,
            "data": data_digest,
            "reference": source.checkpoint_digest if config.beta > 0 else "absent",
        }
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
                model=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                expected_config=semantic_config,
                expected_source_digests=source_digests,
            )

        resolved_config = {
            **asdict(config),
            "world_size": ctx.world_size,
            "source_kind": source.kind,
            "source_weights_path": str(source.weights_path),
            "source_architecture_path": str(source.architecture_path),
            "model_revision": source.model_revision,
            "policy_init_digest": source.checkpoint_digest,
            "data_digest": data_digest,
            "resolved_rollout_mode": resolved_rollout_mode,
            "memory_layout": build_memory_layout_record(
                beta=config.beta,
                rollout_mode=resolved_rollout_mode,
            ),
        }
        identity = _broadcast_identity(resolved_config, ctx)
        if ctx.rank == 0:
            print(json.dumps(resolved_config, sort_keys=True, indent=2), flush=True)
            write_run_config(
                config.output_dir,
                resolved_config,
                filename="run_config.json",
            )

        rollout_config = RolloutConfig(
            num_generations=config.num_generations,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            scale_rewards="none" if config.scale_rewards == "batch" else config.scale_rewards,
            rollout_mode=resolved_rollout_mode,
            teacher_forcing_microbatch_size=config.policy_microbatch_size,
        )
        record_fields = (
            "run_id",
            "stage",
            "global_step",
            "world_size",
            "loss",
            "global_loss_sum",
            "policy_loss_sum",
            "kl_sum",
            "clip_ratio",
            "global_completion_count",
            "global_valid_tokens",
            "normalizer",
            "reward_mean",
            "reward_std",
            "zero_reward_variance_group_frac",
            "completion_length_mean",
            "completion_length_max",
            "preclip_grad_norm",
            "clipped",
            "accumulation_sync",
            "memory_layout",
            "rank_phase_metrics",
        )
        while progress.global_step < config.max_steps:
            rollouts: list[RolloutBatch] = []
            rollout_phases: list[PhaseMetrics] = []
            reference_phases: list[PhaseMetrics] = []
            for _ in range(config.gradient_accumulation_steps):
                row_index = sampler.next_indices(1)[0]
                rollout, rollout_phase = _measure_phase(
                    ctx.device,
                    lambda row_index=row_index: generate_rollout_batch(
                        policy,
                        tokenizer,
                        prompt_texts=[prompts[row_index]],
                        answers=[answers[row_index]],
                        config=rollout_config,
                        ctx=ctx,
                        reference=None,
                        predicted_keep_unsharded_gib=keep_peak_gib,
                        capacity_gib=capacity_gib,
                    ),
                )
                rollout_phases.append(rollout_phase)
                if reference is not None:
                    ref_logps, reference_phase = _measure_phase(
                        ctx.device,
                        lambda rollout=rollout: teacher_forced_logps(
                            reference,
                            prompt_input_ids=rollout.prompt_input_ids,
                            prompt_attention_mask=rollout.prompt_attention_mask,
                            completion_input_ids=rollout.completion_input_ids,
                            completion_mask=rollout.completion_mask,
                            device=ctx.device,
                            microbatch_size=config.policy_microbatch_size,
                        ),
                    )
                    rollout = replace(rollout, ref_logps=ref_logps)
                    reference_phases.append(reference_phase)
                rollouts.append(rollout)

            if config.scale_rewards == "batch":
                rollouts = _apply_global_batch_reward_scaling(rollouts, ctx=ctx)
            reward_stats = _reward_statistics(
                rollouts,
                num_generations=config.num_generations,
                ctx=ctx,
            )
            completion_stats = _completion_statistics(rollouts, ctx=ctx)
            metrics = grpo_optimizer_step(
                policy,
                optimizer,
                scheduler,
                rollouts,
                ctx=ctx,
                policy_microbatch_size=config.policy_microbatch_size,
                max_grad_norm=config.max_grad_norm,
                epsilon_low=config.epsilon_low,
                epsilon_high=config.epsilon_high,
                beta=config.beta,
                max_completion_length=config.max_completion_length,
                accumulation_sync=config.accumulation_sync,
            )
            next_step = progress.global_step + 1
            progress = TrainProgress(
                global_step=next_step,
                consumed_tokens=progress.consumed_tokens + metrics.global_valid_tokens,
                sampler_state=sampler.state_dict(),
                rng_states=[],
                config=semantic_config,
                source_digests=source_digests,
            )
            if next_step == 1:
                expected_dtype = (
                    torch.bfloat16
                    if config.resident_precision == "bf16"
                    else torch.float32
                )
                assert_adamw_moment_dtype(optimizer, expected_dtype)

            checkpoint_phase = PhaseMetrics(0.0, 0, 0)
            if next_step % config.save_steps == 0 or next_step == config.max_steps:
                _, checkpoint_phase = _measure_phase(
                    ctx.device,
                    lambda: save_dcp_checkpoint(
                        checkpoint_root,
                        model=policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=sampler,
                        progress=progress,
                    ),
                )
            phase_record = {
                "rollout": _phase_as_dict(_combine_phases(rollout_phases)),
                "reference": _phase_as_dict(_combine_phases(reference_phases)),
                "policy_forward_backward": _phase_as_dict(metrics.policy_phase),
                "optimizer": _phase_as_dict(metrics.optimizer_phase),
                "checkpoint": _phase_as_dict(checkpoint_phase),
            }
            rank_phases = _rank_phase_records(phase_record, ctx=ctx)
            if ctx.rank == 0:
                record = {
                    "run_id": identity["run_id"],
                    "stage": "grpo",
                    "global_step": next_step,
                    "world_size": ctx.world_size,
                    "loss": metrics.loss,
                    "global_loss_sum": metrics.global_loss_sum,
                    "policy_loss_sum": metrics.global_policy_loss_sum,
                    "kl_sum": metrics.global_kl_sum,
                    "clip_ratio": metrics.clip_ratio,
                    "global_completion_count": metrics.global_completion_count,
                    "global_valid_tokens": metrics.global_valid_tokens,
                    "normalizer": metrics.normalizer,
                    **reward_stats,
                    **completion_stats,
                    "preclip_grad_norm": metrics.preclip_grad_norm,
                    "clipped": metrics.clipped,
                    "accumulation_sync": metrics.accumulation_sync,
                    "memory_layout": resolved_config["memory_layout"],
                    "rank_phase_metrics": rank_phases,
                }
                append_run_record(
                    config.output_dir,
                    "train",
                    record,
                    record_fields,
                )
                if next_step % config.logging_steps == 0:
                    print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        destroy_distributed()


def main(argv: Sequence[str] | None = None) -> None:
    run_grpo(parse_grpo_args(argv))


if __name__ == "__main__":
    main()
