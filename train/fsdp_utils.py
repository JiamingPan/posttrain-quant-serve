"""Direct PyTorch FSDP2 utilities shared by SFT, GRPO, and benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import Iterator, Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

try:
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
except ImportError:  # PyTorch 2.5 local test compatibility; still the FSDP2 API.
    from torch.distributed._composable.fsdp import (  # type: ignore[no-redef]
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )


RolloutMode = Literal["reshard", "keep_unsharded"]


@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    mesh: DeviceMesh


@dataclass(frozen=True)
class FSDPSettings:
    """The homogeneous bf16 FSDP2 policy used by the scaling track."""

    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype = torch.bfloat16
    reshard_after_forward: bool = True

    def __post_init__(self) -> None:
        if self.param_dtype is not torch.bfloat16:
            raise ValueError("param_dtype must be torch.bfloat16 for this FSDP2 track")
        if self.reduce_dtype is not torch.float32:
            raise ValueError("reduce_dtype must be torch.float32 for stable gradient reduction")
        if self.output_dtype is not torch.bfloat16:
            raise ValueError("output_dtype must be torch.bfloat16 for this FSDP2 track")


def _required_env_int(name: str) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raise RuntimeError(f"torchrun environment variable {name} is not set")
    try:
        return int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"torchrun environment variable {name} must be an integer") from error


def init_distributed(*, timeout_seconds: int = 600) -> DistContext:
    """Bind the local CUDA device, initialize NCCL, and create a 1D DP mesh."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP2 training requires CUDA; no CUDA device is available")

    rank = _required_env_int("RANK")
    local_rank = _required_env_int("LOCAL_RANK")
    world_size = _required_env_int("WORLD_SIZE")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise RuntimeError("torchrun rank/world-size values are inconsistent")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} cannot address {torch.cuda.device_count()} CUDA devices"
        )

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
    elif dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("existing process group does not match the torchrun environment")

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
        mesh=mesh,
    )


def destroy_distributed() -> None:
    """Release the active process group without adding a failure-prone barrier."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def apply_qwen_activation_checkpointing(model: nn.Module, *, enabled: bool) -> list[str]:
    """Optionally replace every Qwen decoder block with a non-reentrant wrapper."""

    try:
        layers = model.model.layers  # type: ignore[attr-defined]
        model.config.use_cache = False  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("model does not expose the expected Qwen causal-LM structure") from error

    if not enabled:
        return []

    wrapped: list[str] = []
    for index, block in enumerate(layers):
        layers[index] = checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=True,
        )
        wrapped.append(f"layer.{index}")
    return wrapped


def fully_shard_qwen(
    model: nn.Module,
    ctx: DistContext,
    settings: FSDPSettings,
) -> list[str]:
    """Apply FSDP2 bottom-up to every Qwen communication group."""

    try:
        embedding = model.model.embed_tokens  # type: ignore[attr-defined]
        layers = model.model.layers  # type: ignore[attr-defined]
        lm_head = model.lm_head  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("model does not expose the expected Qwen causal-LM structure") from error

    policy = MixedPrecisionPolicy(
        param_dtype=settings.param_dtype,
        reduce_dtype=settings.reduce_dtype,
        output_dtype=settings.output_dtype,
    )
    shard_kwargs = {
        "mesh": ctx.mesh,
        "reshard_after_forward": settings.reshard_after_forward,
        "mp_policy": policy,
    }
    groups: list[str] = []
    fully_shard(embedding, **shard_kwargs)
    groups.append("embed_tokens")
    for index, block in enumerate(layers):
        fully_shard(block, **shard_kwargs)
        groups.append(f"layer.{index}")
    fully_shard(lm_head, **shard_kwargs)
    groups.append("lm_head")
    fully_shard(model, **shard_kwargs)
    groups.append("root")
    return groups


def fsdp_modules(model: nn.Module) -> list[FSDPModule]:
    """Return FSDP2 groups in module traversal order, root first."""

    return [module for module in model.modules() if isinstance(module, FSDPModule)]


def clip_global_grad_norm_(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Clip one logical global norm across DTensor gradient shards on every rank."""

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        foreach=False,
    )


def _set_reshard_after_forward(module: FSDPModule, enabled: bool) -> None:
    setter = getattr(module, "set_reshard_after_forward", None)
    if setter is None:
        raise RuntimeError("rollout state switching requires PyTorch 2.8 or newer")
    setter(enabled, recurse=False)


@contextmanager
def rollout_parameter_state(
    model: nn.Module,
    mode: RolloutMode,
) -> Iterator[None]:
    """Temporarily retain all policy groups unsharded for rollout generation."""

    if mode not in {"reshard", "keep_unsharded"}:
        raise ValueError("rollout mode must be either 'keep_unsharded' or 'reshard'")
    if mode == "reshard":
        yield
        return

    modules = fsdp_modules(model)
    configured: list[FSDPModule] = []
    unsharded: list[FSDPModule] = []
    try:
        for module in modules:
            _set_reshard_after_forward(module, False)
            configured.append(module)
        for module in modules:
            module.unshard()
            unsharded.append(module)
        yield
    finally:
        for module in reversed(unsharded):
            module.reshard()
        for module in configured:
            _set_reshard_after_forward(module, True)
