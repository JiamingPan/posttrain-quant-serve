"""Fixed-global-batch FSDP2 scaling benchmark and sweep controller."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import nn
import numpy as np

from train.checkpointing import load_hf_weights_into_shards
from train.fsdp_sft import (
    _load_sft_features,
    assert_adamw_moment_dtype,
    prepare_step_batches,
    resolve_sft_source,
    sft_optimizer_step,
)
from train.fsdp_utils import (
    DistContext,
    FSDPSettings,
    apply_qwen_activation_checkpointing,
    destroy_distributed,
    fsdp_modules,
    fully_shard_qwen,
    init_distributed,
)
from train.gsm8k_data import CheckpointableDistributedSampler, TokenBatch
from train.memory_model import GIB, preflight_launch
from train.run_tracking import (
    append_run_record,
    capture_run_identity,
    gather_rank_records,
    write_run_config,
)


SUPPORTED_WORLD_SIZES = (1, 2, 4, 8)
MFU_DEFINITION = (
    "6 * parameter_count * useful_nonpadding_tokens / "
    "(seconds * world_size * peak_bf16_flops_per_gpu)"
)
COMMUNICATION_ACTIVE_DEFINITION = (
    "union duration of NCCL CUDA kernels / profiled CUDA step window"
)
COMMUNICATION_EXPOSED_DEFINITION = (
    "NCCL CUDA-kernel duration not overlapped by compute kernels / "
    "profiled CUDA step window"
)
MEMORY_PROBE_METHOD = {
    "params_grads_optimizer": "deduplicated local tensor storage inventory",
    "activations": "maximum live storage observed by saved_tensors_hooks",
    "collectives": "maximum FSDP post-unshard versus post-reshard allocator delta",
    "other": "allocated peak minus inventoried categories, clamped at zero",
}


SCALING_FIELDS = (
    "run_id",
    "stage",
    "status",
    "benchmark_mode",
    "git_commit",
    "git_dirty",
    "slurm_job_id",
    "hostname",
    "started_at_utc",
    "world_size",
    "model",
    "model_revision",
    "parameter_count",
    "dataset",
    "dataset_digest",
    "seed",
    "global_batch_size",
    "sequence_length",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "activation_checkpointing",
    "accumulation_sync",
    "gpu_name",
    "gpu_capacity_gib",
    "gpu_topology",
    "torch_version",
    "cuda_version",
    "nccl_version",
    "transformers_version",
    "datasets_version",
    "comparison_config_digest",
    "warmup_steps",
    "measure_steps",
    "profile_steps",
    "useful_tokens",
    "elapsed_seconds",
    "tokens_per_sec",
    "peak_bf16_tflops_per_gpu",
    "mfu",
    "mfu_definition",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "rank_memory",
    "memory_components",
    "memory_probe_method",
    "communication_active_fraction",
    "communication_exposed_fraction",
    "communication_active_definition",
    "communication_exposed_definition",
    "step_time_mean_seconds",
    "step_time_std_seconds",
    "scaling_efficiency",
)

RANK_FIELDS = (
    "run_id",
    "benchmark_mode",
    "world_size",
    "rank",
    "gpu_name",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "memory_components",
    "communication_active_fraction",
    "communication_exposed_fraction",
    "step_seconds",
)


@dataclass(frozen=True)
class ScalingConfig:
    output_dir: str | None
    worker: bool = False
    validate_dir: str | None = None
    pilot: bool = False
    model: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    world_sizes: tuple[int, ...] = SUPPORTED_WORLD_SIZES
    global_batch_size: int = 8
    sequence_length: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int | None = None
    warmup_steps: int = 3
    measure_steps: int = 10
    profile_steps: int = 3
    activation_checkpointing: bool = True
    accumulation_sync: str = "reduce_scatter"
    dataset: str = "openai/gsm8k"
    dataset_config: str = "main"
    split: str = "train"
    dataset_limit: int = 256
    seed: int = 42
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    attention_backend: str = "sdpa"
    peak_bf16_tflops: float | None = None
    record_stem: str = "scaling"
    timeout_seconds: int = 600
    correctness_dir: str = "results/fsdp_correctness"

    @property
    def benchmark_mode(self) -> str:
        return "pilot" if self.pilot else "full"

    def __post_init__(self) -> None:
        if self.output_dir is None and self.validate_dir is None:
            raise ValueError("--output_dir is required unless --validate is used")
        if not self.world_sizes or any(
            world_size not in SUPPORTED_WORLD_SIZES for world_size in self.world_sizes
        ):
            raise ValueError("world_sizes must contain only 1, 2, 4, and 8")
        positive = {
            "global_batch_size": self.global_batch_size,
            "sequence_length": self.sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "warmup_steps": self.warmup_steps,
            "measure_steps": self.measure_steps,
            "profile_steps": self.profile_steps,
            "dataset_limit": self.dataset_limit,
            "timeout_seconds": self.timeout_seconds,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.gradient_accumulation_steps is not None
            and self.gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.accumulation_sync not in {"reduce_scatter", "no_sync"}:
            raise ValueError("invalid accumulation_sync mode")
        if self.pilot and self.model == "Qwen/Qwen3-8B" and not (
            self.revision is not None
            and re.fullmatch(r"[0-9a-fA-F]{40}", self.revision)
        ):
            raise ValueError(
                "a Qwen3-8B pilot requires an immutable 40-character commit revision"
            )


def _parse_world_sizes(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("world sizes must be comma-separated integers") from error
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("world sizes must be unique")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--validate", dest="validate_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision")
    parser.add_argument("--world_sizes", type=_parse_world_sizes, default=SUPPORTED_WORLD_SIZES)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--profile_steps", type=int, default=3)
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
    parser.add_argument("--dataset_limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--attention_backend",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--peak_bf16_tflops", type=float)
    parser.add_argument("--record_stem", default="scaling")
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--correctness_dir", default="results/fsdp_correctness")
    return parser


def parse_scaling_args(argv: Sequence[str] | None = None) -> ScalingConfig:
    parser = _build_parser()
    namespace = parser.parse_args(argv)
    try:
        return ScalingConfig(**vars(namespace))
    except ValueError as error:
        parser.error(str(error))
        raise AssertionError("argparse.error always exits") from error


def fixed_accumulation_steps(
    *,
    global_batch_size: int,
    world_size: int,
    micro_batch_size: int,
) -> int:
    if global_batch_size <= 0 or world_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch dimensions and world size must be positive")
    denominator = world_size * micro_batch_size
    if global_batch_size % denominator:
        raise ValueError(
            "global_batch_size must be divisible by world_size * micro_batch_size"
        )
    return global_batch_size // denominator


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for start, end in intervals:
        start_value, end_value = float(start), float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise ValueError("interval endpoints must be finite")
        if end_value < start_value:
            raise ValueError("interval end must not precede its start")
        if end_value > start_value:
            normalized.append((start_value, end_value))
    normalized.sort()
    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _clip_intervals(
    intervals: Iterable[tuple[float, float]],
    window: tuple[float, float],
) -> list[tuple[float, float]]:
    start, end = window
    return merge_intervals(
        (max(left, start), min(right, end))
        for left, right in intervals
        if right > start and left < end
    )


def _interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_duration(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    left_index = right_index = 0
    duration = 0.0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        duration += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return duration


def communication_fractions(
    communication_intervals: Iterable[tuple[float, float]],
    compute_intervals: Iterable[tuple[float, float]],
    *,
    step_window: tuple[float, float],
) -> tuple[float, float]:
    window_start, window_end = map(float, step_window)
    window_duration = window_end - window_start
    if not math.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("step_window must have positive duration")
    communication = _clip_intervals(
        communication_intervals,
        (window_start, window_end),
    )
    compute = _clip_intervals(compute_intervals, (window_start, window_end))
    active_duration = _interval_duration(communication)
    overlap_duration = _intersection_duration(communication, compute)
    exposed_duration = max(active_duration - overlap_duration, 0.0)
    return active_duration / window_duration, exposed_duration / window_duration


def compute_mfu(
    *,
    parameters: int,
    useful_tokens: int,
    seconds: float,
    world_size: int,
    peak_bf16_tflops_per_gpu: float,
) -> float:
    if parameters <= 0 or useful_tokens <= 0 or seconds <= 0 or world_size <= 0:
        raise ValueError("MFU inputs must be positive")
    if peak_bf16_tflops_per_gpu <= 0:
        raise ValueError("peak bf16 throughput must be positive")
    useful_flops = 6 * parameters * useful_tokens
    available_flops = seconds * world_size * peak_bf16_tflops_per_gpu * 1e12
    return useful_flops / available_flops


def scaling_efficiency(
    *,
    throughput: float,
    baseline_throughput: float,
    world_size: int,
) -> float:
    if throughput <= 0 or baseline_throughput <= 0 or world_size <= 0:
        raise ValueError("scaling-efficiency inputs must be positive")
    return throughput / (world_size * baseline_throughput)


def hardware_peak_bf16_tflops(
    gpu_name: str,
    *,
    override: float | None = None,
) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("peak_bf16_tflops override must be positive")
        return override
    normalized = gpu_name.upper()
    if "A100" in normalized:
        return 312.0
    if "A40" in normalized:
        return 149.7
    raise ValueError(
        f"unknown dense bf16 peak for {gpu_name!r}; pass --peak_bf16_tflops"
    )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        tensor = to_local()
    wait = getattr(tensor, "wait", None)
    if callable(wait):
        tensor = wait()
    return tensor


def _tensor_storage_entry(tensor: torch.Tensor) -> tuple[tuple[Any, ...], int]:
    local = _local_tensor(tensor.detach())
    if local.device.type == "meta":
        return (("meta", id(local)), 0)
    storage = local.untyped_storage()
    size = int(storage.nbytes())
    key = (
        local.device.type,
        local.device.index,
        int(storage.data_ptr()),
        size,
    )
    return key, size


def deduplicated_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[tuple[Any, ...], int] = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("storage inventory accepts tensors only")
        key, size = _tensor_storage_entry(tensor)
        storages[key] = size
    return sum(storages.values())


def _nested_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _nested_tensors(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _nested_tensors(nested)


def measure_memory_components(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    activation_bytes: int,
    collective_bytes: int,
    peak_allocated_bytes: int,
    gradient_bytes_override: int | None = None,
) -> dict[str, int]:
    if min(activation_bytes, collective_bytes, peak_allocated_bytes) < 0:
        raise ValueError("memory counters must be non-negative")
    params_bytes = deduplicated_storage_bytes(model.parameters())
    grads_bytes = (
        gradient_bytes_override
        if gradient_bytes_override is not None
        else deduplicated_storage_bytes(
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    optimizer_bytes = deduplicated_storage_bytes(
        _nested_tensors(optimizer.state)
    )
    known = (
        params_bytes
        + grads_bytes
        + optimizer_bytes
        + activation_bytes
        + collective_bytes
    )
    other_bytes = max(peak_allocated_bytes - known, 0)
    return {
        "params_bytes": params_bytes,
        "grads_bytes": grads_bytes,
        "optimizer_bytes": optimizer_bytes,
        "activations_bytes": activation_bytes,
        "collectives_bytes": collective_bytes,
        "other_bytes": other_bytes,
        "category_sum_bytes": known + other_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
    }


class _SavedTensorLiveBytes:
    def __init__(self) -> None:
        self._references: dict[tuple[Any, ...], tuple[int, int]] = {}
        self.current_bytes = 0
        self.maximum_bytes = 0

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        key, size = _tensor_storage_entry(tensor)
        count, existing_size = self._references.get(key, (0, size))
        if count == 0:
            self.current_bytes += size
            self.maximum_bytes = max(self.maximum_bytes, self.current_bytes)
        self._references[key] = (count + 1, existing_size)
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        key, _ = _tensor_storage_entry(tensor)
        count, size = self._references[key]
        if count == 1:
            self.current_bytes -= size
            del self._references[key]
        else:
            self._references[key] = (count - 1, size)
        return tensor


class _CollectiveAllocationTracker:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._unsharded_allocated: dict[int, int] = {}
        self.maximum_delta_bytes = 0
        self._handles: list[Any] = []

    def _pre_forward(self, module: nn.Module, _inputs: Any) -> None:
        self._unsharded_allocated[id(module)] = torch.cuda.memory_allocated(self.device)

    def _post_forward(self, module: nn.Module, _inputs: Any, output: Any) -> Any:
        unsharded = self._unsharded_allocated.pop(id(module), 0)
        resharded = torch.cuda.memory_allocated(self.device)
        self.maximum_delta_bytes = max(
            self.maximum_delta_bytes,
            max(unsharded - resharded, 0),
        )
        return output

    @contextmanager
    def observe(self, model: nn.Module) -> Iterator[None]:
        try:
            for module in fsdp_modules(model):
                self._handles.append(module.register_forward_pre_hook(self._pre_forward))
                self._handles.append(module.register_forward_hook(self._post_forward))
            yield
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()


def _maximum_component_values(
    rank_memory: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    component_names = (
        "params_bytes",
        "grads_bytes",
        "optimizer_bytes",
        "activations_bytes",
        "collectives_bytes",
        "other_bytes",
        "category_sum_bytes",
        "peak_allocated_bytes",
    )
    return {
        name: max(int(rank["memory_components"][name]) for rank in rank_memory)
        for name in component_names
    }


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _nccl_version() -> str:
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return ".".join(str(value) for value in version)
    return str(version)


def _gpu_topology() -> str:
    try:
        completed = subprocess.run(
            ("nvidia-smi", "topo", "-m"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _comparison_digest(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _broadcast_object(value: Any, ctx: DistContext) -> Any:
    values = [value if ctx.rank == 0 else None]
    if ctx.world_size > 1:
        dist.broadcast_object_list(values, src=0)
    return values[0]


def _move_batches_to_device(
    batches: Sequence[TokenBatch],
    device: torch.device,
) -> list[TokenBatch]:
    return [
        TokenBatch(
            input_ids=batch.input_ids.to(device),
            labels=batch.labels.to(device),
            attention_mask=batch.attention_mask.to(device),
            supervised_tokens=batch.supervised_tokens,
        )
        for batch in batches
    ]


def _next_step_batches(
    features: Sequence[Any],
    sampler: CheckpointableDistributedSampler,
    *,
    config: ScalingConfig,
    accumulation_steps: int,
    pad_token_id: int,
    device: torch.device,
) -> list[TokenBatch]:
    batches = prepare_step_batches(
        features,
        sampler,
        local_microbatch_size=config.micro_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        pad_token_id=pad_token_id,
        pad_to_length=config.sequence_length,
    )
    return _move_batches_to_device(batches, device)


def _profile_intervals(profiler: Any) -> tuple[list[tuple[float, float]], list[tuple[float, float]], tuple[float, float]]:
    communication: list[tuple[float, float]] = []
    compute: list[tuple[float, float]] = []
    all_cuda: list[tuple[float, float]] = []
    for event in profiler.events():
        device_type = str(getattr(event, "device_type", "")).lower()
        if "cuda" not in device_type:
            continue
        time_range = getattr(event, "time_range", None)
        if time_range is None:
            continue
        interval = (float(time_range.start), float(time_range.end))
        if interval[1] <= interval[0]:
            continue
        all_cuda.append(interval)
        name = str(getattr(event, "name", "")).lower()
        if "nccl" in name:
            communication.append(interval)
        elif not any(token in name for token in ("memcpy", "memset")):
            compute.append(interval)
    if not all_cuda:
        raise RuntimeError("PyTorch profiler produced no CUDA kernel intervals")
    return communication, compute, (
        min(start for start, _ in all_cuda),
        max(end for _, end in all_cuda),
    )


def _run_profile_steps(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    batches: Sequence[Sequence[TokenBatch]],
    ctx: DistContext,
    config: ScalingConfig,
) -> tuple[float, float]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize(ctx.device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for step_batches in batches:
            sft_optimizer_step(
                model,
                optimizer,
                scheduler,
                step_batches,
                ctx=ctx,
                max_grad_norm=config.max_grad_norm,
                accumulation_sync=config.accumulation_sync,
            )
            profiler.step()
    torch.cuda.synchronize(ctx.device)
    communication, compute, window = _profile_intervals(profiler)
    return communication_fractions(communication, compute, step_window=window)


def _measure_probe_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    batches: Sequence[TokenBatch],
    ctx: DistContext,
    config: ScalingConfig,
) -> tuple[dict[str, int], int, int]:
    saved_tensors = _SavedTensorLiveBytes()
    collectives = _CollectiveAllocationTracker(ctx.device)
    observed_gradient_bytes = 0

    def observe_gradients(observed_model: nn.Module) -> None:
        nonlocal observed_gradient_bytes
        observed_gradient_bytes = deduplicated_storage_bytes(
            parameter.grad
            for parameter in observed_model.parameters()
            if parameter.grad is not None
        )

    with collectives.observe(model):
        with torch.autograd.graph.saved_tensors_hooks(
            saved_tensors.pack,
            saved_tensors.unpack,
        ):
            metrics = sft_optimizer_step(
                model,
                optimizer,
                scheduler,
                batches,
                ctx=ctx,
                max_grad_norm=config.max_grad_norm,
                accumulation_sync=config.accumulation_sync,
                before_optimizer_step=observe_gradients,
            )
    components = measure_memory_components(
        model,
        optimizer,
        activation_bytes=saved_tensors.maximum_bytes,
        collective_bytes=collectives.maximum_delta_bytes,
        peak_allocated_bytes=metrics.peak_allocated_bytes,
        gradient_bytes_override=observed_gradient_bytes,
    )
    return components, metrics.peak_allocated_bytes, metrics.peak_reserved_bytes


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _worker_resolved_config(
    config: ScalingConfig,
    *,
    ctx: DistContext,
    accumulation_steps: int,
    model_revision: str,
    dataset_digest: str,
) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "benchmark_mode": config.benchmark_mode,
            "world_size": ctx.world_size,
            "gradient_accumulation_steps": accumulation_steps,
            "model_revision": model_revision,
            "dataset_digest": dataset_digest,
        }
    )
    return payload


def _comparison_config(
    config: ScalingConfig,
    *,
    model_revision: str,
    dataset_digest: str,
) -> dict[str, Any]:
    return {
        "model": config.model,
        "model_revision": model_revision,
        "dataset": config.dataset,
        "dataset_config": config.dataset_config,
        "split": config.split,
        "dataset_limit": config.dataset_limit,
        "dataset_digest": dataset_digest,
        "seed": config.seed,
        "global_batch_size": config.global_batch_size,
        "sequence_length": config.sequence_length,
        "micro_batch_size": config.micro_batch_size,
        "activation_checkpointing": config.activation_checkpointing,
        "accumulation_sync": config.accumulation_sync,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "attention_backend": config.attention_backend,
        "warmup_steps": config.warmup_steps,
        "measure_steps": config.measure_steps,
        "profile_steps": config.profile_steps,
    }


def run_worker(config: ScalingConfig) -> dict[str, Any] | None:
    """Measure one world size; rank zero writes and returns the complete record."""

    ctx = init_distributed(timeout_seconds=config.timeout_seconds)
    try:
        accumulation_steps = fixed_accumulation_steps(
            global_batch_size=config.global_batch_size,
            world_size=ctx.world_size,
            micro_batch_size=config.micro_batch_size,
        )
        if (
            config.gradient_accumulation_steps is not None
            and config.gradient_accumulation_steps != accumulation_steps
        ):
            raise ValueError(
                "gradient_accumulation_steps does not preserve the configured global batch"
            )

        gpu_properties = torch.cuda.get_device_properties(ctx.device)
        preflight_launch(
            stage="sft",
            world_size=ctx.world_size,
            stage_args=(
                "--model",
                config.model,
                "--activation_checkpointing"
                if config.activation_checkpointing
                else "--no_activation_checkpointing",
            ),
            device_name=gpu_properties.name,
            capacity_gib=gpu_properties.total_memory / GIB,
        )

        random.seed(config.seed + ctx.rank)
        np.random.seed(config.seed + ctx.rank)
        torch.manual_seed(config.seed + ctx.rank)
        torch.cuda.manual_seed(config.seed + ctx.rank)

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        source = resolve_sft_source(config.model, config.revision)
        tokenizer = AutoTokenizer.from_pretrained(source.path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        data_config = type(
            "ScalingDataConfig",
            (),
            {
                "dataset": config.dataset,
                "dataset_config": config.dataset_config,
                "split": config.split,
                "dataset_limit": config.dataset_limit,
                "sequence_length": config.sequence_length,
                "packing": True,
            },
        )()
        features, dataset_digest = _load_sft_features(data_config, tokenizer)
        model_config = AutoConfig.from_pretrained(source.path)
        model_config.use_cache = False
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=config.attention_backend,
            )
        apply_qwen_activation_checkpointing(
            model,
            enabled=config.activation_checkpointing,
        )
        fully_shard_qwen(model, ctx, FSDPSettings())
        load_hf_weights_into_shards(
            model,
            source.path,
            device=ctx.device,
            revision=source.revision,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        sampler = CheckpointableDistributedSampler(
            len(features),
            rank=ctx.rank,
            world_size=ctx.world_size,
            seed=config.seed,
            shuffle=True,
        )

        resolved_config = _worker_resolved_config(
            config,
            ctx=ctx,
            accumulation_steps=accumulation_steps,
            model_revision=source.revision,
            dataset_digest=dataset_digest,
        )
        comparison_digest = _comparison_digest(
            _comparison_config(
                config,
                model_revision=source.revision,
                dataset_digest=dataset_digest,
            )
        )
        identity_value = (
            capture_run_identity("scaling", resolved_config).as_dict()
            if ctx.rank == 0
            else None
        )
        identity = _broadcast_object(identity_value, ctx)
        if identity is None:
            raise RuntimeError("rank zero did not publish a scaling identity")
        controller_commit = os.environ.get("PQS_SCALING_CONTROLLER_COMMIT")
        if controller_commit is not None:
            if identity["git_commit"] != controller_commit:
                raise RuntimeError("scaling worker commit differs from its controller")
            identity["git_dirty"] = False
        elif config.model == "Qwen/Qwen3-8B" and identity["git_dirty"]:
            raise RuntimeError("Qwen3 scaling refuses an uncommitted worktree")

        for warmup_index in range(config.warmup_steps):
            batches = _next_step_batches(
                features,
                sampler,
                config=config,
                accumulation_steps=accumulation_steps,
                pad_token_id=tokenizer.pad_token_id,
                device=ctx.device,
            )
            sft_optimizer_step(
                model,
                optimizer,
                scheduler,
                batches,
                ctx=ctx,
                max_grad_norm=config.max_grad_norm,
                accumulation_sync=config.accumulation_sync,
            )
            if warmup_index == 0:
                assert_adamw_moment_dtype(optimizer, torch.bfloat16)

        torch.cuda.synchronize(ctx.device)
        torch.cuda.reset_peak_memory_stats(ctx.device)
        dist.barrier()
        measurement_started = time.perf_counter()
        step_seconds: list[float] = []
        local_useful_tokens = 0
        peak_allocated = 0
        peak_reserved = 0
        for _ in range(config.measure_steps):
            batches = _next_step_batches(
                features,
                sampler,
                config=config,
                accumulation_steps=accumulation_steps,
                pad_token_id=tokenizer.pad_token_id,
                device=ctx.device,
            )
            local_useful_tokens += sum(
                int(batch.attention_mask.sum().item()) for batch in batches
            )
            metrics = sft_optimizer_step(
                model,
                optimizer,
                scheduler,
                batches,
                ctx=ctx,
                max_grad_norm=config.max_grad_norm,
                accumulation_sync=config.accumulation_sync,
            )
            step_seconds.append(metrics.elapsed_seconds)
            peak_allocated = max(peak_allocated, metrics.peak_allocated_bytes)
            peak_reserved = max(peak_reserved, metrics.peak_reserved_bytes)
        torch.cuda.synchronize(ctx.device)
        dist.barrier()
        elapsed = time.perf_counter() - measurement_started

        token_tensor = torch.tensor(local_useful_tokens, device=ctx.device, dtype=torch.int64)
        elapsed_tensor = torch.tensor(elapsed, device=ctx.device, dtype=torch.float64)
        if ctx.world_size > 1:
            dist.all_reduce(token_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        useful_tokens = int(token_tensor.item())
        elapsed = float(elapsed_tensor.item())

        probe_batches = _next_step_batches(
            features,
            sampler,
            config=config,
            accumulation_steps=accumulation_steps,
            pad_token_id=tokenizer.pad_token_id,
            device=ctx.device,
        )
        components, probe_allocated, probe_reserved = _measure_probe_step(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batches=probe_batches,
            ctx=ctx,
            config=config,
        )
        peak_allocated = max(peak_allocated, probe_allocated)
        peak_reserved = max(peak_reserved, probe_reserved)

        profile_batches = [
            _next_step_batches(
                features,
                sampler,
                config=config,
                accumulation_steps=accumulation_steps,
                pad_token_id=tokenizer.pad_token_id,
                device=ctx.device,
            )
            for _ in range(config.profile_steps)
        ]
        communication_active, communication_exposed = _run_profile_steps(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batches=profile_batches,
            ctx=ctx,
            config=config,
        )

        local_rank_record = {
            "run_id": identity["run_id"],
            "benchmark_mode": config.benchmark_mode,
            "world_size": ctx.world_size,
            "rank": ctx.rank,
            "gpu_name": gpu_properties.name,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "memory_components": components,
            "communication_active_fraction": communication_active,
            "communication_exposed_fraction": communication_exposed,
            "step_seconds": step_seconds,
        }
        rank_memory = gather_rank_records(local_rank_record)
        if ctx.rank != 0:
            return None
        if rank_memory is None or len(rank_memory) != ctx.world_size:
            raise RuntimeError("rank-zero scaling record did not receive every rank")

        all_step_seconds = [
            max(float(rank["step_seconds"][index]) for rank in rank_memory)
            for index in range(config.measure_steps)
        ]
        tokens_per_sec = useful_tokens / elapsed
        peak_tflops = hardware_peak_bf16_tflops(
            gpu_properties.name,
            override=config.peak_bf16_tflops,
        )
        parameter_count = _parameter_count(model)
        record = {
            **identity,
            "status": "ok",
            "benchmark_mode": config.benchmark_mode,
            "world_size": ctx.world_size,
            "model": config.model,
            "model_revision": source.revision,
            "parameter_count": parameter_count,
            "dataset": config.dataset,
            "dataset_digest": dataset_digest,
            "seed": config.seed,
            "global_batch_size": config.global_batch_size,
            "sequence_length": config.sequence_length,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "activation_checkpointing": config.activation_checkpointing,
            "accumulation_sync": config.accumulation_sync,
            "gpu_name": gpu_properties.name,
            "gpu_capacity_gib": gpu_properties.total_memory / GIB,
            "gpu_topology": _gpu_topology(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": _nccl_version(),
            "transformers_version": _package_version("transformers"),
            "datasets_version": _package_version("datasets"),
            "comparison_config_digest": comparison_digest,
            "warmup_steps": config.warmup_steps,
            "measure_steps": config.measure_steps,
            "profile_steps": config.profile_steps,
            "useful_tokens": useful_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "peak_bf16_tflops_per_gpu": peak_tflops,
            "mfu": compute_mfu(
                parameters=parameter_count,
                useful_tokens=useful_tokens,
                seconds=elapsed,
                world_size=ctx.world_size,
                peak_bf16_tflops_per_gpu=peak_tflops,
            ),
            "mfu_definition": MFU_DEFINITION,
            "peak_allocated_bytes": max(
                int(rank["peak_allocated_bytes"]) for rank in rank_memory
            ),
            "peak_reserved_bytes": max(
                int(rank["peak_reserved_bytes"]) for rank in rank_memory
            ),
            "rank_memory": rank_memory,
            "memory_components": _maximum_component_values(rank_memory),
            "memory_probe_method": MEMORY_PROBE_METHOD,
            "communication_active_fraction": max(
                float(rank["communication_active_fraction"])
                for rank in rank_memory
            ),
            "communication_exposed_fraction": max(
                float(rank["communication_exposed_fraction"])
                for rank in rank_memory
            ),
            "communication_active_definition": COMMUNICATION_ACTIVE_DEFINITION,
            "communication_exposed_definition": COMMUNICATION_EXPOSED_DEFINITION,
            "step_time_mean_seconds": statistics.fmean(all_step_seconds),
            "step_time_std_seconds": statistics.pstdev(all_step_seconds),
            "scaling_efficiency": None,
        }

        output_dir = Path(config.output_dir or "")
        write_run_config(output_dir, resolved_config, filename="run_config.json")
        append_run_record(output_dir, config.record_stem, record, SCALING_FIELDS)
        for rank_record in rank_memory:
            append_run_record(output_dir, "ranks", rank_record, RANK_FIELDS)
        (output_dir / "worker_result.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        return record
    finally:
        destroy_distributed()


def _worker_cli_arguments(
    config: ScalingConfig,
    *,
    world_size: int,
    output_dir: Path,
) -> list[str]:
    accumulation_steps = fixed_accumulation_steps(
        global_batch_size=config.global_batch_size,
        world_size=world_size,
        micro_batch_size=config.micro_batch_size,
    )
    arguments = [
        "--worker",
        "--output_dir",
        str(output_dir),
        "--model",
        config.model,
        "--world_sizes",
        str(world_size),
        "--global_batch_size",
        str(config.global_batch_size),
        "--sequence_length",
        str(config.sequence_length),
        "--micro_batch_size",
        str(config.micro_batch_size),
        "--gradient_accumulation_steps",
        str(accumulation_steps),
        "--warmup_steps",
        str(config.warmup_steps),
        "--measure_steps",
        str(config.measure_steps),
        "--profile_steps",
        str(config.profile_steps),
        "--accumulation_sync",
        config.accumulation_sync,
        "--dataset",
        config.dataset,
        "--dataset_config",
        config.dataset_config,
        "--split",
        config.split,
        "--dataset_limit",
        str(config.dataset_limit),
        "--seed",
        str(config.seed),
        "--learning_rate",
        str(config.learning_rate),
        "--weight_decay",
        str(config.weight_decay),
        "--max_grad_norm",
        str(config.max_grad_norm),
        "--attention_backend",
        config.attention_backend,
        "--record_stem",
        config.record_stem,
        "--timeout_seconds",
        str(config.timeout_seconds),
        "--correctness_dir",
        config.correctness_dir,
        "--activation_checkpointing"
        if config.activation_checkpointing
        else "--no_activation_checkpointing",
    ]
    if config.revision is not None:
        arguments.extend(("--revision", config.revision))
    if config.peak_bf16_tflops is not None:
        arguments.extend(("--peak_bf16_tflops", str(config.peak_bf16_tflops)))
    if config.pilot:
        arguments.append("--pilot")
    return arguments


def build_worker_command(
    config: ScalingConfig,
    *,
    world_size: int,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--module",
        "bench.scaling",
        *_worker_cli_arguments(
            config,
            world_size=world_size,
            output_dir=output_dir,
        ),
    ]


def _load_gate_record(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing committed correctness gate: {path}") from error


def _require_committed_correctness_gates(
    directory: str | Path,
    gate_names: Sequence[str],
) -> None:
    from bench.correctness import validate_gate_record

    for name in gate_names:
        path = Path(directory) / name
        record = _load_gate_record(path)
        validate_gate_record(record)
        tracked = subprocess.run(
            ("git", "ls-files", "--error-unmatch", str(path)),
            check=False,
            capture_output=True,
            text=True,
        )
        unchanged = subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", str(path)),
            check=False,
        )
        if tracked.returncode != 0 or unchanged.returncode != 0:
            raise ValueError(f"correctness gate is not committed at HEAD: {path}")


def _visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        return [value.strip() for value in raw.split(",") if value.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def required_controller_gpu_count(config: ScalingConfig) -> int:
    if not config.pilot:
        if tuple(config.world_sizes) != SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "the full scaling controller requires world sizes 1,2,4,8"
            )
        return 8
    if (
        len(set(config.world_sizes)) != len(config.world_sizes)
        or tuple(sorted(config.world_sizes)) != tuple(config.world_sizes)
    ):
        raise ValueError("pilot world sizes must be strictly increasing")
    return max(config.world_sizes)


def required_correctness_gate_names(config: ScalingConfig) -> tuple[str, ...]:
    if config.model != "Qwen/Qwen3-8B":
        return ()
    if config.pilot:
        return ("sft_gate.json",)
    return ("sft_gate.json", "grpo_gate.json")


def apply_scaling_efficiencies(records: list[dict[str, Any]]) -> None:
    baseline_record = next(
        (record for record in records if int(record["world_size"]) == 1),
        None,
    )
    if baseline_record is None:
        for record in records:
            record["scaling_efficiency"] = None
        return
    baseline = float(baseline_record["tokens_per_sec"])
    for record in records:
        record["scaling_efficiency"] = scaling_efficiency(
            throughput=float(record["tokens_per_sec"]),
            baseline_throughput=baseline,
            world_size=int(record["world_size"]),
        )


def validate_controller_gpu_inventory(
    config: ScalingConfig,
    *,
    visible_gpu_ids: Sequence[str],
    device_count: int,
    gpu_names: Sequence[str],
) -> int:
    required = required_controller_gpu_count(config)
    if len(visible_gpu_ids) != required or device_count != required:
        raise ValueError(
            f"the scaling controller requires exactly {required} visible GPUs"
        )
    if len(gpu_names) != required:
        raise ValueError("GPU inventory does not cover every visible device")
    if len(set(gpu_names)) != 1:
        raise ValueError("the scaling controller requires homogeneous GPU hardware")
    return required


def run_sweep_controller(config: ScalingConfig) -> list[dict[str, Any]]:
    """Run isolated workers sequentially on one homogeneous GPU node."""

    if config.worker:
        raise ValueError("controller cannot run with --worker")
    required_controller_gpu_count(config)
    gate_names = required_correctness_gate_names(config)
    if gate_names:
        _require_committed_correctness_gates(config.correctness_dir, gate_names)
    controller_identity = capture_run_identity(
        "scaling-controller",
        asdict(config),
    )
    if config.model == "Qwen/Qwen3-8B" and controller_identity.git_dirty:
        raise RuntimeError("Qwen3 scaling refuses an uncommitted worktree")
    visible = _visible_gpu_ids()
    device_count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    validate_controller_gpu_inventory(
        config,
        visible_gpu_ids=visible,
        device_count=device_count,
        gpu_names=names,
    )

    output_dir = Path(config.output_dir or "")
    worker_root = output_dir / "workers"
    records: list[dict[str, Any]] = []
    for world_size in config.world_sizes:
        worker_output = worker_root / f"w{world_size}"
        command = build_worker_command(
            config,
            world_size=world_size,
            output_dir=worker_output,
        )
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible[:world_size])
        if not controller_identity.git_dirty:
            environment["PQS_SCALING_CONTROLLER_COMMIT"] = controller_identity.git_commit
        subprocess.run(command, check=True, env=environment)
        record_path = worker_output / "worker_result.json"
        records.append(json.loads(record_path.read_text(encoding="utf-8")))

    digests = {record["comparison_config_digest"] for record in records}
    gpu_names = {record["gpu_name"] for record in records}
    if len(digests) != 1:
        raise RuntimeError("worker comparison configuration digests differ")
    if len(gpu_names) != 1:
        raise RuntimeError("worker GPU identities differ")
    apply_scaling_efficiencies(records)
    if config.pilot:
        validate_pilot_scaling_records(
            records,
            expected_world_sizes=config.world_sizes,
        )
    else:
        validate_scaling_records(records)

    controller_config = asdict(config)
    controller_config["benchmark_mode"] = config.benchmark_mode
    controller_config["gpu_name"] = names[0]
    write_run_config(output_dir, controller_config, filename="run_config.json")
    for record in records:
        append_run_record(output_dir, config.record_stem, record, SCALING_FIELDS)
        for rank_record in record["rank_memory"]:
            append_run_record(output_dir, "ranks", rank_record, RANK_FIELDS)
    return records


def _validate_scaling_record_set(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_world_sizes: tuple[int, ...],
    benchmark_mode: str,
) -> None:
    if (
        not expected_world_sizes
        or tuple(sorted(set(expected_world_sizes))) != expected_world_sizes
        or any(
            world_size not in SUPPORTED_WORLD_SIZES
            for world_size in expected_world_sizes
        )
    ):
        raise ValueError(
            "expected world sizes must be a strictly increasing supported subset"
        )
    by_world_size = {int(record["world_size"]): record for record in records}
    if (
        tuple(sorted(by_world_size)) != expected_world_sizes
        or len(records) != len(expected_world_sizes)
    ):
        raise ValueError("scaling records do not match the requested world sizes")
    if {str(record["benchmark_mode"]) for record in records} != {benchmark_mode}:
        raise ValueError("scaling record benchmark mode does not match its controller")
    if len({str(record["gpu_name"]) for record in records}) != 1:
        raise ValueError("scaling records require homogeneous GPU hardware")
    if len(records) > 1 and len(
        {str(record["comparison_config_digest"]) for record in records}
    ) != 1:
        raise ValueError("scaling records have mismatched comparison configurations")
    if {int(record["global_batch_size"]) for record in records} != {8}:
        raise ValueError("scaling records must use fixed global batch size eight")

    for world_size, record in by_world_size.items():
        for name in ("tokens_per_sec", "mfu", "step_time_mean_seconds"):
            value = float(record[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        step_time_std = float(record["step_time_std_seconds"])
        if not math.isfinite(step_time_std) or step_time_std < 0:
            raise ValueError(
                "step_time_std_seconds must be finite and non-negative"
            )
        for name in (
            "communication_active_fraction",
            "communication_exposed_fraction",
        ):
            value = float(record[name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

        rank_memory = list(record["rank_memory"])
        ranks = {int(rank["rank"]) for rank in rank_memory}
        if len(rank_memory) != world_size or ranks != set(range(world_size)):
            raise ValueError(
                f"world size {world_size} is missing complete rank memory records"
            )
        measure_steps = int(record["measure_steps"])
        for rank in rank_memory:
            allocated = int(rank["peak_allocated_bytes"])
            reserved = int(rank["peak_reserved_bytes"])
            if allocated <= 0:
                raise ValueError("rank allocated peak must be positive")
            if reserved < allocated:
                raise ValueError("rank reserved peak must cover allocated peak")
            durations = [float(value) for value in rank["step_seconds"]]
            if len(durations) != measure_steps or any(
                not math.isfinite(value) or value <= 0 for value in durations
            ):
                raise ValueError("rank record has invalid measured step durations")


def validate_scaling_records(records: Sequence[Mapping[str, Any]]) -> None:
    _validate_scaling_record_set(
        records,
        expected_world_sizes=SUPPORTED_WORLD_SIZES,
        benchmark_mode="full",
    )
    for record in records:
        efficiency = record["scaling_efficiency"]
        if efficiency is None:
            raise ValueError("full scaling efficiency must be finite and positive")
        value = float(efficiency)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("full scaling efficiency must be finite and positive")


def validate_pilot_scaling_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_world_sizes: tuple[int, ...],
) -> None:
    _validate_scaling_record_set(
        records,
        expected_world_sizes=expected_world_sizes,
        benchmark_mode="pilot",
    )
    has_baseline = 1 in expected_world_sizes
    for record in records:
        efficiency = record["scaling_efficiency"]
        if has_baseline:
            if efficiency is None:
                raise ValueError(
                    "pilot scaling efficiency requires a positive world-size-1 baseline"
                )
            value = float(efficiency)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "pilot scaling efficiency requires a positive world-size-1 baseline"
                )
        elif efficiency is not None:
            raise ValueError("scaling efficiency requires a world-size-1 baseline")


def validate_scaling_directory(directory: str | Path) -> list[dict[str, Any]]:
    directory_path = Path(directory)
    path = directory_path / "scaling.jsonl"
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError as error:
        raise ValueError(f"missing scaling record file: {path}") from error
    config_path = directory_path / "run_config.json"
    try:
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing scaling run config: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid scaling run config: {config_path}") from error
    if not isinstance(run_config, dict) or not {
        "pilot",
        "world_sizes",
    }.issubset(run_config):
        raise ValueError("scaling run config must record pilot and world_sizes")
    try:
        expected_world_sizes = tuple(int(value) for value in run_config["world_sizes"])
    except (TypeError, ValueError) as error:
        raise ValueError("scaling run config has invalid world_sizes") from error
    if run_config["pilot"] is True:
        validate_pilot_scaling_records(
            records,
            expected_world_sizes=expected_world_sizes,
        )
    elif run_config["pilot"] is False:
        if expected_world_sizes != SUPPORTED_WORLD_SIZES:
            raise ValueError("full scaling run config must record world sizes 1,2,4,8")
        validate_scaling_records(records)
    else:
        raise ValueError("scaling run config pilot flag must be boolean")
    return records


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_scaling_args(argv)
    if config.validate_dir is not None:
        records = validate_scaling_directory(config.validate_dir)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "world_sizes": [record["world_size"] for record in records],
                },
                sort_keys=True,
            )
        )
    elif config.worker:
        run_worker(config)
    else:
        run_sweep_controller(config)


if __name__ == "__main__":
    main()
