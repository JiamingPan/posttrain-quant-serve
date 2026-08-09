"""Analytical GPU-memory estimates used as a preflight safety check."""

from __future__ import annotations

from dataclasses import dataclass


GIB = 1024**3
BF16_BYTES = 2
SUPPORTED_WORLD_SIZES = {1, 2, 4, 8}


@dataclass(frozen=True)
class MemoryPrediction:
    stage: str
    world_size: int
    checkpointing: bool
    params_gib: float
    grads_gib: float
    optimizer_gib: float
    activations_gib: float
    collectives_gib: float
    other_gib: float
    allocated_gib: float
    reserved_gib: float


@dataclass(frozen=True)
class GRPOMemoryPrediction:
    stage: str
    world_size: int
    rollout_mode: str
    beta: float
    reference_gib: float
    training_phase_gib: float
    rollout_phase_gib: float
    peak_gib: float
    reserved_gib: float


def _validate_inputs(parameter_count: int, world_size: int) -> None:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"world_size must be one of {sorted(SUPPORTED_WORLD_SIZES)}")


def _bf16_gib(parameter_count: int) -> float:
    return parameter_count * BF16_BYTES / GIB


def predict_sft_peak(
    parameter_count: int,
    world_size: int,
    *,
    checkpointing: bool,
) -> MemoryPrediction:
    """Predict native-bf16 SFT peak memory using the approved component ledger."""

    _validate_inputs(parameter_count, world_size)
    shard_gib = _bf16_gib(parameter_count) / world_size
    params_gib = shard_gib
    grads_gib = shard_gib
    optimizer_gib = 2 * shard_gib
    activations_gib = 3.5 if checkpointing else 13.5
    collectives_gib = 0.0 if world_size == 1 else 2.7
    other_gib = 1.5
    # The committed ledger reports components to two decimal places, so its
    # displayed total and reserve headroom are computed from those same values.
    allocated_gib = sum(
        round(component, 2)
        for component in (
            params_gib,
            grads_gib,
            optimizer_gib,
            activations_gib,
            collectives_gib,
            other_gib,
        )
    )
    return MemoryPrediction(
        stage="sft",
        world_size=world_size,
        checkpointing=checkpointing,
        params_gib=params_gib,
        grads_gib=grads_gib,
        optimizer_gib=optimizer_gib,
        activations_gib=activations_gib,
        collectives_gib=collectives_gib,
        other_gib=other_gib,
        allocated_gib=allocated_gib,
        reserved_gib=allocated_gib * 1.08,
    )


def predict_grpo_peak(
    parameter_count: int,
    *,
    world_size: int,
    beta: float,
    rollout_mode: str,
) -> GRPOMemoryPrediction:
    """Predict separate GRPO training and rollout phases.

    ``keep_unsharded`` models the simple baseline where each rank retains a
    full bf16 policy during generation. ``reshard`` models layer-wise gathers
    whose temporary footprint is bounded by the largest transformer block.
    """

    _validate_inputs(parameter_count, world_size)
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if rollout_mode not in {"keep_unsharded", "reshard"}:
        raise ValueError("rollout_mode must be 'keep_unsharded' or 'reshard'")

    full_model_gib = _bf16_gib(parameter_count)
    shard_gib = full_model_gib / world_size
    reference_gib = shard_gib if beta > 0 else 0.0

    training_phase_gib = (
        4 * shard_gib
        + 3.0
        + (0.0 if world_size == 1 else 2.7)
        + 1.5
        + reference_gib
    )
    if world_size == 1:
        generation_gather_gib = 0.0
    elif rollout_mode == "keep_unsharded":
        generation_gather_gib = full_model_gib
    else:
        generation_gather_gib = 1.16
    rollout_phase_gib = (
        3 * shard_gib
        + generation_gather_gib
        + 1.69
        + 2.5
        + reference_gib
    )
    peak_gib = max(training_phase_gib, rollout_phase_gib)
    return GRPOMemoryPrediction(
        stage="grpo",
        world_size=world_size,
        rollout_mode=rollout_mode,
        beta=beta,
        reference_gib=reference_gib,
        training_phase_gib=training_phase_gib,
        rollout_phase_gib=rollout_phase_gib,
        peak_gib=peak_gib,
        reserved_gib=peak_gib * 1.08,
    )


def assert_memory_fits(
    prediction: MemoryPrediction | GRPOMemoryPrediction,
    *,
    capacity_gib: float,
    model_name: str,
    state_precision: str,
    utilization_limit: float = 0.95,
) -> None:
    """Reject a launch whose predicted reserved peak exceeds safe capacity."""

    if capacity_gib <= 0:
        raise ValueError("capacity_gib must be positive")
    if not 0 < utilization_limit <= 1:
        raise ValueError("utilization_limit must be in (0, 1]")
    if prediction.reserved_gib > capacity_gib * utilization_limit:
        raise ValueError(
            f"Memory preflight failed for {model_name}: "
            f"world_size={prediction.world_size} state_precision={state_precision} "
            f"predicted_reserved_gib={prediction.reserved_gib:.2f} "
            f"capacity_gib={capacity_gib:.2f} exceeds "
            f"the {utilization_limit:.0%} utilization limit"
        )
