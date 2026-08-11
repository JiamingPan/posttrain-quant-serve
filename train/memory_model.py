"""Analytical GPU-memory estimates used as a preflight safety check."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from typing import Any, Sequence


GIB = 1024**3
BF16_BYTES = 2
QWEN3_8B_PARAMETERS = 8_190_735_360
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


def _parse_training_memory_args(
    stage: str,
    stage_args: Sequence[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
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
    if stage == "grpo":
        parser.add_argument("--beta", type=float, default=0.0)
        parser.add_argument(
            "--rollout_mode",
            choices=("auto", "reshard", "keep_unsharded"),
            default="auto",
        )
    parsed, _ = parser.parse_known_args(tuple(stage_args))
    return parsed


def preflight_launch(
    *,
    stage: str,
    world_size: int,
    stage_args: Sequence[str],
    device_name: str,
    capacity_gib: float,
) -> dict[str, Any]:
    """Evaluate a launcher memory check without loading model weights."""

    if stage not in {"sft", "grpo"}:
        raise ValueError("launch memory preflight supports only sft or grpo")
    _validate_inputs(QWEN3_8B_PARAMETERS, world_size)
    if not device_name:
        raise ValueError("device_name must not be empty")
    if capacity_gib <= 0:
        raise ValueError("capacity_gib must be positive")

    parsed = _parse_training_memory_args(stage, stage_args)
    base = {
        "capacity_gib": capacity_gib,
        "device_name": device_name,
        "model": parsed.model,
        "stage": stage,
        "status": "skipped_unknown_model_size",
        "world_size": world_size,
    }
    if parsed.model != "Qwen/Qwen3-8B":
        return base
    if parsed.resident_precision == "fp32" and world_size == 1:
        raise ValueError("the fp32-resident Qwen3-8B profile is rejected at world size one")
    if world_size == 1 and (
        "A100" not in device_name.upper() or capacity_gib < 79.0
    ):
        raise ValueError(
            "the Qwen3-8B world-size-1 point requires an A100 80 GiB GPU"
        )

    warning = None
    if world_size == 2 and capacity_gib < 50.0:
        warning = (
            "world-size-2 has a narrow memory margin on A40/40-48 GiB devices; "
            "keep activation checkpointing enabled"
        )

    if stage == "sft":
        prediction = predict_sft_peak(
            QWEN3_8B_PARAMETERS,
            world_size,
            checkpointing=parsed.activation_checkpointing,
        )
        assert_memory_fits(
            prediction,
            capacity_gib=capacity_gib,
            model_name=parsed.model,
            state_precision=parsed.resident_precision,
        )
        result = {
            **base,
            "activation_checkpointing": parsed.activation_checkpointing,
            "checkpointing": parsed.activation_checkpointing,
            "prediction": asdict(prediction),
            "predicted_allocated_gib": prediction.allocated_gib,
            "predicted_reserved_gib": prediction.reserved_gib,
            "resident_precision": parsed.resident_precision,
            "status": "fit",
        }
    else:
        keep_prediction = predict_grpo_peak(
            QWEN3_8B_PARAMETERS,
            world_size=world_size,
            beta=parsed.beta,
            rollout_mode="keep_unsharded",
        )
        if parsed.rollout_mode == "auto":
            rollout_mode = (
                "keep_unsharded"
                if keep_prediction.reserved_gib <= capacity_gib * 0.95
                else "reshard"
            )
        else:
            rollout_mode = parsed.rollout_mode
        prediction = predict_grpo_peak(
            QWEN3_8B_PARAMETERS,
            world_size=world_size,
            beta=parsed.beta,
            rollout_mode=rollout_mode,
        )
        assert_memory_fits(
            prediction,
            capacity_gib=capacity_gib,
            model_name=parsed.model,
            state_precision=parsed.resident_precision,
        )
        result = {
            **base,
            "activation_checkpointing": parsed.activation_checkpointing,
            "beta": parsed.beta,
            "keep_unsharded_reserved_gib": keep_prediction.reserved_gib,
            "prediction": asdict(prediction),
            "predicted_reserved_gib": prediction.reserved_gib,
            "resident_precision": parsed.resident_precision,
            "rollout_mode": rollout_mode,
            "status": "fit",
        }
    if warning is not None:
        result["warning"] = warning
    return result


def _build_preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3 FSDP2 launch memory preflight")
    parser.add_argument("--preflight-stage", choices=("sft", "grpo"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("stage_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_preflight_parser().parse_args(argv)
    stage_args = tuple(args.stage_args)
    if stage_args[:1] == ("--",):
        stage_args = stage_args[1:]

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available for launch memory preflight")
    properties = torch.cuda.get_device_properties(0)
    result = preflight_launch(
        stage=args.preflight_stage,
        world_size=args.world_size,
        stage_args=stage_args,
        device_name=properties.name,
        capacity_gib=properties.total_memory / GIB,
    )
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)
    if "warning" in result:
        print(f"WARNING: {result['warning']}", flush=True)


if __name__ == "__main__":
    main()
