from __future__ import annotations

import pytest
import torch

import bench.scaling as scaling_module
from bench.scaling import (
    communication_fractions,
    compute_mfu,
    deduplicated_storage_bytes,
    fixed_accumulation_steps,
    hardware_peak_bf16_tflops,
    measure_memory_components,
    merge_intervals,
    scaling_efficiency,
    validate_scaling_records,
)


def test_merge_intervals_unions_nested_touching_and_disjoint_ranges() -> None:
    assert merge_intervals([(8, 10), (1, 4), (3, 8), (20, 21)]) == [
        (1.0, 10.0),
        (20.0, 21.0),
    ]


def test_exposed_communication_subtracts_compute_overlap() -> None:
    communication = [(0, 10), (20, 30)]
    compute = [(5, 25)]

    active, exposed = communication_fractions(
        communication,
        compute,
        step_window=(0, 40),
    )

    assert active == pytest.approx(0.50)
    assert exposed == pytest.approx(0.25)


def test_communication_fraction_rejects_an_invalid_step_window() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        communication_fractions([], [], step_window=(10, 10))


def test_mfu_uses_aggregate_dense_peak() -> None:
    mfu = compute_mfu(
        parameters=8_190_735_360,
        useful_tokens=16_384,
        seconds=10.0,
        world_size=4,
        peak_bf16_tflops_per_gpu=312.0,
    )

    expected = 6 * 8_190_735_360 * 16_384 / (10 * 4 * 312e12)
    assert mfu == pytest.approx(expected)


def test_storage_bytes_deduplicates_views() -> None:
    tensor = torch.zeros(32, dtype=torch.bfloat16)

    assert deduplicated_storage_bytes([tensor, tensor.view(4, 8)]) == 64


def test_memory_components_inventory_storage_and_leave_an_auditable_residual() -> None:
    model = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.state[parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }

    components = measure_memory_components(
        model,
        optimizer,
        activation_bytes=12,
        collective_bytes=20,
        peak_allocated_bytes=100,
    )

    assert components == {
        "params_bytes": 16,
        "grads_bytes": 16,
        "optimizer_bytes": 36,
        "activations_bytes": 12,
        "collectives_bytes": 20,
        "other_bytes": 0,
        "category_sum_bytes": 100,
        "peak_allocated_bytes": 100,
    }


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [(1, 8), (2, 4), (4, 2), (8, 1)],
)
def test_fixed_global_batch_derives_the_approved_accumulation(
    world_size: int,
    expected: int,
) -> None:
    assert fixed_accumulation_steps(
        global_batch_size=8,
        world_size=world_size,
        micro_batch_size=1,
    ) == expected


def test_fixed_global_batch_rejects_non_integral_accumulation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        fixed_accumulation_steps(
            global_batch_size=10,
            world_size=4,
            micro_batch_size=1,
        )


def test_scaling_efficiency_uses_the_world_size_one_throughput() -> None:
    assert scaling_efficiency(
        throughput=300.0,
        baseline_throughput=100.0,
        world_size=4,
    ) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NVIDIA A100-SXM4-80GB", 312.0),
        ("NVIDIA A40", 149.7),
    ],
)
def test_known_gpu_names_resolve_dense_bf16_peak(name: str, expected: float) -> None:
    assert hardware_peak_bf16_tflops(name) == expected


def test_unknown_gpu_requires_an_explicit_peak() -> None:
    with pytest.raises(ValueError, match="peak_bf16_tflops"):
        hardware_peak_bf16_tflops("NVIDIA Future GPU")
    assert hardware_peak_bf16_tflops(
        "NVIDIA Future GPU",
        override=999.0,
    ) == 999.0


def _valid_record(world_size: int) -> dict[str, object]:
    return {
        "benchmark_mode": "full",
        "comparison_config_digest": "same-config",
        "communication_active_fraction": 0.2,
        "communication_exposed_fraction": 0.1,
        "global_batch_size": 8,
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "measure_steps": 10,
        "mfu": 0.25,
        "rank_memory": [
            {
                "peak_allocated_bytes": 1_000 + rank,
                "peak_reserved_bytes": 1_100 + rank,
                "rank": rank,
                "step_seconds": [1.0] * 10,
            }
            for rank in range(world_size)
        ],
        "scaling_efficiency": 1.0,
        "step_time_mean_seconds": 1.0,
        "step_time_std_seconds": 0.05,
        "tokens_per_sec": 100.0 * world_size,
        "world_size": world_size,
    }


def test_scaling_record_validation_requires_all_four_homogeneous_points() -> None:
    records = [_valid_record(world_size) for world_size in (1, 2, 4, 8)]

    validate_scaling_records(records)

    records[2]["gpu_name"] = "NVIDIA A40"
    with pytest.raises(ValueError, match="homogeneous GPU"):
        validate_scaling_records(records)


def test_scaling_record_validation_requires_rank_memory_from_every_rank() -> None:
    records = [_valid_record(world_size) for world_size in (1, 2, 4, 8)]
    records[-1]["rank_memory"] = records[-1]["rank_memory"][:-1]

    with pytest.raises(ValueError, match="rank memory"):
        validate_scaling_records(records)


def test_pilot_validation_accepts_a_two_point_lower_bound() -> None:
    records = [_valid_record(world_size) for world_size in (1, 2)]
    for record in records:
        record["benchmark_mode"] = "pilot"
    records[0]["scaling_efficiency"] = 1.0
    records[1]["scaling_efficiency"] = 0.75

    scaling_module.validate_pilot_scaling_records(
        records,
        expected_world_sizes=(1, 2),
    )


def test_isolated_pilot_point_requires_null_efficiency() -> None:
    record = _valid_record(2)
    record["benchmark_mode"] = "pilot"
    record["scaling_efficiency"] = None

    scaling_module.validate_pilot_scaling_records(
        [record],
        expected_world_sizes=(2,),
    )

    record["scaling_efficiency"] = 1.0
    with pytest.raises(ValueError, match="world-size-1 baseline"):
        scaling_module.validate_pilot_scaling_records(
            [record],
            expected_world_sizes=(2,),
        )


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        (metric, value)
        for metric in (
            "tokens_per_sec",
            "mfu",
            "communication_active_fraction",
            "communication_exposed_fraction",
            "step_time_mean_seconds",
        )
        for value in (float("nan"), float("inf"))
    ],
)
def test_pilot_validation_rejects_nonfinite_metrics(metric: str, value: float) -> None:
    record = _valid_record(2)
    record["benchmark_mode"] = "pilot"
    record["scaling_efficiency"] = None
    record[metric] = value

    with pytest.raises(ValueError, match=metric):
        scaling_module.validate_pilot_scaling_records(
            [record],
            expected_world_sizes=(2,),
        )


def test_pilot_validation_requires_every_measured_step_duration() -> None:
    record = _valid_record(2)
    record["benchmark_mode"] = "pilot"
    record["scaling_efficiency"] = None
    record["rank_memory"][0]["step_seconds"] = [1.0] * 9

    with pytest.raises(ValueError, match="measured step durations"):
        scaling_module.validate_pilot_scaling_records(
            [record],
            expected_world_sizes=(2,),
        )


@pytest.mark.parametrize(
    "expected_world_sizes",
    [(), (2, 1), (1, 1), (3,)],
)
def test_pilot_validation_rejects_an_invalid_requested_subset(
    expected_world_sizes: tuple[int, ...],
) -> None:
    record = _valid_record(2)
    record["benchmark_mode"] = "pilot"
    record["scaling_efficiency"] = None

    with pytest.raises(ValueError, match="strictly increasing supported subset"):
        scaling_module.validate_pilot_scaling_records(
            [record],
            expected_world_sizes=expected_world_sizes,
        )
