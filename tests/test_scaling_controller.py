from __future__ import annotations

import json
from pathlib import Path

import pytest

import bench.scaling as scaling_module
from bench.scaling import ScalingConfig, build_worker_command
from train.run_tracking import RunIdentity


IMMUTABLE_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _config(tmp_path: Path, **overrides: object) -> ScalingConfig:
    values: dict[str, object] = {
        "output_dir": str(tmp_path),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    values.update(overrides)
    return ScalingConfig(**values)


def _pilot_record(world_size: int) -> dict[str, object]:
    return {
        "benchmark_mode": "pilot",
        "comparison_config_digest": "pilot-config",
        "communication_active_fraction": 0.2,
        "communication_exposed_fraction": 0.1,
        "global_batch_size": 8,
        "git_dirty": False,
        "gpu_name": "NVIDIA A40",
        "measure_steps": 10,
        "mfu": 0.25,
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": IMMUTABLE_REVISION,
        "rank_memory": [
            {
                "peak_allocated_bytes": 1_000 + rank,
                "peak_reserved_bytes": 1_100 + rank,
                "rank": rank,
                "step_seconds": [1.0] * 10,
            }
            for rank in range(world_size)
        ],
        "scaling_efficiency": 1.0 if world_size == 1 else 0.75,
        "step_time_mean_seconds": 1.0,
        "step_time_std_seconds": 0.05,
        "tokens_per_sec": 100.0 * world_size,
        "world_size": world_size,
    }


def test_full_controller_still_requires_the_complete_eight_gpu_sweep(
    tmp_path: Path,
) -> None:
    assert scaling_module.required_controller_gpu_count(_config(tmp_path)) == 8
    with pytest.raises(ValueError, match="1,2,4,8"):
        scaling_module.required_controller_gpu_count(
            _config(tmp_path, world_sizes=(1, 2))
        )


def test_pilot_controller_requires_only_its_largest_world_size(
    tmp_path: Path,
) -> None:
    assert scaling_module.required_controller_gpu_count(
        _config(tmp_path, pilot=True, world_sizes=(1, 2))
    ) == 2
    assert scaling_module.required_controller_gpu_count(
        _config(tmp_path, pilot=True, world_sizes=(2,))
    ) == 2


@pytest.mark.parametrize("world_sizes", [(2, 1), (1, 1)])
def test_pilot_controller_rejects_nonincreasing_world_sizes(
    tmp_path: Path,
    world_sizes: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        scaling_module.required_controller_gpu_count(
            _config(tmp_path, pilot=True, world_sizes=world_sizes)
        )


def test_qwen3_pilot_requires_only_the_committed_sft_gate(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        pilot=True,
        world_sizes=(2,),
        model="Qwen/Qwen3-8B",
        revision=IMMUTABLE_REVISION,
    )

    assert scaling_module.required_correctness_gate_names(config) == (
        "sft_gate.json",
    )


def test_qwen3_full_sweep_retains_both_gate_requirements(tmp_path: Path) -> None:
    config = _config(tmp_path, model="Qwen/Qwen3-8B")

    assert scaling_module.required_correctness_gate_names(config) == (
        "sft_gate.json",
        "grpo_gate.json",
    )


@pytest.mark.parametrize("revision", [None, "main", "deadbeef"])
def test_qwen3_pilot_requires_an_immutable_commit_revision(
    tmp_path: Path,
    revision: str | None,
) -> None:
    with pytest.raises(ValueError, match="40-character commit revision"):
        _config(
            tmp_path,
            pilot=True,
            world_sizes=(2,),
            model="Qwen/Qwen3-8B",
            revision=revision,
        )


def test_worker_command_propagates_pilot_mode(tmp_path: Path) -> None:
    command = build_worker_command(
        _config(tmp_path, pilot=True, world_sizes=(1, 2)),
        world_size=1,
        output_dir=tmp_path / "w1",
    )

    assert "--pilot" in command


def test_efficiency_is_null_without_world_size_one() -> None:
    records = [
        {
            "world_size": 2,
            "tokens_per_sec": 200.0,
            "scaling_efficiency": None,
        }
    ]

    scaling_module.apply_scaling_efficiencies(records)

    assert records[0]["scaling_efficiency"] is None


def test_efficiency_uses_world_size_one_when_present() -> None:
    records = [
        {
            "world_size": 1,
            "tokens_per_sec": 100.0,
            "scaling_efficiency": None,
        },
        {
            "world_size": 2,
            "tokens_per_sec": 150.0,
            "scaling_efficiency": None,
        },
    ]

    scaling_module.apply_scaling_efficiencies(records)

    assert records[0]["scaling_efficiency"] == pytest.approx(1.0)
    assert records[1]["scaling_efficiency"] == pytest.approx(0.75)


def test_controller_gpu_inventory_requires_the_exact_homogeneous_allocation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, pilot=True, world_sizes=(1, 2))

    assert scaling_module.validate_controller_gpu_inventory(
        config,
        visible_gpu_ids=("0", "1"),
        device_count=2,
        gpu_names=("NVIDIA A40", "NVIDIA A40"),
    ) == 2

    with pytest.raises(ValueError, match="exactly 2 visible GPUs"):
        scaling_module.validate_controller_gpu_inventory(
            config,
            visible_gpu_ids=("0",),
            device_count=1,
            gpu_names=("NVIDIA A40",),
        )
    with pytest.raises(ValueError, match="homogeneous"):
        scaling_module.validate_controller_gpu_inventory(
            config,
            visible_gpu_ids=("0", "1"),
            device_count=2,
            gpu_names=("NVIDIA A40", "NVIDIA A100-SXM4-80GB"),
        )


def test_controller_rejects_any_dirty_pilot_before_starting_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = RunIdentity(
        run_id="scaling-controller-test",
        stage="scaling-controller",
        git_commit=IMMUTABLE_REVISION,
        git_dirty=True,
        slurm_job_id=None,
        hostname="test-host",
        started_at_utc="2026-08-13T00:00:00Z",
    )
    monkeypatch.setattr(
        scaling_module,
        "capture_run_identity",
        lambda *_args, **_kwargs: identity,
    )
    monkeypatch.setattr(
        scaling_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("dirty pilot started a worker"),
    )

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        scaling_module.run_sweep_controller(
            _config(tmp_path, pilot=True, world_sizes=(1, 2))
        )


def test_validate_scaling_directory_uses_recorded_pilot_world_sizes(
    tmp_path: Path,
) -> None:
    records = [_pilot_record(world_size) for world_size in (1, 2)]
    (tmp_path / "run_config.json").write_text(
        json.dumps({"pilot": True, "world_sizes": [1, 2]}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "scaling.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert scaling_module.validate_scaling_directory(tmp_path) == records


def test_validate_scaling_directory_refuses_to_infer_a_pilot_subset(
    tmp_path: Path,
) -> None:
    records = [_pilot_record(world_size) for world_size in (1, 2)]
    (tmp_path / "scaling.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run config"):
        scaling_module.validate_scaling_directory(tmp_path)
