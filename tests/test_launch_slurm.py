from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_slurm.sh"


def run_launcher(
    *arguments: str,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
    ):
        environment.pop(name, None)
    environment.update({"SLURM_JOB_ID": "dry", "SLURM_NNODES": "1"})
    if env_overrides:
        environment.update(env_overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
def test_launcher_builds_one_node_torchrun_for_every_supported_size(gpus: int) -> None:
    result = run_launcher(
        "--stage",
        "sft",
        "--gpus",
        str(gpus),
        "--dry-run",
        "--",
        "--output_dir",
        "/tmp/sft",
    )

    assert result.returncode == 0, result.stderr
    assert f"--nproc-per-node={gpus}" in result.stdout
    assert "--module train.fsdp_sft" in result.stdout


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("sft", "--module train.fsdp_sft"),
        ("grpo", "--module train.fsdp_grpo"),
        ("correctness", "--module bench.correctness"),
        ("consolidate", "--module scripts.consolidate_dcp"),
        ("scaling-worker", "--module bench.scaling --worker"),
    ],
)
def test_launcher_maps_worker_stages_to_the_existing_modules(
    stage: str,
    expected: str,
) -> None:
    result = run_launcher(
        "--stage",
        stage,
        "--gpus",
        "1",
        "--dry-run",
        "--",
        "--output_dir",
        "/tmp/run",
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


def test_scaling_controller_uses_python_and_requires_the_full_eight_gpu_allocation() -> None:
    valid = run_launcher(
        "--stage",
        "scaling",
        "--gpus",
        "8",
        "--dry-run",
        "--",
        "--output_dir",
        "/tmp/scaling",
    )
    invalid = run_launcher(
        "--stage",
        "scaling",
        "--gpus",
        "4",
        "--dry-run",
        "--",
        "--output_dir",
        "/tmp/scaling",
    )

    assert valid.returncode == 0, valid.stderr
    assert "python -m bench.scaling" in valid.stdout
    assert "torchrun" not in valid.stdout
    assert invalid.returncode != 0
    assert "requires --gpus 8" in invalid.stderr


@pytest.mark.parametrize("gpus", ["0", "3", "16", "eight"])
def test_launcher_rejects_unsupported_gpu_counts(gpus: str) -> None:
    result = run_launcher("--stage", "sft", "--gpus", gpus, "--dry-run")

    assert result.returncode != 0
    assert "one of 1, 2, 4, or 8" in result.stderr


def test_launcher_rejects_multiple_nodes_and_visible_device_mismatches() -> None:
    multi_node = run_launcher(
        "--stage",
        "sft",
        "--gpus",
        "2",
        "--dry-run",
        env_overrides={"SLURM_NNODES": "2"},
    )
    mismatch = run_launcher(
        "--stage",
        "sft",
        "--gpus",
        "4",
        "--dry-run",
        env_overrides={"CUDA_VISIBLE_DEVICES": "0,1"},
    )

    assert multi_node.returncode != 0
    assert "single Slurm node" in multi_node.stderr
    assert mismatch.returncode != 0
    assert "visible GPU count" in mismatch.stderr


def test_launcher_preserves_stage_argument_boundaries_and_prints_nccl_controls() -> None:
    result = run_launcher(
        "--stage",
        "sft",
        "--gpus",
        "1",
        "--distributed-timeout-seconds",
        "900",
        "--dry-run",
        "--",
        "--output_dir",
        "/tmp/run with spaces",
    )

    assert result.returncode == 0, result.stderr
    assert "stage_arg[1]=/tmp/run with spaces" in result.stdout
    assert "NCCL_DEBUG=INFO" in result.stdout
    assert "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" in result.stdout
    assert "TORCH_NCCL_BLOCKING_WAIT=1" in result.stdout
    assert "PQS_DISTRIBUTED_TIMEOUT_SECONDS=900" in result.stdout
    assert "--timeout_seconds 900" in result.stdout


def test_launcher_rejects_unknown_stage() -> None:
    result = run_launcher(
        "--stage",
        "mystery",
        "--gpus",
        "1",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "unknown stage" in result.stderr
