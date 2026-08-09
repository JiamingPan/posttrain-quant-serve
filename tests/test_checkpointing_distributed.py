from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from train.checkpointing import (
    TrainProgress,
    checkpoint_manifest,
    load_dcp_checkpoint,
    load_dcp_model_only,
    resolve_hf_checkpoint_source,
    resolve_resume_checkpoint,
    save_dcp_checkpoint,
)
from train.gsm8k_data import CheckpointableDistributedSampler


def _manifest(step: int, *, world_size: int = 2) -> dict[str, object]:
    return {
        "format_version": 1,
        "global_step": step,
        "world_size": world_size,
        "bitwise_resume": True,
        "model_state_keys": 2,
        "optimizer_state_keys": 1,
        "progress": {
            "global_step": step,
            "consumed_tokens": step * 8,
            "sampler_state": {"global_cursor": step * 2},
            "rng_states": [],
            "config": {"model": "tiny"},
            "source_digests": {"model": "abc123"},
        },
    }


def _publish_fake_checkpoint(root, step: int, *, complete: bool = True):
    checkpoint = root / f"step-{step:08d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(
        json.dumps(_manifest(step), sort_keys=True),
        encoding="utf-8",
    )
    if complete:
        (checkpoint / "_SUCCESS").write_text("\n", encoding="utf-8")
    return checkpoint


def test_latest_resume_ignores_partial_directories_and_sorts_numeric_steps(tmp_path) -> None:
    _publish_fake_checkpoint(tmp_path, 2)
    expected = _publish_fake_checkpoint(tmp_path, 10)
    _publish_fake_checkpoint(tmp_path, 99, complete=False)
    (tmp_path / ".step-00000100-deadbeef.tmp").mkdir()

    assert resolve_resume_checkpoint("latest", output_dir=tmp_path) == expected
    assert resolve_resume_checkpoint("none", output_dir=tmp_path) is None


def test_explicit_resume_refuses_checkpoint_without_success_marker(tmp_path) -> None:
    partial = _publish_fake_checkpoint(tmp_path, 4, complete=False)

    with pytest.raises(ValueError, match="not a published checkpoint"):
        resolve_resume_checkpoint(str(partial), output_dir=tmp_path)


def test_local_hf_source_resolves_to_an_immutable_content_digest(tmp_path) -> None:
    model_dir = tmp_path / "hf"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights-v1")

    first = resolve_hf_checkpoint_source(model_dir)
    claimed = resolve_hf_checkpoint_source(model_dir, revision="external-label")
    weights.write_bytes(b"weights-v2")
    second = resolve_hf_checkpoint_source(model_dir)

    assert first.path == model_dir.resolve()
    assert len(first.revision) == 64
    assert claimed.revision == first.revision
    assert first.revision != second.revision


def test_resolved_hf_cache_snapshot_preserves_its_commit_revision(tmp_path) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "models--org--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"weights")

    source = resolve_hf_checkpoint_source(snapshot, revision=commit)

    assert source.revision == commit


def test_manifest_validates_directory_step(tmp_path) -> None:
    checkpoint = _publish_fake_checkpoint(tmp_path, 7)
    payload = _manifest(8)
    (checkpoint / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="directory step 7.*manifest step 8"):
        checkpoint_manifest(checkpoint)


@pytest.mark.filterwarnings("ignore:torch.distributed is unavailable or uninitialized")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
def test_single_process_dcp_restores_model_optimizer_scheduler_sampler_and_rng(tmp_path) -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.25), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    sampler = CheckpointableDistributedSampler(16, rank=0, world_size=1, seed=5)
    sampler.next_indices(2)

    inputs = torch.randn(2, 3)
    model(inputs).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    saved_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer_keys_before = set(optimizer.state_dict()["state"])
    progress = TrainProgress(
        global_step=1,
        consumed_tokens=8,
        sampler_state=sampler.state_dict(),
        rng_states=[],
        config={"model": "tiny", "world_size": 1},
        source_digests={"model": "abc123", "data": "def456"},
    )

    checkpoint = save_dcp_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        progress=progress,
    )
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_indices = sampler.next_indices(2)

    model(torch.randn(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    loaded = load_dcp_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        expected_config=progress.config,
        expected_source_digests=progress.source_digests,
    )

    assert (checkpoint / "_SUCCESS").is_file()
    assert loaded.global_step == 1
    assert loaded.bitwise_resume is True
    assert scheduler.last_epoch == 1
    assert sampler.next_indices(2) == expected_indices
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert optimizer_keys_before == set(optimizer.state_dict()["state"])
    for actual, expected in zip(model.parameters(), saved_parameters):
        assert torch.equal(actual, expected)


@pytest.mark.filterwarnings("ignore:torch.distributed is unavailable or uninitialized")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
def test_model_only_load_initializes_a_policy_from_a_training_checkpoint(tmp_path) -> None:
    torch.manual_seed(31)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    sampler = CheckpointableDistributedSampler(8, rank=0, world_size=1, seed=2)
    model(torch.randn(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected = [parameter.detach().clone() for parameter in model.parameters()]
    checkpoint = save_dcp_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        progress=TrainProgress(
            global_step=1,
            consumed_tokens=4,
            sampler_state=sampler.state_dict(),
            rng_states=[],
            config={"stage": "sft"},
            source_digests={"model": "base", "data": "fixed"},
        ),
    )
    initialized = nn.Linear(3, 2)

    load_dcp_model_only(checkpoint, model=initialized)

    for actual, wanted in zip(initialized.parameters(), expected):
        assert torch.equal(actual, wanted)


@pytest.mark.cuda
@pytest.mark.distributed
def test_dcp_resume_restores_exact_next_step(torchrun_result, tmp_path) -> None:
    row = torchrun_result(
        "tests/workers/checkpoint_worker.py",
        nproc=2,
        mode="exact",
        output_dir=tmp_path,
    )

    assert row["success_marker_present"] is True
    assert row["partial_directory_selected"] is False
    assert row["next_loss_resumed"] == pytest.approx(
        row["next_loss_uninterrupted"],
        abs=1e-6,
    )
    assert row["optimizer_state_keys_before"] == row["optimizer_state_keys_after"]
    assert row["sampler_cursor_before"] == row["sampler_cursor_after"]
    assert row["bitwise_resume"] is True


@pytest.mark.cuda
@pytest.mark.distributed
def test_dcp_can_reshard_model_and_optimizer_at_a_new_world_size(
    torchrun_result,
    tmp_path,
) -> None:
    torchrun_result(
        "tests/workers/checkpoint_worker.py",
        nproc=2,
        mode="save",
        output_dir=tmp_path,
    )
    row = torchrun_result(
        "tests/workers/checkpoint_worker.py",
        nproc=1,
        mode="load-reshard",
        output_dir=tmp_path,
    )

    assert row["model_load_finite"] is True
    assert row["optimizer_state_restored"] is True
    assert row["bitwise_resume"] is False


@pytest.mark.cuda
@pytest.mark.distributed
def test_hugging_face_safetensors_load_directly_into_dtensor_shards(
    torchrun_result,
    tmp_path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("transformers")
    from tests.tiny_qwen import write_tiny_qwen3

    model_dir = write_tiny_qwen3(tmp_path / "tiny-hf")
    row = torchrun_result(
        "tests/workers/checkpoint_worker.py",
        nproc=2,
        mode="hf-load",
        output_dir=tmp_path / "unused",
        hf_model_dir=model_dir,
    )

    assert row["all_parameters_dtensor"] is True
    assert row["loaded_weight_matches_source"] is True
    assert len(row["resolved_revision"]) == 64
