"""Sharded DCP persistence and direct Hugging Face safetensor loading."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
from typing import Any, Mapping, Protocol
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)


FORMAT_VERSION = 1
CHECKPOINT_RE = re.compile(r"^step-(\d+)$")


class Stateful(Protocol):
    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs: Any) -> Any: ...


@dataclass
class TrainProgress:
    global_step: int
    consumed_tokens: int
    sampler_state: dict[str, Any]
    rng_states: list[dict[str, Any]]
    config: dict[str, Any]
    source_digests: dict[str, str]
    bitwise_resume: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "consumed_tokens": self.consumed_tokens,
            "sampler_state": self.sampler_state,
            "rng_states": self.rng_states,
            "config": self.config,
            "source_digests": self.source_digests,
            "bitwise_resume": self.bitwise_resume,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainProgress:
        return cls(
            global_step=int(payload["global_step"]),
            consumed_tokens=int(payload["consumed_tokens"]),
            sampler_state=dict(payload["sampler_state"]),
            rng_states=[dict(state) for state in payload["rng_states"]],
            config=dict(payload["config"]),
            source_digests={
                str(key): str(value)
                for key, value in dict(payload["source_digests"]).items()
            },
            bitwise_resume=bool(payload.get("bitwise_resume", True)),
        )


@dataclass(frozen=True)
class HFCheckpointSource:
    path: Path
    revision: str


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _pickle_to_text(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode("ascii")


def _pickle_from_text(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _tensor_state_to_text(state: torch.Tensor) -> str:
    return base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii")


def _tensor_state_from_text(state: str) -> torch.Tensor:
    return torch.tensor(list(base64.b64decode(state.encode("ascii"))), dtype=torch.uint8)


def _capture_local_rng_state() -> dict[str, Any]:
    cuda_states: list[str] = []
    if torch.cuda.is_available():
        cuda_states = [_tensor_state_to_text(state) for state in torch.cuda.get_rng_state_all()]
    return {
        "python": _pickle_to_text(random.getstate()),
        "numpy": _pickle_to_text(np.random.get_state()),
        "torch_cpu": _tensor_state_to_text(torch.get_rng_state()),
        "torch_cuda": cuda_states,
    }


def _gather_rng_states() -> list[dict[str, Any]]:
    local_state = _capture_local_rng_state()
    _, world_size = _rank_world()
    if world_size == 1:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    return [state for state in gathered if state is not None]


def _restore_local_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(_pickle_from_text(str(state["python"])))
    np.random.set_state(_pickle_from_text(str(state["numpy"])))
    torch.set_rng_state(_tensor_state_from_text(str(state["torch_cpu"])))
    cuda_states = [
        _tensor_state_from_text(str(encoded)) for encoded in list(state["torch_cuda"])
    ]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)


class _TrainerState:
    def __init__(
        self,
        scheduler: Stateful,
        sampler: Stateful,
        progress: TrainProgress,
        *,
        allow_world_size_change: bool,
    ) -> None:
        self.scheduler = scheduler
        self.sampler = sampler
        self.progress = progress
        self.allow_world_size_change = allow_world_size_change

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "progress": self.progress.as_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.scheduler.load_state_dict(dict(state_dict["scheduler"]))
        self.progress = TrainProgress.from_dict(state_dict["progress"])
        try:
            self.sampler.load_state_dict(
                self.progress.sampler_state,
                allow_world_size_change=self.allow_world_size_change,
            )
        except TypeError:
            if self.allow_world_size_change:
                raise TypeError(
                    "sampler does not support loading across a world-size change"
                ) from None
            self.sampler.load_state_dict(self.progress.sampler_state)


def _optimizer_signature(optimizer_state: Mapping[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for parameter_name, state in dict(optimizer_state.get("state", {})).items():
        state_signature: dict[str, Any] = {}
        for state_name, value in dict(state).items():
            if isinstance(value, torch.Tensor):
                state_signature[str(state_name)] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            else:
                state_signature[str(state_name)] = {"type": type(value).__name__}
        signature[str(parameter_name)] = state_signature
    return signature


def _validate_json(value: Any, *, name: str) -> None:
    try:
        json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON serializable") from error


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_optimizer_step_boundary(model: nn.Module) -> None:
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(
            "DCP checkpoints may only be saved at an optimizer-step boundary after zero_grad"
        )


def save_dcp_checkpoint(
    output_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Stateful,
    sampler: Stateful,
    progress: TrainProgress,
) -> Path:
    """Collectively save shards and publish only after every rank succeeds."""

    if progress.global_step < 0 or progress.consumed_tokens < 0:
        raise ValueError("training progress counters must be non-negative")
    _assert_optimizer_step_boundary(model)
    rank, world_size = _rank_world()
    output_path = Path(output_dir)
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
    _barrier()

    token: list[str | None] = [uuid.uuid4().hex if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(token, src=0)
    checkpoint_token = token[0]
    if checkpoint_token is None:
        raise RuntimeError("rank zero did not provide a checkpoint UUID")
    final_path = output_path / f"step-{progress.global_step:08d}"
    temporary_path = output_path / (
        f".step-{progress.global_step:08d}-{checkpoint_token}.tmp"
    )
    if final_path.exists():
        raise FileExistsError(f"checkpoint already exists: {final_path}")

    saved_progress = replace(
        progress,
        sampler_state=dict(sampler.state_dict()),
        rng_states=_gather_rng_states(),
        bitwise_resume=True,
    )
    _validate_json(saved_progress.as_dict(), name="TrainProgress")
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(strict=True),
    )
    trainer_state = _TrainerState(
        scheduler,
        sampler,
        saved_progress,
        allow_world_size_change=False,
    )
    dcp.save(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "trainer": trainer_state,
        },
        checkpoint_id=temporary_path,
    )
    _barrier()

    manifest = {
        "format_version": FORMAT_VERSION,
        "global_step": saved_progress.global_step,
        "world_size": world_size,
        "bitwise_resume": True,
        "model_state_keys": len(model_state),
        "optimizer_state_keys": len(optimizer_state.get("state", {})),
        "optimizer_signature": _optimizer_signature(optimizer_state),
        "progress": saved_progress.as_dict(),
    }
    if rank == 0:
        _write_json_atomic(temporary_path / "manifest.json", manifest)
        success_path = temporary_path / "_SUCCESS"
        with success_path.open("x", encoding="utf-8") as handle:
            handle.write(f"step={saved_progress.global_step}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)
    _barrier()
    return final_path


def checkpoint_manifest(checkpoint: str | Path) -> dict[str, Any]:
    """Read and validate the publication marker and numeric step identity."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_dir() or not (checkpoint_path / "_SUCCESS").is_file():
        raise ValueError(f"{checkpoint_path} is not a published checkpoint")
    match = CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError(f"checkpoint directory has an invalid name: {checkpoint_path.name}")
    manifest_path = checkpoint_path / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint has an invalid manifest: {checkpoint_path}") from error
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"checkpoint has an unsupported format version: {checkpoint_path}")
    directory_step = int(match.group(1))
    manifest_step = int(payload.get("global_step", -1))
    if directory_step != manifest_step:
        raise ValueError(
            f"checkpoint directory step {directory_step} does not match "
            f"manifest step {manifest_step}"
        )
    if int(payload.get("world_size", 0)) <= 0:
        raise ValueError("checkpoint manifest world_size must be positive")
    if not isinstance(payload.get("progress"), dict):
        raise ValueError("checkpoint manifest is missing training progress")
    return payload


def resolve_resume_checkpoint(
    resume: str | Path,
    *,
    output_dir: str | Path,
) -> Path | None:
    """Resolve none/latest/explicit while ignoring unpublished directories."""

    resume_value = str(resume)
    if resume_value == "none":
        return None
    output_path = Path(output_dir)
    if resume_value == "latest":
        candidates: list[tuple[int, Path]] = []
        if output_path.is_dir():
            for path in output_path.iterdir():
                match = CHECKPOINT_RE.fullmatch(path.name)
                if match and path.is_dir() and (path / "_SUCCESS").is_file():
                    candidates.append((int(match.group(1)), path))
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: item[0])[1]
        checkpoint_manifest(selected)
        return selected

    explicit = Path(resume_value)
    if not explicit.is_absolute() and not explicit.exists():
        explicit = output_path / explicit
    checkpoint_manifest(explicit)
    return explicit


def _initialize_adamw_state(optimizer: torch.optim.Optimizer) -> None:
    """Allocate sharded destinations before DCP loads AdamW moment tensors."""

    if optimizer.state:
        return
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("checkpoint resume currently requires torch.optim.AdamW")
    for group in optimizer.param_groups:
        amsgrad = bool(group.get("amsgrad", False))
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            state = optimizer.state[parameter]
            state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(parameter)


def _configs_match(
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    allow_world_size_change: bool,
) -> bool:
    saved_config = dict(saved)
    expected_config = dict(expected)
    if allow_world_size_change:
        saved_config.pop("world_size", None)
        expected_config.pop("world_size", None)
    return saved_config == expected_config


def load_dcp_checkpoint(
    checkpoint: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Stateful,
    sampler: Stateful,
    expected_config: Mapping[str, Any],
    expected_source_digests: Mapping[str, str],
    allow_world_size_change: bool = False,
) -> TrainProgress:
    """Collectively restore sharded state before the next backward pass."""

    checkpoint_path = Path(checkpoint)
    manifest = checkpoint_manifest(checkpoint_path)
    manifest_progress = TrainProgress.from_dict(manifest["progress"])
    if not _configs_match(
        manifest_progress.config,
        expected_config,
        allow_world_size_change=allow_world_size_change,
    ):
        raise ValueError("checkpoint configuration does not match the resolved run config")
    if manifest_progress.source_digests != dict(expected_source_digests):
        raise ValueError("checkpoint source digests do not match the current sources")

    rank, world_size = _rank_world()
    saved_world_size = int(manifest["world_size"])
    if world_size != saved_world_size and not allow_world_size_change:
        raise ValueError(
            f"checkpoint world_size={saved_world_size} cannot resume at "
            f"world_size={world_size} without allow_world_size_change"
        )

    _assert_optimizer_step_boundary(model)
    _initialize_adamw_state(optimizer)
    model_state, optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(strict=True),
    )
    trainer_state = _TrainerState(
        scheduler,
        sampler,
        manifest_progress,
        allow_world_size_change=allow_world_size_change,
    )
    dcp.load(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "trainer": trainer_state,
        },
        checkpoint_id=checkpoint_path,
    )
    incompatible = set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=StateDictOptions(strict=True),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict DCP load produced incompatible model keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    loaded_progress = trainer_state.progress
    if loaded_progress.global_step != int(manifest["global_step"]):
        raise RuntimeError("DCP progress step does not match the checkpoint manifest")
    loaded_optimizer_state = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(strict=True),
    )[1]
    actual_signature = _optimizer_signature(loaded_optimizer_state)
    expected_signature = manifest.get("optimizer_signature", actual_signature)
    if actual_signature != expected_signature:
        raise RuntimeError("restored optimizer tensor shapes or dtypes do not match the manifest")
    if len(loaded_optimizer_state.get("state", {})) != int(
        manifest["optimizer_state_keys"]
    ):
        raise RuntimeError("restored optimizer FQN count does not match the manifest")

    loaded_progress.bitwise_resume = world_size == saved_world_size
    if not loaded_progress.rng_states:
        raise RuntimeError("checkpoint does not contain per-rank RNG states")
    rng_state = loaded_progress.rng_states[rank % len(loaded_progress.rng_states)]
    _restore_local_rng_state(rng_state)
    return loaded_progress


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.name == "config.json" or candidate.suffix == ".safetensors"
    )
    if not files:
        raise ValueError(f"no Hugging Face model files found in {path}")
    for file_path in files:
        digest.update(file_path.name.encode())
        with file_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_hf_source(
    model_name_or_path: str | Path,
    *,
    revision: str | None,
    local_files_only: bool,
) -> HFCheckpointSource:
    candidate = Path(model_name_or_path)
    if candidate.is_dir():
        return HFCheckpointSource(candidate.resolve(), revision or _directory_digest(candidate))

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=str(model_name_or_path),
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else revision
    if resolved_revision is None:
        raise RuntimeError("could not resolve the immutable Hugging Face snapshot revision")
    return HFCheckpointSource(snapshot, resolved_revision)


def load_hf_weights_into_shards(
    model: nn.Module,
    model_name_or_path: str | Path,
    *,
    device: torch.device,
    revision: str | None = None,
    local_files_only: bool = False,
) -> HFCheckpointSource:
    """Load safetensors directly into an already-FSDP2-sharded meta model."""

    source = _resolve_hf_source(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
    )
    reader_type = getattr(dcp, "HuggingFaceStorageReader", None)
    if reader_type is None:
        raise RuntimeError("HuggingFaceStorageReader requires the pinned PyTorch 2.8 runtime")
    model.to_empty(device=device)
    model_state = get_model_state_dict(
        model,
        options=StateDictOptions(strict=True),
    )
    dcp.load(model_state, storage_reader=reader_type(path=str(source.path)))
    incompatible = set_model_state_dict(
        model,
        model_state,
        options=StateDictOptions(strict=True),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict Hugging Face load produced incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return source
