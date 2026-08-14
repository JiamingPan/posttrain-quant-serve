"""Consolidate one sharded DCP model into a verified Hugging Face directory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence
import uuid

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from train.checkpointing import (
    checkpoint_manifest,
    load_dcp_model_only,
    resolve_hf_checkpoint_source,
)
from train.fsdp_grpo import PolicySource, resolve_policy_source
from train.fsdp_utils import (
    FSDPSettings,
    destroy_distributed,
    fully_shard_qwen,
    init_distributed,
)


GIB = 1024**3


def required_host_memory_bytes(
    parameter_count: int,
    *,
    bytes_per_parameter: int,
    workspace_factor: float = 1.25,
) -> int:
    if parameter_count <= 0 or bytes_per_parameter <= 0:
        raise ValueError("parameter count and element size must be positive")
    if workspace_factor < 1:
        raise ValueError("workspace_factor must be at least one")
    return int(parameter_count * bytes_per_parameter * workspace_factor)


def assert_host_memory_for_consolidation(
    *,
    parameter_count: int,
    bytes_per_parameter: int,
    available_bytes: int,
) -> int:
    required = required_host_memory_bytes(
        parameter_count,
        bytes_per_parameter=bytes_per_parameter,
    )
    if available_bytes < required:
        raise MemoryError(
            "DCP consolidation host-memory preflight failed: "
            f"{required / GIB:.2f} GiB required, "
            f"{available_bytes / GIB:.2f} GiB available"
        )
    return required


def available_host_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError) as error:
        raise RuntimeError("could not determine available host memory") from error


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _selected_tensor_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    keys = sorted(key for key, value in state.items() if isinstance(value, torch.Tensor))
    if not keys:
        raise RuntimeError("gathered model state contains no tensors")
    selected_indices = sorted({0, len(keys) // 2, len(keys) - 1})
    return {keys[index]: tensor_sha256(state[keys[index]]) for index in selected_indices}


def _resolve_consolidation_source(
    checkpoint: Path,
    source_model: str | None,
    revision: str | None,
) -> PolicySource:
    if source_model is None:
        return resolve_policy_source(checkpoint, revision=revision)
    checkpoint_manifest(checkpoint)
    source = resolve_hf_checkpoint_source(source_model, revision=revision)
    manifest_digest = hashlib.sha256(
        (checkpoint / "manifest.json").read_bytes()
    ).hexdigest()
    return PolicySource(
        kind="dcp",
        weights_path=checkpoint.resolve(),
        architecture_path=source.path,
        model_revision=source.revision,
        checkpoint_digest=manifest_digest,
    )


def _preflight_collective(
    *,
    parameter_count: int,
    bytes_per_parameter: int,
    ctx: Any,
) -> int:
    status: list[dict[str, Any] | None] = [None]
    if ctx.rank == 0:
        try:
            available = available_host_memory_bytes()
            required = assert_host_memory_for_consolidation(
                parameter_count=parameter_count,
                bytes_per_parameter=bytes_per_parameter,
                available_bytes=available,
            )
            status[0] = {
                "error": None,
                "available_bytes": available,
                "required_bytes": required,
            }
        except Exception as error:  # Propagate the same preflight failure to every rank.
            status[0] = {"error": f"{type(error).__name__}: {error}"}
    if ctx.world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is None:
        raise RuntimeError("rank zero did not publish consolidation preflight status")
    if status[0].get("error"):
        raise MemoryError(str(status[0]["error"]))
    return int(status[0]["required_bytes"])


def consolidate_checkpoint(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    ctx: Any,
    source_model: str | None = None,
    revision: str | None = None,
    max_shard_size: str = "5GB",
) -> dict[str, Any]:
    """Collectively gather model state, publish HF files, and verify rank zero."""

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(checkpoint).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"consolidation destination already exists: {destination}")
    source = _resolve_consolidation_source(checkpoint_path, source_model, revision)
    model_config = AutoConfig.from_pretrained(source.architecture_path)
    model_config.use_cache = False
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            model_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    fully_shard_qwen(model, ctx, FSDPSettings())
    model.to_empty(device=ctx.device)
    load_dcp_model_only(checkpoint_path, model=model)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(source.architecture_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fixed = tokenizer(
        "Consolidation verification",
        return_tensors="pt",
        add_special_tokens=True,
    )
    fixed_input_ids = fixed["input_ids"].to(ctx.device)
    fixed_attention_mask = fixed.get("attention_mask", torch.ones_like(fixed_input_ids)).to(
        ctx.device
    )
    with torch.inference_mode():
        distributed_logits = model(
            input_ids=fixed_input_ids,
            attention_mask=fixed_attention_mask,
            use_cache=False,
        ).logits.float().cpu()

    required_host_bytes = _preflight_collective(
        parameter_count=parameter_count,
        bytes_per_parameter=torch.tensor([], dtype=torch.bfloat16).element_size(),
        ctx=ctx,
    )
    full_state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    publication: list[dict[str, Any] | None] = [None]
    temporary = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.tmp"
    if ctx.rank == 0:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            expected_hashes = _selected_tensor_hashes(dict(full_state))
            model.save_pretrained(
                temporary,
                state_dict=full_state,
                safe_serialization=True,
                max_shard_size=max_shard_size,
            )
            tokenizer.save_pretrained(temporary)
            del full_state
            gc.collect()

            reloaded = AutoModelForCausalLM.from_pretrained(
                temporary,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            reloaded.eval()
            actual_parameter_count = sum(
                parameter.numel() for parameter in reloaded.parameters()
            )
            reloaded_state = reloaded.state_dict()
            actual_hashes = {
                key: tensor_sha256(reloaded_state[key]) for key in expected_hashes
            }
            with torch.inference_mode():
                reloaded_logits = reloaded(
                    input_ids=fixed["input_ids"],
                    attention_mask=fixed.get(
                        "attention_mask",
                        torch.ones_like(fixed["input_ids"]),
                    ),
                    use_cache=False,
                ).logits.float()
            logits_error = float(
                (reloaded_logits - distributed_logits).abs().max().item()
            )
            parameter_count_match = actual_parameter_count == parameter_count
            hashes_match = actual_hashes == expected_hashes
            if not parameter_count_match:
                raise RuntimeError("consolidated parameter count does not match DCP")
            if not hashes_match:
                raise RuntimeError("consolidated selected tensor hashes do not match DCP")
            if logits_error >= 1e-5:
                raise RuntimeError(
                    f"consolidated fixed logits differ by {logits_error}, expected < 1e-5"
                )
            manifest = {
                "format_version": 1,
                "checkpoint": str(checkpoint_path),
                "checkpoint_digest": source.checkpoint_digest,
                "source_model": str(source.architecture_path),
                "model_revision": source.model_revision,
                "parameter_count": parameter_count,
                "required_host_bytes": required_host_bytes,
                "optimizer_loaded": False,
                "hf_reload_ok": True,
                "parameter_count_match": parameter_count_match,
                "selected_hashes_match": hashes_match,
                "selected_tensor_hashes": expected_hashes,
                "fixed_logits_max_abs_error": logits_error,
                "success_marker_present": True,
            }
            (temporary / "consolidation_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            (temporary / "_SUCCESS").write_text("verified=true\n", encoding="utf-8")
            os.replace(temporary, destination)
            publication[0] = {"error": None, "manifest": manifest}
        except Exception as error:
            if temporary.is_dir():
                shutil.rmtree(temporary)
            publication[0] = {"error": f"{type(error).__name__}: {error}"}
    if ctx.world_size > 1:
        dist.broadcast_object_list(publication, src=0)
    if publication[0] is None:
        raise RuntimeError("rank zero did not publish consolidation status")
    if publication[0].get("error"):
        raise RuntimeError(str(publication[0]["error"]))
    if ctx.world_size > 1:
        dist.barrier()
    return dict(publication[0]["manifest"])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_model")
    parser.add_argument("--revision")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args(argv)
    ctx = init_distributed(timeout_seconds=args.timeout_seconds)
    try:
        manifest = consolidate_checkpoint(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            ctx=ctx,
            source_model=args.source_model,
            revision=args.revision,
            max_shard_size=args.max_shard_size,
        )
        if ctx.rank == 0:
            print(json.dumps(manifest, sort_keys=True), flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
