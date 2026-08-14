"""Selected-tensor helpers for correctness-gate update comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import torch


def selected_qwen_tensor_names(num_hidden_layers: int) -> tuple[str, ...]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    return (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        f"model.layers.{num_hidden_layers - 1}.self_attn.o_proj.weight",
    )


def update_cosine(
    initial: Mapping[str, torch.Tensor],
    oracle: Mapping[str, torch.Tensor],
    fsdp2: Mapping[str, torch.Tensor],
) -> float:
    names = tuple(initial)
    if not names or set(oracle) != set(names) or set(fsdp2) != set(names):
        raise ValueError("selected tensor mappings must have identical non-empty keys")
    dot = oracle_norm = fsdp_norm = 0.0
    for name in names:
        initial_tensor = initial[name].detach().float()
        oracle_delta = oracle[name].detach().float() - initial_tensor
        fsdp_delta = fsdp2[name].detach().float() - initial_tensor
        dot += float(torch.sum(oracle_delta * fsdp_delta, dtype=torch.float64).item())
        oracle_norm += float(torch.sum(oracle_delta.square(), dtype=torch.float64).item())
        fsdp_norm += float(torch.sum(fsdp_delta.square(), dtype=torch.float64).item())
    if oracle_norm == 0 or fsdp_norm == 0:
        raise ValueError("selected parameter updates must be nonzero")
    return dot / math.sqrt(oracle_norm * fsdp_norm)


def load_hf_selected_tensors(
    model_directory: str | Path,
    names: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Read selected tensors directly from one HF safetensor directory."""

    directory = Path(model_directory)
    requested = tuple(names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("selected tensor names must be unique and non-empty")
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = dict(payload.get("weight_map", {}))
        missing = set(requested) - set(weight_map)
        if missing:
            raise KeyError(f"selected tensors are absent from the HF index: {sorted(missing)}")
        files = {name: directory / str(weight_map[name]) for name in requested}
        candidate_paths = sorted(set(files.values()))
    else:
        candidate_paths = sorted(directory.glob("*.safetensors"))
        if not candidate_paths:
            raise FileNotFoundError(f"no safetensor weights found in {directory}")
        files = {}

    from safetensors import safe_open

    selected: dict[str, torch.Tensor] = {}
    for path in candidate_paths:
        needed = (
            [name for name, mapped_path in files.items() if mapped_path == path]
            if files
            else [name for name in requested if name not in selected]
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in needed:
                if name in available:
                    selected[name] = handle.get_tensor(name)
    missing = set(requested) - set(selected)
    if missing:
        raise KeyError(f"selected tensors are absent from HF weights: {sorted(missing)}")
    return {name: selected[name] for name in requested}


def probe_dcp_selected_tensors(
    *,
    source_model: str,
    revision: str | None,
    checkpoint: str | Path,
    output: str | Path,
    timeout_seconds: int,
) -> None:
    """Load one DCP through FSDP2 and publish selected full CPU tensors."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )
    from train.checkpointing import (
        load_dcp_model_only,
        resolve_hf_checkpoint_source,
    )
    from train.fsdp_utils import (
        FSDPSettings,
        destroy_distributed,
        fully_shard_qwen,
        init_distributed,
    )

    ctx = init_distributed(timeout_seconds=timeout_seconds)
    try:
        if ctx.world_size != 1:
            raise ValueError("correctness state probes require world size one")
        from transformers import AutoConfig, AutoModelForCausalLM

        source = resolve_hf_checkpoint_source(source_model, revision=revision)
        model_config = AutoConfig.from_pretrained(source.path)
        model_config.use_cache = False
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        fully_shard_qwen(model, ctx, FSDPSettings())
        model.to_empty(device=ctx.device)
        load_dcp_model_only(checkpoint, model=model)
        full_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if ctx.rank == 0:
            names = selected_qwen_tensor_names(int(model_config.num_hidden_layers))
            missing = set(names) - set(full_state)
            if missing:
                raise KeyError(f"DCP full state is missing selected tensors: {sorted(missing)}")
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"state probe output already exists: {destination}")
            torch.save({name: full_state[name].cpu() for name in names}, destination)
    finally:
        destroy_distributed()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    probe_dcp_selected_tensors(
        source_model=args.source_model,
        revision=args.revision,
        checkpoint=args.checkpoint,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
