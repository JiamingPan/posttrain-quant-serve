"""Weight outlier and W4 proxy diagnostics for causal LM checkpoints.

This script is intentionally offline: it does not serve, quantize in place, or
write model weights. It measures whether two checkpoints look different from a
quantization perspective by summarizing per-layer weight distributions and a
simple symmetric group-wise int4 reconstruction error.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, model = spec.split("=", 1)
    else:
        model = spec
        label = Path(spec).name if Path(spec).name else spec.replace("/", "_")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    if not label:
        raise ValueError(f"Could not infer a label from model spec: {spec!r}")
    return label, model


def dtype_from_arg(value: str) -> torch.dtype | str:
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}; use one of {sorted(mapping)}")
    return mapping[value]


def tensor_kurtosis(x: torch.Tensor) -> float:
    flat = x.float().reshape(-1)
    if flat.numel() < 2:
        return float("nan")
    mean = flat.mean()
    centered = flat - mean
    var = centered.pow(2).mean()
    if var <= 0:
        return float("nan")
    return float(centered.pow(4).mean().div(var.pow(2)).item())


def quant_error_w4_symmetric(weight: torch.Tensor, group_size: int) -> dict[str, float]:
    """Approximate symmetric W4 group quantization error along the input axis."""
    w = weight.detach().float().cpu()
    if w.ndim != 2 or group_size <= 0:
        return {
            "w4_group_size": float(group_size),
            "w4_mse": float("nan"),
            "w4_relative_mse": float("nan"),
            "w4_snr_db": float("nan"),
            "w4_max_abs_error": float("nan"),
        }

    rows, cols = w.shape
    pad = (group_size - cols % group_size) % group_size
    if pad:
        w_grouped = torch.nn.functional.pad(w, (0, pad))
    else:
        w_grouped = w
    w_grouped = w_grouped.reshape(rows, -1, group_size)

    max_abs = w_grouped.abs().amax(dim=-1, keepdim=True)
    scale = max_abs / 7.0
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(w_grouped / safe_scale), -7, 7)
    deq = q * scale
    if pad:
        deq = deq.reshape(rows, -1)[:, :cols]
    else:
        deq = deq.reshape(rows, cols)

    err = deq - w
    mse = err.pow(2).mean()
    signal = w.pow(2).mean()
    rel = mse / signal.clamp_min(1e-30)
    snr = 10.0 * torch.log10(signal.clamp_min(1e-30) / mse.clamp_min(1e-30))
    return {
        "w4_group_size": float(group_size),
        "w4_mse": float(mse.item()),
        "w4_relative_mse": float(rel.item()),
        "w4_snr_db": float(snr.item()),
        "w4_max_abs_error": float(err.abs().max().item()),
    }


def summarize_weight(
    *,
    model_label: str,
    module_name: str,
    module: nn.Module,
    weight: torch.Tensor,
    group_size: int,
    outlier_multiplier: float,
    compute_quant_error: bool,
) -> dict[str, Any]:
    w = weight.detach().float().cpu()
    abs_w = w.abs()
    flat_abs = abs_w.reshape(-1)
    median_abs = flat_abs.median()
    std = w.std(unbiased=False)
    rms = w.pow(2).mean().sqrt()

    if w.ndim == 2:
        channel_max = abs_w.amax(dim=1)
        channel_median = channel_max.median()
        channel_threshold = channel_median * outlier_multiplier
        channel_outlier_frac = (channel_max > channel_threshold).float().mean()
        channel_stats = {
            "out_channels": int(w.shape[0]),
            "in_channels": int(w.shape[1]),
            "channel_max_abs_mean": float(channel_max.mean().item()),
            "channel_max_abs_median": float(channel_median.item()),
            "channel_max_abs_p95": float(torch.quantile(channel_max, 0.95).item()),
            "channel_max_abs_p99": float(torch.quantile(channel_max, 0.99).item()),
            "channel_max_abs_max": float(channel_max.max().item()),
            "channel_outlier_threshold": float(channel_threshold.item()),
            "channel_outlier_frac": float(channel_outlier_frac.item()),
        }
    else:
        channel_stats = {
            "out_channels": None,
            "in_channels": None,
            "channel_max_abs_mean": float("nan"),
            "channel_max_abs_median": float("nan"),
            "channel_max_abs_p95": float("nan"),
            "channel_max_abs_p99": float("nan"),
            "channel_max_abs_max": float("nan"),
            "channel_outlier_threshold": float("nan"),
            "channel_outlier_frac": float("nan"),
        }

    abs_threshold = median_abs * outlier_multiplier
    row: dict[str, Any] = {
        "model": model_label,
        "module_name": module_name,
        "module_type": type(module).__name__,
        "shape": "x".join(str(dim) for dim in w.shape),
        "ndim": int(w.ndim),
        "numel": int(w.numel()),
        "dtype": str(weight.dtype).replace("torch.", ""),
        "max_abs": float(flat_abs.max().item()),
        "mean_abs": float(flat_abs.mean().item()),
        "median_abs": float(median_abs.item()),
        "std": float(std.item()),
        "rms": float(rms.item()),
        "kurtosis": tensor_kurtosis(w),
        "abs_outlier_threshold": float(abs_threshold.item()),
        "abs_outlier_frac": float((flat_abs > abs_threshold).float().mean().item()),
        **channel_stats,
    }
    if compute_quant_error and w.ndim == 2:
        row.update(quant_error_w4_symmetric(w, group_size=group_size))
    return row


def iter_weight_modules(model: nn.Module, include_embeddings: bool) -> list[tuple[str, nn.Module, torch.Tensor]]:
    modules: list[tuple[str, nn.Module, torch.Tensor]] = []
    seen_ptrs: set[int] = set()
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.ndim != 2:
            continue
        if isinstance(module, nn.Embedding) and not include_embeddings:
            continue
        ptr = weight.untyped_storage().data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        modules.append((name or "<root>", module, weight))
    return modules


def summarize_model(
    *,
    label: str,
    model_name_or_path: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"\n=== Loading {label}: {model_name_or_path} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype_from_arg(args.dtype),
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()

    rows = []
    modules = iter_weight_modules(model, include_embeddings=args.include_embeddings)
    if args.max_tensors is not None:
        modules = modules[: args.max_tensors]
    print(f"Found {len(modules)} 2D weight tensors for {label}", flush=True)

    with torch.no_grad():
        for index, (name, module, weight) in enumerate(modules, start=1):
            if index % 20 == 0 or index == len(modules):
                print(f"[{label}] {index}/{len(modules)} {name}", flush=True)
            rows.append(
                summarize_weight(
                    model_label=label,
                    module_name=name,
                    module=module,
                    weight=weight,
                    group_size=args.group_size,
                    outlier_multiplier=args.outlier_multiplier,
                    compute_quant_error=not args.no_quant_error,
                )
            )

    df = pd.DataFrame(rows)
    summary = summarize_layer_table(label, model_name_or_path, df)

    del model
    gc.collect()
    return df, summary


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str = "numel") -> float | None:
    if df.empty or value_col not in df:
        return None
    values = pd.to_numeric(df[value_col], errors="coerce")
    weights = pd.to_numeric(df[weight_col], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return None
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, int)) and not math.isfinite(float(value)):
        return None
    return value


def records_for_json(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    records = df[columns].to_dict(orient="records")
    return [{key: none_if_nan(value) for key, value in row.items()} for row in records]


def summarize_layer_table(label: str, model_name_or_path: str, df: pd.DataFrame) -> dict[str, Any]:
    snr_mean = None
    if "w4_snr_db" in df:
        snr_value = float(pd.to_numeric(df["w4_snr_db"], errors="coerce").mean())
        snr_mean = none_if_nan(snr_value)

    summary: dict[str, Any] = {
        "model": label,
        "model_name_or_path": model_name_or_path,
        "num_weight_tensors": int(len(df)),
        "total_numel": int(df["numel"].sum()) if "numel" in df else 0,
        "max_abs_max": float(df["max_abs"].max()) if "max_abs" in df and len(df) else None,
        "channel_outlier_frac_weighted": weighted_mean(df, "channel_outlier_frac"),
        "abs_outlier_frac_weighted": weighted_mean(df, "abs_outlier_frac"),
        "w4_relative_mse_weighted": weighted_mean(df, "w4_relative_mse"),
        "w4_snr_db_mean": snr_mean,
    }
    if "w4_relative_mse" in df:
        top = df.sort_values("w4_relative_mse", ascending=False).head(10)
        columns = ["module_name", "module_type", "shape", "w4_relative_mse", "channel_outlier_frac"]
        summary["top_w4_relative_mse_modules"] = records_for_json(top, columns)
    if "channel_outlier_frac" in df:
        top = df.sort_values("channel_outlier_frac", ascending=False).head(10)
        columns = ["module_name", "module_type", "shape", "channel_outlier_frac", "channel_max_abs_max"]
        summary["top_channel_outlier_modules"] = records_for_json(top, columns)
    return summary


def write_outputs(output_dir: Path, layer_tables: list[pd.DataFrame], summaries: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(layer_tables, ignore_index=True) if layer_tables else pd.DataFrame()
    combined_path = output_dir / "weight_outlier_layers.csv"
    summary_path = output_dir / "weight_outlier_summary.json"
    combined.to_csv(combined_path, index=False)
    summary_path.write_text(json.dumps({"models": summaries}, indent=2, allow_nan=False))
    print(f"\nWrote {combined_path}")
    print(f"Wrote {summary_path}")

    if not combined.empty and "model" in combined:
        for label, frame in combined.groupby("model"):
            model_path = output_dir / f"{label}_layers.csv"
            frame.to_csv(model_path, index=False)
            print(f"Wrote {model_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model id/path, optionally label=model_id. Repeat for comparisons.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"])
    parser.add_argument("--group_size", type=int, default=128, help="Group size for the W4 proxy error.")
    parser.add_argument("--outlier_multiplier", type=float, default=6.0)
    parser.add_argument("--include_embeddings", action="store_true")
    parser.add_argument("--no_quant_error", action="store_true", help="Skip the W4 proxy reconstruction error.")
    parser.add_argument("--max_tensors", type=int, default=None, help="Debug option to process only the first N tensors.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    layer_tables = []
    summaries = []
    for spec in args.model:
        label, model_name_or_path = parse_model_spec(spec)
        df, summary = summarize_model(label=label, model_name_or_path=model_name_or_path, args=args)
        layer_tables.append(df)
        summaries.append(summary)

    write_outputs(args.output_dir, layer_tables, summaries)


if __name__ == "__main__":
    main()
