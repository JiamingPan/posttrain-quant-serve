"""Validate and publish the one-GPU SFT/GRPO parity gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, variance
from typing import Any, Literal, Mapping, Sequence


GateName = Literal["sft", "grpo"]
EXPECTED_SEEDS = (41, 42, 43)
FIRST_LOSS_ATOL = 5e-3
FIXED_ROLLOUT_ATOL = 5e-4
GRAD_NORM_RTOL = 1e-2
UPDATE_COSINE_MIN = 0.999
RESUME_NEXT_LOSS_ATOL = 1e-6
CURVE_NOISE_FLOOR = 1e-3


def _curve_points(
    comparisons: Sequence[Mapping[str, Any]],
    field: str,
) -> list[dict[str, float | int]]:
    oracle_curves = [list(row[field]["oracle"]) for row in comparisons]
    fsdp_curves = [list(row[field]["fsdp2"]) for row in comparisons]
    lengths = {len(curve) for curve in (*oracle_curves, *fsdp_curves)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"{field} must contain equal non-empty curves")
    points: list[dict[str, float | int]] = []
    for step in range(next(iter(lengths))):
        oracle = [float(curve[step]) for curve in oracle_curves]
        fsdp2 = [float(curve[step]) for curve in fsdp_curves]
        mean_abs_diff = abs(fmean(oracle) - fmean(fsdp2))
        noise_bound = (
            2.0
            * (variance(oracle) / len(oracle) + variance(fsdp2) / len(fsdp2))
            ** 0.5
            + CURVE_NOISE_FLOOR
        )
        points.append(
            {
                "step": step + 1,
                "oracle_mean": fmean(oracle),
                "fsdp2_mean": fmean(fsdp2),
                "mean_abs_diff": mean_abs_diff,
                "noise_bound": noise_bound,
            }
        )
    return points


def _consistent_value(
    comparisons: Sequence[Mapping[str, Any]],
    field: str,
) -> str:
    values = {str(row[field]) for row in comparisons}
    if len(values) != 1:
        label = field.replace("_", " ")
        raise ValueError(f"correctness comparisons have mismatched {label}s")
    return values.pop()


def build_gate_record(
    gate: GateName,
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate three seed comparisons and reject before publication on failure."""

    if gate not in {"sft", "grpo"}:
        raise ValueError("gate must be 'sft' or 'grpo'")
    if len(comparisons) != 3:
        raise ValueError("correctness gate requires exactly three seed comparisons")
    seeds = sorted(int(row["seed"]) for row in comparisons)
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError("correctness gate requires exactly three seeds: 41, 42, 43")
    model_digest = _consistent_value(comparisons, "model_digest")
    dataset_digest = _consistent_value(comparisons, "dataset_digest")
    source_records: list[dict[str, Any]] = []
    for row in comparisons:
        for implementation in ("oracle", "fsdp2"):
            source_records.append(
                {
                    "seed": int(row["seed"]),
                    "implementation": implementation,
                    "path": str(row[f"{implementation}_record"]),
                    "sha256": str(row[f"{implementation}_record_sha256"]),
                }
            )

    metrics: dict[str, Any] = {
        "first_loss_abs_error_max": max(
            float(row["first_loss_abs_error"]) for row in comparisons
        ),
        "grad_norm_relative_error_max": max(
            float(row["grad_norm_relative_error_max"]) for row in comparisons
        ),
        "update_cosine_min": min(
            float(row["update_cosine_min"]) for row in comparisons
        ),
        "loss_curve_points": _curve_points(comparisons, "loss_curve"),
    }
    if gate == "grpo":
        metrics.update(
            fixed_rollout_loss_abs_error_max=max(
                float(row["fixed_rollout_loss_abs_error"])
                for row in comparisons
            ),
            resume_next_loss_abs_error_max=max(
                float(row["resume_next_loss_abs_error"])
                for row in comparisons
            ),
            reward_curve_points=_curve_points(comparisons, "reward_curve"),
        )
    record = {
        "format_version": 1,
        "gate": gate,
        "status": "pass",
        "seeds": seeds,
        "model_digest": model_digest,
        "dataset_digest": dataset_digest,
        "thresholds": {
            "first_loss_abs_error_max": FIRST_LOSS_ATOL,
            "fixed_rollout_loss_abs_error_max": FIXED_ROLLOUT_ATOL,
            "grad_norm_relative_error_max": GRAD_NORM_RTOL,
            "update_cosine_min": UPDATE_COSINE_MIN,
            "resume_next_loss_abs_error_max": RESUME_NEXT_LOSS_ATOL,
            "curve_noise_floor": CURVE_NOISE_FLOOR,
        },
        "metrics": metrics,
        "source_records": source_records,
    }
    validate_gate_record(record)
    return record


def _validate_curve(points: Any, *, label: str) -> None:
    if not isinstance(points, list) or not points:
        raise ValueError(f"{label} curve points are missing")
    for point in points:
        if float(point["mean_abs_diff"]) > float(point["noise_bound"]):
            raise ValueError(f"{label} curve exceeds its three-seed noise bound")


def validate_gate_record(record: Mapping[str, Any]) -> None:
    """Apply fixed acceptance thresholds; record-declared thresholds are informational."""

    gate = record.get("gate")
    if gate not in {"sft", "grpo"}:
        raise ValueError("correctness record has an invalid gate")
    seeds = list(record.get("seeds", []))
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError("correctness gate requires exactly three seeds: 41, 42, 43")
    if not record.get("model_digest") or not record.get("dataset_digest"):
        raise ValueError("correctness record is missing source digests")
    sources = record.get("source_records")
    if not isinstance(sources, list) or len(sources) != 6:
        raise ValueError("correctness gate requires six traceable source records")
    for source in sources:
        if not source.get("path") or len(str(source.get("sha256", ""))) != 64:
            raise ValueError("source records require paths and SHA-256 digests")

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("correctness record is missing metrics")
    if float(metrics["first_loss_abs_error_max"]) > FIRST_LOSS_ATOL:
        raise ValueError("first loss exceeds the 5e-3 absolute tolerance")
    if float(metrics["grad_norm_relative_error_max"]) > GRAD_NORM_RTOL:
        raise ValueError("gradient norm exceeds the 1e-2 relative tolerance")
    if float(metrics["update_cosine_min"]) < UPDATE_COSINE_MIN:
        raise ValueError("selected update cosine must be at least 0.999")
    _validate_curve(metrics["loss_curve_points"], label="loss")
    if gate == "grpo":
        if float(metrics["fixed_rollout_loss_abs_error_max"]) > FIXED_ROLLOUT_ATOL:
            raise ValueError("fixed-rollout loss exceeds the 5e-4 tolerance")
        if float(metrics["resume_next_loss_abs_error_max"]) > RESUME_NEXT_LOSS_ATOL:
            raise ValueError("resume next-step loss exceeds the 1e-6 tolerance")
        _validate_curve(metrics["reward_curve_points"], label="reward")
    if record.get("status") != "pass":
        raise ValueError("a published correctness gate must have pass status")


def _run_gate(
    gate: GateName,
    comparisons_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    comparisons_file = Path(comparisons_path)
    try:
        comparisons = json.loads(comparisons_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid comparison input: {comparisons_file}") from error
    if not isinstance(comparisons, list):
        raise ValueError("comparison input must be a JSON list")
    record = build_gate_record(gate, comparisons)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"gate record already exists: {destination}")
    destination.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def run_sft_gate(
    comparisons_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return _run_gate("sft", comparisons_path, output_path)


def run_grpo_gate(
    comparisons_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return _run_gate("grpo", comparisons_path, output_path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("sft", "grpo"), required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runner = run_sft_gate if args.gate == "sft" else run_grpo_gate
    record = runner(args.comparisons, args.output)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
