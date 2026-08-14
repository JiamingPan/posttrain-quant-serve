from __future__ import annotations

import json

import pytest

from bench.correctness import (
    build_gate_record,
    run_grpo_gate,
    run_sft_gate,
    validate_gate_record,
)


def _comparison(seed: int, *, gate: str = "sft") -> dict[str, object]:
    comparison: dict[str, object] = {
        "seed": seed,
        "model_digest": "model-immutable",
        "dataset_digest": "gsm8k-first-16",
        "oracle_record": f"oracle-{seed}.json",
        "fsdp2_record": f"fsdp2-{seed}.json",
        "oracle_record_sha256": f"{seed:064x}",
        "fsdp2_record_sha256": f"{seed + 100:064x}",
        "first_loss_abs_error": 0.001,
        "grad_norm_relative_error_max": 0.005,
        "update_cosine_min": 0.9995,
        "loss_curve": {
            "oracle": [2.0 - seed / 1000, 1.9 - seed / 1000],
            "fsdp2": [2.0005 - seed / 1000, 1.9005 - seed / 1000],
        },
    }
    if gate == "grpo":
        comparison.update(
            fixed_rollout_loss_abs_error=0.0002,
            resume_next_loss_abs_error=0.0000005,
            reward_curve={
                "oracle": [0.25 + seed / 1000, 0.5 + seed / 1000],
                "fsdp2": [0.2505 + seed / 1000, 0.5005 + seed / 1000],
            },
        )
    return comparison


def _passing_record(gate: str = "sft") -> dict[str, object]:
    return build_gate_record(
        gate,
        [_comparison(seed, gate=gate) for seed in (41, 42, 43)],
    )


def test_gate_rejects_missing_seed() -> None:
    record = _passing_record()
    record["seeds"] = [41, 42]
    with pytest.raises(ValueError, match="exactly three seeds"):
        validate_gate_record(record)


def test_sft_gate_records_bf16_trajectory_diagnostics_without_rejecting() -> None:
    record = _passing_record()
    record["metrics"]["grad_norm_relative_error_max"] = 0.25
    record["metrics"]["update_cosine_min"] = 0.95

    validate_gate_record(record)

    assert record["metrics"]["grad_norm_relative_error_max"] == 0.25
    assert record["metrics"]["update_cosine_min"] == 0.95
    assert record["diagnostic_only_metrics"] == [
        "grad_norm_relative_error_max",
        "update_cosine_min",
    ]


@pytest.mark.parametrize(
    "metric",
    ["grad_norm_relative_error_max", "update_cosine_min"],
)
def test_sft_gate_rejects_nonfinite_diagnostics(metric: str) -> None:
    record = _passing_record()
    record["metrics"][metric] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_gate_record(record)


def test_gate_rejects_nonfinite_loss_curve() -> None:
    record = _passing_record()
    record["metrics"]["loss_curve_points"][0]["mean_abs_diff"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        validate_gate_record(record)


def test_grpo_keeps_strict_gradient_and_update_checks() -> None:
    record = _passing_record("grpo")
    record["metrics"]["grad_norm_relative_error_max"] = 0.02
    with pytest.raises(ValueError, match="gradient norm"):
        validate_gate_record(record)

    record = _passing_record("grpo")
    record["metrics"]["update_cosine_min"] = 0.998
    with pytest.raises(ValueError, match="0.999"):
        validate_gate_record(record)


def test_gate_rejects_mismatched_model_or_dataset_digests() -> None:
    comparisons = [_comparison(seed) for seed in (41, 42, 43)]
    comparisons[1]["model_digest"] = "different-model"
    with pytest.raises(ValueError, match="model digest"):
        build_gate_record("sft", comparisons)

    comparisons = [_comparison(seed) for seed in (41, 42, 43)]
    comparisons[2]["dataset_digest"] = "different-data"
    with pytest.raises(ValueError, match="dataset digest"):
        build_gate_record("sft", comparisons)


def test_gate_rejects_curve_mean_outside_the_three_seed_noise_bound() -> None:
    record = _passing_record()
    record["metrics"]["loss_curve_points"][0]["mean_abs_diff"] = 0.5

    with pytest.raises(ValueError, match="loss curve.*noise bound"):
        validate_gate_record(record)


def test_grpo_gate_requires_fixed_rollout_and_exact_resume() -> None:
    record = _passing_record("grpo")
    record["metrics"]["fixed_rollout_loss_abs_error_max"] = 0.0006
    with pytest.raises(ValueError, match="fixed-rollout.*5e-4"):
        validate_gate_record(record)

    record = _passing_record("grpo")
    record["metrics"]["resume_next_loss_abs_error_max"] = 0.000002
    with pytest.raises(ValueError, match="resume.*1e-6"):
        validate_gate_record(record)


@pytest.mark.parametrize("gate", ["sft", "grpo"])
def test_passing_gate_contains_traceable_source_records(gate: str) -> None:
    record = _passing_record(gate)

    validate_gate_record(record)

    assert record["status"] == "pass"
    assert record["seeds"] == [41, 42, 43]
    assert len(record["source_records"]) == 6
    assert all("path" in source for source in record["source_records"])
    assert all(len(source["sha256"]) == 64 for source in record["source_records"])


def test_gate_retains_additional_resume_and_state_probe_evidence() -> None:
    comparisons = [_comparison(seed, gate="grpo") for seed in (41, 42, 43)]
    comparisons[0]["evidence_records"] = [
        {
            "implementation": "fsdp2_resume",
            "path": "resume-41.jsonl",
            "sha256": "a" * 64,
        }
    ]

    record = build_gate_record("grpo", comparisons)

    assert len(record["source_records"]) == 7
    assert {
        "seed": 41,
        "implementation": "fsdp2_resume",
        "path": "resume-41.jsonl",
        "sha256": "a" * 64,
    } in record["source_records"]


def test_gate_writes_only_after_all_comparisons_pass(tmp_path) -> None:
    comparisons_path = tmp_path / "comparisons.json"
    comparisons_path.write_text(
        json.dumps([_comparison(seed) for seed in (41, 42, 43)]),
        encoding="utf-8",
    )
    output_path = tmp_path / "sft_gate.json"

    record = run_sft_gate(comparisons_path, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == record

    failed = [_comparison(seed, gate="grpo") for seed in (41, 42, 43)]
    failed[0]["resume_next_loss_abs_error"] = 0.01
    comparisons_path.write_text(json.dumps(failed), encoding="utf-8")
    failed_output = tmp_path / "grpo_gate.json"
    with pytest.raises(ValueError, match="resume"):
        run_grpo_gate(comparisons_path, failed_output)
    assert not failed_output.exists()
