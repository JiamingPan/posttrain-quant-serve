import csv
import json

import pytest

from train.run_tracking import (
    append_run_record,
    capture_run_identity,
    gather_rank_records,
    write_run_config,
)


def test_append_record_writes_jsonl_and_stable_csv_columns(tmp_path) -> None:
    row = {
        "run_id": "sft-w1-seed42",
        "world_size": 1,
        "tokens_per_sec": 10.0,
    }

    jsonl_path, csv_path = append_run_record(tmp_path, "scaling", row, tuple(row))

    assert json.loads(jsonl_path.read_text().strip()) == row
    with csv_path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [
            {"run_id": "sft-w1-seed42", "world_size": "1", "tokens_per_sec": "10.0"}
        ]


def test_append_record_rejects_schema_drift(tmp_path) -> None:
    jsonl_path, _ = append_run_record(
        tmp_path,
        "scaling",
        {"run_id": "one", "world_size": 1},
        ("run_id", "world_size"),
    )

    with pytest.raises(ValueError, match="CSV schema"):
        append_run_record(
            tmp_path,
            "scaling",
            {"run_id": "two", "tokens_per_sec": 10.0},
            ("run_id", "tokens_per_sec"),
        )

    assert len(jsonl_path.read_text().splitlines()) == 1


def test_nested_csv_values_are_canonical_json(tmp_path) -> None:
    _, csv_path = append_run_record(
        tmp_path,
        "ranks",
        {"run_id": "one", "peaks": {"0": 12.0, "1": 13.0}},
        ("run_id", "peaks"),
    )

    with csv_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["peaks"] == '{"0":12.0,"1":13.0}'


def test_write_run_config_is_sorted_and_refuses_overwrite_with_different_content(tmp_path) -> None:
    path = write_run_config(tmp_path, {"world_size": 2, "model": "Qwen/Qwen3-8B"})

    assert path.read_text() == '{\n  "model": "Qwen/Qwen3-8B",\n  "world_size": 2\n}\n'
    write_run_config(tmp_path, {"model": "Qwen/Qwen3-8B", "world_size": 2})
    with pytest.raises(ValueError, match="different resolved configuration"):
        write_run_config(tmp_path, {"model": "Qwen/Qwen3-8B", "world_size": 4})


def test_identity_uses_config_hash_and_slurm_job_id(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    first = capture_run_identity("sft", {"model": "tiny", "seed": 7})
    second = capture_run_identity("sft", {"seed": 7, "model": "tiny"})

    assert first.run_id.rsplit("-", 1)[-1] == second.run_id.rsplit("-", 1)[-1]
    assert first.slurm_job_id == "12345"
    assert first.stage == "sft"
    assert first.git_commit


def test_gather_rank_records_returns_local_record_without_distributed_process_group() -> None:
    row = {"rank": 0, "allocated_gib": 1.5}

    assert gather_rank_records(row) == [row]
