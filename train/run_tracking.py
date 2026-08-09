"""Small, dependency-light helpers for reproducible training run records."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RunIdentity:
    """Identifiers captured once and copied into every record for a run."""

    run_id: str
    stage: str
    git_commit: str
    git_dirty: bool
    slurm_job_id: str | None
    hostname: str
    started_at_utc: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def capture_run_identity(stage: str, config: Mapping[str, Any]) -> RunIdentity:
    """Create a deterministic config suffix plus runtime/git provenance."""

    config_hash = hashlib.sha256(_canonical_json(config).encode()).hexdigest()[:10]
    now = datetime.now(UTC)
    started_at = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    commit = _git_output("rev-parse", "HEAD")
    dirty = _git_output("status", "--porcelain") not in {"", "unknown"}
    return RunIdentity(
        run_id=f"{stage}-{timestamp}-{config_hash}",
        stage=stage,
        git_commit=commit,
        git_dirty=dirty,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=socket.gethostname(),
        started_at_utc=started_at,
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    return value


def append_run_record(
    output_dir: str | Path,
    stem: str,
    record: Mapping[str, Any],
    csv_fields: Sequence[str],
) -> tuple[Path, Path]:
    """Append one record to canonical JSONL and a fixed-schema CSV file."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fields = tuple(csv_fields)
    if set(record) != set(fields):
        raise ValueError("record fields must exactly match the declared CSV schema")

    jsonl_path = output_path / f"{stem}.jsonl"
    csv_path = output_path / f"{stem}.csv"
    file_exists = csv_path.exists()
    if file_exists:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing_fields = tuple(next(csv.reader(handle), ()))
        if existing_fields != fields:
            raise ValueError(
                f"CSV schema mismatch: existing={existing_fields!r}, requested={fields!r}"
            )

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(record)) + "\n")

    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: _csv_value(record[field]) for field in fields})
    return jsonl_path, csv_path


def write_run_config(output_dir: str | Path, config: Mapping[str, Any]) -> Path:
    """Write the resolved config once and refuse ambiguous run-directory reuse."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "config.json"
    rendered = json.dumps(config, sort_keys=True, indent=2) + "\n"
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != rendered:
            raise ValueError("run directory already contains a different resolved configuration")
        return config_path
    config_path.write_text(rendered, encoding="utf-8")
    return config_path


def gather_rank_records(
    local_record: Mapping[str, Any], dst: int = 0
) -> list[dict[str, Any]] | None:
    """Gather per-rank measurements without requiring distributed initialization."""

    try:
        import torch.distributed as dist
    except ImportError:
        return [dict(local_record)]

    if not dist.is_available() or not dist.is_initialized():
        return [dict(local_record)]

    rank = dist.get_rank()
    gathered: list[dict[str, Any] | None] | None
    if rank == dst:
        gathered = [None] * dist.get_world_size()
    else:
        gathered = None
    dist.gather_object(dict(local_record), gathered, dst=dst)
    if gathered is None:
        return None
    return [record for record in gathered if record is not None]
