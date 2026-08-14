from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.fixture
def torchrun_result(tmp_path):
    """Launch a real torchrun worker and return its rank-zero JSON payload."""

    def run(worker: str, nproc: int, **worker_args: object) -> dict[str, object]:
        if not torch.cuda.is_available() or torch.cuda.device_count() < nproc:
            pytest.skip(f"requires {nproc} CUDA GPUs")
        result_path = tmp_path / f"{Path(worker).stem}-{nproc}.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            worker,
            "--result-path",
            str(result_path),
        ]
        for key, value in worker_args.items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            pytest.fail(
                f"torchrun failed with exit code {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        if not result_path.exists():
            pytest.fail("torchrun worker did not publish its rank-zero result")
        return json.loads(result_path.read_text(encoding="utf-8"))

    return run
