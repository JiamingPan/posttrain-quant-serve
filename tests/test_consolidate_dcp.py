from __future__ import annotations

import pytest
import torch

from scripts.consolidate_dcp import (
    assert_host_memory_for_consolidation,
    required_host_memory_bytes,
    tensor_sha256,
)


def test_host_memory_preflight_includes_twenty_five_percent_workspace() -> None:
    assert required_host_memory_bytes(8_000_000_000, bytes_per_parameter=2) == 20_000_000_000

    with pytest.raises(MemoryError, match="18.63 GiB required.*16.00 GiB available"):
        assert_host_memory_for_consolidation(
            parameter_count=8_000_000_000,
            bytes_per_parameter=2,
            available_bytes=16 * 1024**3,
        )


def test_tensor_hash_covers_bfloat16_storage_bytes() -> None:
    first = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    second = torch.tensor([1.0, 3.0], dtype=torch.bfloat16)

    assert len(tensor_sha256(first)) == 64
    assert tensor_sha256(first) != tensor_sha256(second)


@pytest.mark.cuda
@pytest.mark.distributed
def test_consolidated_directory_loads_with_transformers(torchrun_result, tmp_path) -> None:
    row = torchrun_result(
        "tests/workers/consolidate_worker.py",
        nproc=2,
        output_dir=tmp_path,
    )

    assert row["optimizer_loaded"] is False
    assert row["hf_reload_ok"] is True
    assert row["parameter_count_match"] is True
    assert row["selected_hashes_match"] is True
    assert row["fixed_logits_max_abs_error"] < 1e-5
    assert row["success_marker_present"] is True
