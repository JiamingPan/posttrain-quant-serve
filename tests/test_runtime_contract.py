from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def _requirement_named(name: str) -> Requirement:
    for raw_line in Path("requirements.txt").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.name == name:
            return requirement
    raise AssertionError(f"Missing requirement: {name}")


def test_transformers_requirement_accepts_qwen3_floor_and_rejects_older_release() -> None:
    transformers = _requirement_named("transformers")

    assert Version("4.51.0") in transformers.specifier
    assert Version("4.50.9") not in transformers.specifier


def test_cluster_runtime_documents_the_pinned_cuda_build() -> None:
    requirements = Path("requirements.txt").read_text()

    assert "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0" in requirements
