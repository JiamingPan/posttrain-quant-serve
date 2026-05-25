"""Print a minimal CUDA/PyTorch environment check."""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this on a GPU node.")

    print(f"device count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        print(f"gpu {idx}: {props.name}, {total_gb:.1f} GiB")

    x = torch.ones((2, 2), device="cuda")
    print(f"cuda tensor sum: {x.sum().item():.1f}")


if __name__ == "__main__":
    main()
