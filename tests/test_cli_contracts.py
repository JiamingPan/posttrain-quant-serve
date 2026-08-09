from __future__ import annotations

from train.fsdp_sft import parse_sft_args


def test_sft_cli_defaults_enable_checkpointing() -> None:
    args = parse_sft_args(["--output_dir", "/tmp/run"])

    assert args.model == "Qwen/Qwen3-8B"
    assert args.sharding == "fsdp2"
    assert args.activation_checkpointing is True
    assert args.gradient_accumulation_steps == 8
    assert args.max_grad_norm == 1.0
    assert args.accumulation_sync == "reduce_scatter"
    assert args.resident_precision == "bf16"


def test_sft_cli_supports_the_explicit_checkpointing_ablation() -> None:
    args = parse_sft_args(
        [
            "--output_dir",
            "/tmp/run",
            "--no_activation_checkpointing",
            "--sharding",
            "none",
        ]
    )

    assert args.activation_checkpointing is False
    assert args.sharding == "none"
