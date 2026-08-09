from __future__ import annotations

from train.fsdp_sft import parse_sft_args
from train.fsdp_grpo import parse_grpo_args


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


def test_grpo_defaults_match_committed_single_gpu_recipe() -> None:
    args = parse_grpo_args(["--model", "/sft", "--output_dir", "/run"])

    assert args.num_generations == 8
    assert args.loss_type == "dr_grpo"
    assert args.scale_rewards == "none"
    assert args.beta == 0.0
    assert args.temperature == 1.0
    assert args.gradient_accumulation_steps == 8
    assert args.activation_checkpointing is True
    assert args.accumulation_sync == "reduce_scatter"


def test_grpo_cli_exposes_reference_and_rollout_memory_choices() -> None:
    args = parse_grpo_args(
        [
            "--model",
            "/sft",
            "--output_dir",
            "/run",
            "--beta",
            "0.02",
            "--rollout_mode",
            "keep_unsharded",
            "--no_activation_checkpointing",
        ]
    )

    assert args.beta == 0.02
    assert args.rollout_mode == "keep_unsharded"
    assert args.activation_checkpointing is False
