from __future__ import annotations

from train.fsdp_sft import parse_sft_args
from train.fsdp_grpo import parse_grpo_args
from scripts.train_grpo_gsm8k import parse_args as parse_oracle_grpo_args
from bench.scaling import parse_scaling_args


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


def test_existing_grpo_oracle_adds_deterministic_controls_without_changing_defaults() -> None:
    defaults = parse_oracle_grpo_args(["--output_dir", "/run"])
    controlled = parse_oracle_grpo_args(
        [
            "--output_dir",
            "/run",
            "--seed",
            "43",
            "--run_record",
            "/records/oracle.json",
        ]
    )

    assert defaults.seed == 42
    assert defaults.run_record is None
    assert controlled.seed == 43
    assert controlled.run_record == "/records/oracle.json"


def test_scaling_worker_defaults_match_the_fixed_global_batch_recipe() -> None:
    args = parse_scaling_args(["--worker", "--output_dir", "/tmp/scaling"])

    assert args.worker is True
    assert args.world_sizes == (1, 2, 4, 8)
    assert args.model == "Qwen/Qwen3-8B"
    assert args.global_batch_size == 8
    assert args.sequence_length == 2048
    assert args.micro_batch_size == 1
    assert args.gradient_accumulation_steps is None
    assert args.warmup_steps == 3
    assert args.measure_steps == 10
    assert args.profile_steps == 3
    assert args.activation_checkpointing is True


def test_scaling_pilot_is_explicit_and_keeps_measurement_defaults() -> None:
    full = parse_scaling_args(["--output_dir", "/tmp/full"])
    pilot = parse_scaling_args(
        [
            "--pilot",
            "--world_sizes",
            "1,2",
            "--output_dir",
            "/tmp/pilot",
        ]
    )

    assert full.pilot is False
    assert full.benchmark_mode == "full"
    assert pilot.pilot is True
    assert pilot.benchmark_mode == "pilot"
    assert pilot.world_sizes == (1, 2)
    assert (pilot.warmup_steps, pilot.measure_steps, pilot.profile_steps) == (3, 10, 3)
