from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bench.correctness import (
    build_sft_seed_comparison,
    build_grpo_commands,
    build_sft_commands,
    build_state_probe_command,
    extract_oracle_grpo_curves,
    fixed_rollout_loss_abs_error,
    has_dirty_paths_outside,
    parse_correctness_args,
    relative_error_max,
)
from bench.correctness_state import (
    load_hf_selected_tensors,
    selected_qwen_tensor_names,
    update_cosine,
)


def test_documented_correctness_cli_selects_execution_mode() -> None:
    config = parse_correctness_args(
        [
            "--gate",
            "sft",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--seeds",
            "41,42,43",
            "--dataset_limit",
            "16",
            "--max_steps",
            "20",
            "--output_dir",
            "/results/correctness",
        ]
    )

    assert config.execute is True
    assert config.gate == "sft"
    assert config.seeds == (41, 42, 43)
    assert config.output_dir == "/results/correctness"
    assert config.max_steps == 20


def test_sft_commands_differ_only_in_sharding_and_output_track(tmp_path) -> None:
    config = parse_correctness_args(
        ["--gate", "sft", "--output_dir", str(tmp_path)]
    )

    commands = build_sft_commands(config, seed=41)

    assert "--module" in commands["oracle"]
    assert "train.fsdp_sft" in commands["oracle"]
    assert commands["oracle"][commands["oracle"].index("--sharding") + 1] == "none"
    assert commands["fsdp2"][commands["fsdp2"].index("--sharding") + 1] == "fsdp2"
    for command in commands.values():
        assert command[command.index("--seed") + 1] == "41"
        assert command[command.index("--save_steps") + 1] == "20"
        assert command[command.index("--max_steps") + 1] == "20"


def test_grpo_commands_include_oracle_primary_and_exact_resume_run(tmp_path) -> None:
    config = parse_correctness_args(
        [
            "--gate",
            "grpo",
            "--output_dir",
            str(tmp_path),
            "--num_generations",
            "8",
            "--max_steps",
            "20",
        ]
    )

    commands = build_grpo_commands(config, seed=43)

    assert set(commands) == {"oracle", "fsdp2", "resume"}
    assert "scripts.train_grpo_gsm8k" in commands["oracle"]
    assert commands["oracle"][commands["oracle"].index("--beta") + 1] == "0.0"
    assert commands["fsdp2"][commands["fsdp2"].index("--max_steps") + 1] == "21"
    assert commands["fsdp2"][commands["fsdp2"].index("--save_steps") + 1] == "20"
    resume_checkpoint = commands["resume"][commands["resume"].index("--resume") + 1]
    assert resume_checkpoint.endswith("checkpoints/step-00000020")


def test_relative_error_uses_a_stable_nonzero_denominator() -> None:
    assert relative_error_max([2.0, 0.0], [2.01, 1e-14]) == pytest.approx(0.005)
    with pytest.raises(ValueError, match="equal non-empty"):
        relative_error_max([1.0], [1.0, 2.0])


def test_selected_state_names_cover_first_and_last_qwen_blocks() -> None:
    assert selected_qwen_tensor_names(24) == (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.23.self_attn.o_proj.weight",
    )


def test_update_cosine_compares_parameter_deltas_not_final_weights() -> None:
    initial = {"weight": torch.tensor([10.0, 10.0])}
    oracle = {"weight": torch.tensor([11.0, 10.0])}
    same_update = {"weight": torch.tensor([12.0, 10.0])}
    orthogonal_update = {"weight": torch.tensor([10.0, 11.0])}

    assert update_cosine(initial, oracle, same_update) == pytest.approx(1.0)
    assert update_cosine(initial, oracle, orthogonal_update) == pytest.approx(0.0)


def test_selected_hf_loader_reads_only_requested_safetensors(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file(
        {
            "keep": torch.tensor([1.0, 2.0]),
            "ignore": torch.ones(100),
        },
        tmp_path / "model.safetensors",
    )

    selected = load_hf_selected_tensors(tmp_path, ("keep",))

    assert tuple(selected) == ("keep",)
    assert torch.equal(selected["keep"], torch.tensor([1.0, 2.0]))


def test_state_probe_command_uses_torchrun_and_preserves_revision(tmp_path) -> None:
    command = build_state_probe_command(
        source_model="Qwen/model",
        revision="immutable-revision",
        checkpoint=tmp_path / "step-00000020",
        output=tmp_path / "selected.pt",
        timeout_seconds=900,
    )

    assert "--nproc-per-node=1" in command
    assert "bench.correctness_state" in command
    assert command[command.index("--revision") + 1] == "immutable-revision"
    assert command[command.index("--timeout_seconds") + 1] == "900"


def test_legacy_comparison_mode_remains_available(tmp_path) -> None:
    config = parse_correctness_args(
        [
            "--gate",
            "sft",
            "--comparisons",
            str(tmp_path / "comparisons.json"),
            "--output",
            str(tmp_path / "gate.json"),
        ]
    )

    assert config.execute is False
    assert config.comparisons == str(tmp_path / "comparisons.json")
    assert config.output == str(tmp_path / "gate.json")


def test_sft_seed_comparison_uses_real_records_and_selected_updates(tmp_path) -> None:
    oracle_dir = tmp_path / "oracle"
    fsdp_dir = tmp_path / "fsdp2"
    for directory, losses, norms in (
        (oracle_dir, [2.0, 1.5], [1.0, 0.5]),
        (fsdp_dir, [2.001, 1.499], [1.005, 0.499]),
    ):
        directory.mkdir()
        (directory / "run_config.json").write_text(
            '{"model_digest":"model","data_digest":"data"}\n',
            encoding="utf-8",
        )
        (directory / "train.jsonl").write_text(
            "".join(
                f'{{"global_step":{step},"loss":{loss},"preclip_grad_norm":{norm}}}\n'
                for step, (loss, norm) in enumerate(zip(losses, norms), start=1)
            ),
            encoding="utf-8",
        )
    initial = {"weight": torch.tensor([0.0, 0.0])}
    oracle = {"weight": torch.tensor([1.0, 0.0])}
    fsdp2 = {"weight": torch.tensor([2.0, 0.0])}

    comparison = build_sft_seed_comparison(
        seed=41,
        oracle_dir=oracle_dir,
        fsdp2_dir=fsdp_dir,
        initial_tensors=initial,
        oracle_tensors=oracle,
        fsdp2_tensors=fsdp2,
        max_steps=2,
    )

    assert comparison["first_loss_abs_error"] == pytest.approx(0.001)
    assert comparison["grad_norm_relative_error_max"] == pytest.approx(0.005)
    assert comparison["update_cosine_min"] == pytest.approx(1.0)
    assert comparison["loss_curve"] == {
        "oracle": [2.0, 1.5],
        "fsdp2": [2.001, 1.499],
    }
    assert len(comparison["oracle_record_sha256"]) == 64


def test_oracle_grpo_curve_extraction_uses_step_logs_and_reward_mean() -> None:
    curves = extract_oracle_grpo_curves(
        {
            "log_history": [
                {
                    "step": 1,
                    "loss": 0.2,
                    "grad_norm": 1.0,
                    "rewards/exact_match/mean": 0.25,
                },
                {"step": 1, "train_runtime": 10.0},
                {
                    "step": 2,
                    "loss": 0.1,
                    "grad_norm": 0.5,
                    "rewards/exact_match/mean": 0.5,
                },
            ]
        },
        max_steps=2,
    )

    assert curves == {
        "loss": [0.2, 0.1],
        "grad_norm": [1.0, 0.5],
        "reward": [0.25, 0.5],
    }


def test_fixed_rollout_fixture_is_recomputed_for_the_gate() -> None:
    assert fixed_rollout_loss_abs_error() < 1e-12


def test_generated_gate_outputs_do_not_make_the_next_gate_look_code_dirty(tmp_path) -> None:
    repository = tmp_path / "repo"
    output = repository / "results" / "fsdp_correctness"

    assert has_dirty_paths_outside(
        ["results/fsdp_correctness/sft_gate.json"],
        repository_root=repository,
        output_dir=output,
    ) is False
    assert has_dirty_paths_outside(
        ["results/fsdp_correctness/sft_gate.json", "bench/correctness.py"],
        repository_root=repository,
        output_dir=output,
    ) is True
