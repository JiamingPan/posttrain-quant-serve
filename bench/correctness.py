"""Validate and publish the one-GPU SFT/GRPO parity gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shlex
from statistics import fmean, variance
import subprocess
import sys
from typing import Any, Literal, Mapping, Sequence


GateName = Literal["sft", "grpo"]
EXPECTED_SEEDS = (41, 42, 43)
FIRST_LOSS_ATOL = 5e-3
FIXED_ROLLOUT_ATOL = 5e-4
GRAD_NORM_RTOL = 1e-2
UPDATE_COSINE_MIN = 0.999
RESUME_NEXT_LOSS_ATOL = 1e-6
CURVE_NOISE_FLOOR = 1e-3


@dataclass(frozen=True)
class CorrectnessConfig:
    gate: GateName
    execute: bool
    comparisons: str | None = None
    output: str | None = None
    output_dir: str | None = None
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    seeds: tuple[int, ...] = EXPECTED_SEEDS
    dataset_limit: int = 16
    max_steps: int = 20
    num_generations: int = 8
    sequence_length: int = 512
    max_completion_length: int = 128
    timeout_seconds: int = 600


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if seeds != EXPECTED_SEEDS:
        raise argparse.ArgumentTypeError("correctness execution requires seeds 41,42,43")
    return seeds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=("sft", "grpo"), required=True)
    parser.add_argument("--comparisons")
    parser.add_argument("--output")
    parser.add_argument("--output_dir")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--seeds", type=_parse_seeds, default=EXPECTED_SEEDS)
    parser.add_argument("--dataset_limit", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    return parser


def parse_correctness_args(argv: Sequence[str] | None = None) -> CorrectnessConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)
    legacy_values = (args.comparisons, args.output)
    if any(legacy_values) and not all(legacy_values):
        parser.error("legacy mode requires both --comparisons and --output")
    execute = not all(legacy_values)
    if execute and args.output_dir is None:
        parser.error("execution mode requires --output_dir")
    if args.dataset_limit <= 0 or args.max_steps <= 0:
        parser.error("dataset_limit and max_steps must be positive")
    if args.num_generations < 2 or args.max_completion_length <= 0:
        parser.error("num_generations must be at least two and completion length positive")
    return CorrectnessConfig(execute=execute, **vars(args))


def _torchrun_module(module: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        "--module",
        module,
    ]


def _optional_revision(revision: str | None) -> list[str]:
    return ["--revision", revision] if revision is not None else []


def build_sft_commands(
    config: CorrectnessConfig,
    *,
    seed: int,
) -> dict[str, list[str]]:
    if config.gate != "sft" or config.output_dir is None:
        raise ValueError("SFT command construction requires SFT execution mode")
    root = Path(config.output_dir) / "runs" / "sft" / f"seed-{seed}"
    commands: dict[str, list[str]] = {}
    for implementation, sharding in (("oracle", "none"), ("fsdp2", "fsdp2")):
        commands[implementation] = [
            *_torchrun_module("train.fsdp_sft"),
            "--output_dir",
            str(root / implementation),
            "--model",
            config.model,
            *_optional_revision(config.revision),
            "--sharding",
            sharding,
            "--resident_precision",
            "bf16",
            "--activation_checkpointing",
            "--accumulation_sync",
            "reduce_scatter",
            "--dataset_limit",
            str(config.dataset_limit),
            "--sequence_length",
            str(config.sequence_length),
            "--local_microbatch_size",
            "1",
            "--gradient_accumulation_steps",
            "1",
            "--max_steps",
            str(config.max_steps),
            "--seed",
            str(seed),
            "--logging_steps",
            "1",
            "--save_steps",
            str(config.max_steps),
            "--attention_backend",
            "sdpa",
            "--timeout_seconds",
            str(config.timeout_seconds),
        ]
    return commands


def build_grpo_commands(
    config: CorrectnessConfig,
    *,
    seed: int,
) -> dict[str, list[str]]:
    if config.gate != "grpo" or config.output_dir is None:
        raise ValueError("GRPO command construction requires GRPO execution mode")
    root = Path(config.output_dir) / "runs" / "grpo" / f"seed-{seed}"
    oracle_output = root / "oracle"
    fsdp_output = root / "fsdp2"
    resume_output = root / "resume"
    primary_steps = config.max_steps + 1
    common_fsdp = [
        "--model",
        config.model,
        *_optional_revision(config.revision),
        "--resident_precision",
        "bf16",
        "--activation_checkpointing",
        "--accumulation_sync",
        "reduce_scatter",
        "--dataset_limit",
        str(config.dataset_limit),
        "--num_generations",
        str(config.num_generations),
        "--gradient_accumulation_steps",
        "1",
        "--policy_microbatch_size",
        "1",
        "--learning_rate",
        "1e-6",
        "--beta",
        "0.0",
        "--scale_rewards",
        "none",
        "--temperature",
        "1.0",
        "--top_p",
        "1.0",
        "--top_k",
        "0",
        "--max_prompt_length",
        "512",
        "--max_completion_length",
        str(config.max_completion_length),
        "--rollout_mode",
        "reshard",
        "--seed",
        str(seed),
        "--logging_steps",
        "1",
        "--attention_backend",
        "sdpa",
        "--timeout_seconds",
        str(config.timeout_seconds),
    ]
    oracle = [
        sys.executable,
        "-m",
        "scripts.train_grpo_gsm8k",
        "--model",
        config.model,
        "--output_dir",
        str(oracle_output),
        "--dataset_limit",
        str(config.dataset_limit),
        "--max_steps",
        str(config.max_steps),
        "--num_generations",
        str(config.num_generations),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "1e-6",
        "--beta",
        "0.0",
        "--max_prompt_length",
        "512",
        "--max_completion_length",
        str(config.max_completion_length),
        "--logging_steps",
        "1",
        "--save_steps",
        str(config.max_steps),
        "--seed",
        str(seed),
        "--run_record",
        str(oracle_output / "oracle_run.json"),
        "--temperature",
        "1.0",
        "--top_p",
        "1.0",
        "--top_k",
        "0",
        "--scale_rewards",
        "none",
        "--loss_type",
        "dr_grpo",
        "--epsilon_high",
        "0.2",
    ]
    primary = [
        *_torchrun_module("train.fsdp_grpo"),
        *common_fsdp,
        "--output_dir",
        str(fsdp_output),
        "--max_steps",
        str(primary_steps),
        "--save_steps",
        str(config.max_steps),
    ]
    resume = [
        *_torchrun_module("train.fsdp_grpo"),
        *common_fsdp,
        "--output_dir",
        str(resume_output),
        "--resume",
        str(fsdp_output / "checkpoints" / f"step-{config.max_steps:08d}"),
        "--max_steps",
        str(primary_steps),
        "--save_steps",
        str(primary_steps),
    ]
    return {"oracle": oracle, "fsdp2": primary, "resume": resume}


def build_state_probe_command(
    *,
    source_model: str,
    revision: str | None,
    checkpoint: str | Path,
    output: str | Path,
    timeout_seconds: int,
) -> list[str]:
    command = [
        *_torchrun_module("bench.correctness_state"),
        "--source_model",
        source_model,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--timeout_seconds",
        str(timeout_seconds),
    ]
    if revision is not None:
        command.extend(("--revision", revision))
    return command


def relative_error_max(reference: Sequence[float], actual: Sequence[float]) -> float:
    if len(reference) != len(actual) or not reference:
        raise ValueError("relative-error curves must be equal non-empty sequences")
    return max(
        abs(float(left) - float(right))
        / max(abs(float(left)), 1e-8)
        for left, right in zip(reference, actual)
    )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _training_rows(path: str | Path, *, max_steps: int) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training JSONL: {path}") from error
    rows = sorted(
        (row for row in rows if int(row.get("global_step", -1)) <= max_steps),
        key=lambda row: int(row["global_step"]),
    )
    if [int(row["global_step"]) for row in rows] != list(range(1, max_steps + 1)):
        raise ValueError(f"training record does not contain steps 1..{max_steps}: {path}")
    return rows


def build_sft_seed_comparison(
    *,
    seed: int,
    oracle_dir: str | Path,
    fsdp2_dir: str | Path,
    initial_tensors: Mapping[str, Any],
    oracle_tensors: Mapping[str, Any],
    fsdp2_tensors: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    from bench.correctness_state import update_cosine

    oracle_root = Path(oracle_dir)
    fsdp_root = Path(fsdp2_dir)
    oracle_config = _json_object(oracle_root / "run_config.json")
    fsdp_config = _json_object(fsdp_root / "run_config.json")
    for field in ("model_digest", "data_digest"):
        if oracle_config.get(field) != fsdp_config.get(field):
            raise ValueError(f"SFT correctness runs have mismatched {field}")
    oracle_record = oracle_root / "train.jsonl"
    fsdp_record = fsdp_root / "train.jsonl"
    oracle_rows = _training_rows(oracle_record, max_steps=max_steps)
    fsdp_rows = _training_rows(fsdp_record, max_steps=max_steps)
    oracle_losses = [float(row["loss"]) for row in oracle_rows]
    fsdp_losses = [float(row["loss"]) for row in fsdp_rows]
    oracle_norms = [float(row["preclip_grad_norm"]) for row in oracle_rows]
    fsdp_norms = [float(row["preclip_grad_norm"]) for row in fsdp_rows]
    return {
        "seed": seed,
        "model_digest": str(oracle_config["model_digest"]),
        "dataset_digest": str(oracle_config["data_digest"]),
        "oracle_record": str(oracle_record),
        "fsdp2_record": str(fsdp_record),
        "oracle_record_sha256": _sha256_file(oracle_record),
        "fsdp2_record_sha256": _sha256_file(fsdp_record),
        "first_loss_abs_error": abs(oracle_losses[0] - fsdp_losses[0]),
        "grad_norm_relative_error_max": relative_error_max(
            oracle_norms,
            fsdp_norms,
        ),
        "update_cosine_min": update_cosine(
            initial_tensors,
            oracle_tensors,
            fsdp2_tensors,
        ),
        "loss_curve": {"oracle": oracle_losses, "fsdp2": fsdp_losses},
    }


def extract_oracle_grpo_curves(
    record: Mapping[str, Any],
    *,
    max_steps: int,
) -> dict[str, list[float]]:
    step_rows: dict[int, Mapping[str, Any]] = {}
    for row in record.get("log_history", []):
        if not isinstance(row, Mapping) or "loss" not in row or "step" not in row:
            continue
        step = int(row["step"])
        if 1 <= step <= max_steps:
            step_rows[step] = row
    if set(step_rows) != set(range(1, max_steps + 1)):
        raise ValueError("oracle GRPO record is missing per-step loss entries")

    loss: list[float] = []
    grad_norm: list[float] = []
    reward: list[float] = []
    for step in range(1, max_steps + 1):
        row = step_rows[step]
        if "grad_norm" not in row:
            raise ValueError("oracle GRPO record is missing per-step grad_norm")
        reward_keys = sorted(
            key
            for key in row
            if "reward" in key.lower()
            and (
                key.lower() == "reward"
                or key.lower().endswith("/mean")
                or key.lower().endswith("_mean")
            )
        )
        if not reward_keys:
            raise ValueError("oracle GRPO record is missing per-step reward mean")
        loss.append(float(row["loss"]))
        grad_norm.append(float(row["grad_norm"]))
        reward.append(float(row[reward_keys[0]]))
    return {"loss": loss, "grad_norm": grad_norm, "reward": reward}


def fixed_rollout_loss_abs_error() -> float:
    import torch
    from train.grpo_core import grpo_loss_sum

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "grpo_fixed_rollout.json"
    )
    payload = _json_object(fixture_path)
    tensor_names = {
        "current_logps",
        "old_logps",
        "ref_logps",
        "advantages",
        "completion_mask",
    }
    arguments = {
        key: torch.tensor(value, dtype=torch.float64)
        if key in tensor_names
        else value
        for key, value in payload.items()
        if not key.startswith("expected_")
    }
    result = grpo_loss_sum(**arguments)
    return abs(float(result.loss_sum.item()) - float(payload["expected_loss_sum"]))


def _run_command(command: Sequence[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(tuple(command), check=True)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(rendered)


def _load_selected_probe(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as error:
        raise ValueError(f"missing selected-tensor probe: {path}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"invalid selected-tensor probe: {path}")
    return payload


def _source_state(config: CorrectnessConfig) -> tuple[Path, str, dict[str, Any]]:
    from transformers import AutoConfig
    from bench.correctness_state import (
        load_hf_selected_tensors,
        selected_qwen_tensor_names,
    )
    from train.checkpointing import resolve_hf_checkpoint_source

    source = resolve_hf_checkpoint_source(config.model, revision=config.revision)
    model_config = AutoConfig.from_pretrained(source.path)
    names = selected_qwen_tensor_names(int(model_config.num_hidden_layers))
    tensors = load_hf_selected_tensors(source.path, names)
    return source.path, source.revision, tensors


def _probe_checkpoint(
    config: CorrectnessConfig,
    *,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    command = build_state_probe_command(
        source_model=config.model,
        revision=config.revision,
        checkpoint=checkpoint,
        output=output,
        timeout_seconds=config.timeout_seconds,
    )
    _run_command(command)
    return _load_selected_probe(output)


def has_dirty_paths_outside(
    dirty_paths: Sequence[str],
    *,
    repository_root: str | Path,
    output_dir: str | Path,
) -> bool:
    repository = Path(repository_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = repository / output
    output = output.resolve()
    for dirty_path in dirty_paths:
        candidate = Path(dirty_path)
        if not candidate.is_absolute():
            candidate = repository / candidate
        try:
            candidate.resolve().relative_to(output)
        except ValueError:
            return True
    return False


def _execution_identity(config: CorrectnessConfig) -> dict[str, Any]:
    from train.run_tracking import capture_run_identity

    identity = capture_run_identity(f"correctness-{config.gate}", asdict(config))
    repository = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        check=True,
        capture_output=True,
    ).stdout
    dirty_paths: list[str] = []
    for raw_entry in status.split(b"\0"):
        if not raw_entry:
            continue
        entry = raw_entry.decode("utf-8", errors="surrogateescape")
        dirty_paths.append(entry[3:] if len(entry) >= 3 and entry[2] == " " else entry)
    source_dirty = has_dirty_paths_outside(
        dirty_paths,
        repository_root=repository,
        output_dir=config.output_dir or "",
    )
    if source_dirty:
        raise RuntimeError("correctness execution refuses an uncommitted worktree")
    if identity.slurm_job_id is None:
        raise RuntimeError("correctness execution requires a Slurm allocation")
    payload = identity.as_dict()
    payload["git_dirty"] = False
    payload["ignored_generated_output_changes"] = bool(dirty_paths)
    return payload


def _publish_execution_gate(
    config: CorrectnessConfig,
    *,
    comparisons: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    commands_manifest: Path,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir or "")
    comparison_path = output_dir / f"{config.gate}_comparisons.json"
    _write_json_exclusive(comparison_path, list(comparisons))
    record = build_gate_record(config.gate, comparisons)
    record.update(
        {
            "run_identity": dict(identity),
            "commands_manifest": str(commands_manifest),
            "commands_manifest_sha256": _sha256_file(commands_manifest),
            "comparisons_record": str(comparison_path),
            "comparisons_record_sha256": _sha256_file(comparison_path),
        }
    )
    validate_gate_record(record)
    destination = output_dir / f"{config.gate}_gate.json"
    _write_json_exclusive(destination, record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def _execute_sft_gate(
    config: CorrectnessConfig,
    *,
    identity: Mapping[str, Any],
    commands_manifest: Path,
) -> dict[str, Any]:
    _, _, initial_tensors = _source_state(config)
    comparisons: list[dict[str, Any]] = []
    for seed in config.seeds:
        commands = build_sft_commands(config, seed=seed)
        _run_command(commands["oracle"])
        _run_command(commands["fsdp2"])
        seed_root = Path(config.output_dir or "") / "runs" / "sft" / f"seed-{seed}"
        oracle_dir = seed_root / "oracle"
        fsdp2_dir = seed_root / "fsdp2"
        oracle_probe = seed_root / "oracle-selected.pt"
        fsdp2_probe = seed_root / "fsdp2-selected.pt"
        oracle_tensors = _probe_checkpoint(
            config,
            checkpoint=(
                oracle_dir
                / "checkpoints"
                / f"step-{config.max_steps:08d}"
            ),
            output=oracle_probe,
        )
        fsdp2_tensors = _probe_checkpoint(
            config,
            checkpoint=(
                fsdp2_dir
                / "checkpoints"
                / f"step-{config.max_steps:08d}"
            ),
            output=fsdp2_probe,
        )
        comparison = build_sft_seed_comparison(
            seed=seed,
            oracle_dir=oracle_dir,
            fsdp2_dir=fsdp2_dir,
            initial_tensors=initial_tensors,
            oracle_tensors=oracle_tensors,
            fsdp2_tensors=fsdp2_tensors,
            max_steps=config.max_steps,
        )
        comparison["evidence_records"] = [
            {
                "implementation": "oracle_selected_updates",
                "path": str(oracle_probe),
                "sha256": _sha256_file(oracle_probe),
            },
            {
                "implementation": "fsdp2_selected_updates",
                "path": str(fsdp2_probe),
                "sha256": _sha256_file(fsdp2_probe),
            },
        ]
        comparisons.append(comparison)
    return _publish_execution_gate(
        config,
        comparisons=comparisons,
        identity=identity,
        commands_manifest=commands_manifest,
    )


def _training_row_at(path: str | Path, step: int) -> dict[str, Any]:
    try:
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid training JSONL: {path}") from error
    matches = [row for row in rows if int(row.get("global_step", -1)) == step]
    if len(matches) != 1:
        raise ValueError(f"training record must contain exactly one step {step}: {path}")
    return matches[0]


def _build_grpo_seed_comparison(
    config: CorrectnessConfig,
    *,
    seed: int,
    initial_tensors: Mapping[str, Any],
    oracle_tensors: Mapping[str, Any],
    fsdp2_tensors: Mapping[str, Any],
) -> dict[str, Any]:
    from bench.correctness_state import update_cosine

    root = Path(config.output_dir or "") / "runs" / "grpo" / f"seed-{seed}"
    oracle_record_path = root / "oracle" / "oracle_run.json"
    fsdp_record_path = root / "fsdp2" / "train.jsonl"
    resume_record_path = root / "resume" / "train.jsonl"
    oracle_record = _json_object(oracle_record_path)
    oracle_curves = extract_oracle_grpo_curves(
        oracle_record,
        max_steps=config.max_steps,
    )
    fsdp_rows = _training_rows(fsdp_record_path, max_steps=config.max_steps)
    fsdp_loss = [float(row["loss"]) for row in fsdp_rows]
    fsdp_norm = [float(row["preclip_grad_norm"]) for row in fsdp_rows]
    fsdp_reward = [float(row["reward_mean"]) for row in fsdp_rows]
    fsdp_config = _json_object(root / "fsdp2" / "run_config.json")
    if oracle_record.get("model_digest") != fsdp_config.get("model_revision"):
        raise ValueError("GRPO correctness runs have mismatched model digests")
    if oracle_record.get("dataset_digest") != fsdp_config.get("data_digest"):
        raise ValueError("GRPO correctness runs have mismatched dataset digests")
    uninterrupted = _training_row_at(
        fsdp_record_path,
        config.max_steps + 1,
    )
    resumed = _training_row_at(
        resume_record_path,
        config.max_steps + 1,
    )
    return {
        "seed": seed,
        "model_digest": str(oracle_record["model_digest"]),
        "dataset_digest": str(oracle_record["dataset_digest"]),
        "oracle_record": str(oracle_record_path),
        "fsdp2_record": str(fsdp_record_path),
        "oracle_record_sha256": _sha256_file(oracle_record_path),
        "fsdp2_record_sha256": _sha256_file(fsdp_record_path),
        "first_loss_abs_error": abs(oracle_curves["loss"][0] - fsdp_loss[0]),
        "fixed_rollout_loss_abs_error": fixed_rollout_loss_abs_error(),
        "grad_norm_relative_error_max": relative_error_max(
            oracle_curves["grad_norm"],
            fsdp_norm,
        ),
        "update_cosine_min": update_cosine(
            initial_tensors,
            oracle_tensors,
            fsdp2_tensors,
        ),
        "resume_next_loss_abs_error": abs(
            float(uninterrupted["loss"]) - float(resumed["loss"])
        ),
        "loss_curve": {"oracle": oracle_curves["loss"], "fsdp2": fsdp_loss},
        "reward_curve": {
            "oracle": oracle_curves["reward"],
            "fsdp2": fsdp_reward,
        },
        "evidence_records": [
            {
                "implementation": "fsdp2_resume",
                "path": str(resume_record_path),
                "sha256": _sha256_file(resume_record_path),
            }
        ],
    }


def _execute_grpo_gate(
    config: CorrectnessConfig,
    *,
    identity: Mapping[str, Any],
    commands_manifest: Path,
) -> dict[str, Any]:
    from bench.correctness_state import load_hf_selected_tensors

    _, _, initial_tensors = _source_state(config)
    comparisons: list[dict[str, Any]] = []
    for seed in config.seeds:
        commands = build_grpo_commands(config, seed=seed)
        _run_command(commands["oracle"])
        _run_command(commands["fsdp2"])
        _run_command(commands["resume"])
        seed_root = Path(config.output_dir or "") / "runs" / "grpo" / f"seed-{seed}"
        oracle_tensors = load_hf_selected_tensors(
            seed_root / "oracle",
            tuple(initial_tensors),
        )
        fsdp2_probe = seed_root / "fsdp2-selected.pt"
        fsdp2_tensors = _probe_checkpoint(
            config,
            checkpoint=(
                seed_root
                / "fsdp2"
                / "checkpoints"
                / f"step-{config.max_steps:08d}"
            ),
            output=fsdp2_probe,
        )
        comparison = _build_grpo_seed_comparison(
            config,
            seed=seed,
            initial_tensors=initial_tensors,
            oracle_tensors=oracle_tensors,
            fsdp2_tensors=fsdp2_tensors,
        )
        comparison["evidence_records"].append(
            {
                "implementation": "fsdp2_selected_updates",
                "path": str(fsdp2_probe),
                "sha256": _sha256_file(fsdp2_probe),
            }
        )
        comparisons.append(comparison)
    return _publish_execution_gate(
        config,
        comparisons=comparisons,
        identity=identity,
        commands_manifest=commands_manifest,
    )


def execute_correctness_gate(config: CorrectnessConfig) -> dict[str, Any]:
    if not config.execute or config.output_dir is None:
        raise ValueError("correctness execution requires execution-mode configuration")
    identity = _execution_identity(config)
    commands_by_seed = {
        str(seed): (
            build_sft_commands(config, seed=seed)
            if config.gate == "sft"
            else build_grpo_commands(config, seed=seed)
        )
        for seed in config.seeds
    }
    commands_manifest = Path(config.output_dir) / f"{config.gate}_commands.json"
    _write_json_exclusive(
        commands_manifest,
        {
            "config": asdict(config),
            "commands": commands_by_seed,
            "run_identity": identity,
        },
    )
    if config.gate == "sft":
        return _execute_sft_gate(
            config,
            identity=identity,
            commands_manifest=commands_manifest,
        )
    return _execute_grpo_gate(
        config,
        identity=identity,
        commands_manifest=commands_manifest,
    )


def _curve_points(
    comparisons: Sequence[Mapping[str, Any]],
    field: str,
) -> list[dict[str, float | int]]:
    oracle_curves = [list(row[field]["oracle"]) for row in comparisons]
    fsdp_curves = [list(row[field]["fsdp2"]) for row in comparisons]
    lengths = {len(curve) for curve in (*oracle_curves, *fsdp_curves)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError(f"{field} must contain equal non-empty curves")
    points: list[dict[str, float | int]] = []
    for step in range(next(iter(lengths))):
        oracle = [float(curve[step]) for curve in oracle_curves]
        fsdp2 = [float(curve[step]) for curve in fsdp_curves]
        mean_abs_diff = abs(fmean(oracle) - fmean(fsdp2))
        noise_bound = (
            2.0
            * (variance(oracle) / len(oracle) + variance(fsdp2) / len(fsdp2))
            ** 0.5
            + CURVE_NOISE_FLOOR
        )
        points.append(
            {
                "step": step + 1,
                "oracle_mean": fmean(oracle),
                "fsdp2_mean": fmean(fsdp2),
                "mean_abs_diff": mean_abs_diff,
                "noise_bound": noise_bound,
            }
        )
    return points


def _consistent_value(
    comparisons: Sequence[Mapping[str, Any]],
    field: str,
) -> str:
    values = {str(row[field]) for row in comparisons}
    if len(values) != 1:
        label = field.replace("_", " ")
        raise ValueError(f"correctness comparisons have mismatched {label}s")
    return values.pop()


def build_gate_record(
    gate: GateName,
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate three seed comparisons and reject before publication on failure."""

    if gate not in {"sft", "grpo"}:
        raise ValueError("gate must be 'sft' or 'grpo'")
    if len(comparisons) != 3:
        raise ValueError("correctness gate requires exactly three seed comparisons")
    seeds = sorted(int(row["seed"]) for row in comparisons)
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError("correctness gate requires exactly three seeds: 41, 42, 43")
    model_digest = _consistent_value(comparisons, "model_digest")
    dataset_digest = _consistent_value(comparisons, "dataset_digest")
    source_records: list[dict[str, Any]] = []
    for row in comparisons:
        for implementation in ("oracle", "fsdp2"):
            source_records.append(
                {
                    "seed": int(row["seed"]),
                    "implementation": implementation,
                    "path": str(row[f"{implementation}_record"]),
                    "sha256": str(row[f"{implementation}_record_sha256"]),
                }
            )
        for evidence in row.get("evidence_records", []):
            source_records.append(
                {
                    "seed": int(row["seed"]),
                    "implementation": str(evidence["implementation"]),
                    "path": str(evidence["path"]),
                    "sha256": str(evidence["sha256"]),
                }
            )

    metrics: dict[str, Any] = {
        "first_loss_abs_error_max": max(
            float(row["first_loss_abs_error"]) for row in comparisons
        ),
        "grad_norm_relative_error_max": max(
            float(row["grad_norm_relative_error_max"]) for row in comparisons
        ),
        "update_cosine_min": min(
            float(row["update_cosine_min"]) for row in comparisons
        ),
        "loss_curve_points": _curve_points(comparisons, "loss_curve"),
    }
    if gate == "grpo":
        metrics.update(
            fixed_rollout_loss_abs_error_max=max(
                float(row["fixed_rollout_loss_abs_error"])
                for row in comparisons
            ),
            resume_next_loss_abs_error_max=max(
                float(row["resume_next_loss_abs_error"])
                for row in comparisons
            ),
            reward_curve_points=_curve_points(comparisons, "reward_curve"),
        )
    record = {
        "format_version": 1,
        "gate": gate,
        "status": "pass",
        "seeds": seeds,
        "model_digest": model_digest,
        "dataset_digest": dataset_digest,
        "thresholds": {
            "first_loss_abs_error_max": FIRST_LOSS_ATOL,
            "fixed_rollout_loss_abs_error_max": FIXED_ROLLOUT_ATOL,
            "grad_norm_relative_error_max": GRAD_NORM_RTOL,
            "update_cosine_min": UPDATE_COSINE_MIN,
            "resume_next_loss_abs_error_max": RESUME_NEXT_LOSS_ATOL,
            "curve_noise_floor": CURVE_NOISE_FLOOR,
        },
        "metrics": metrics,
        "source_records": source_records,
    }
    validate_gate_record(record)
    return record


def _validate_curve(points: Any, *, label: str) -> None:
    if not isinstance(points, list) or not points:
        raise ValueError(f"{label} curve points are missing")
    for point in points:
        if float(point["mean_abs_diff"]) > float(point["noise_bound"]):
            raise ValueError(f"{label} curve exceeds its three-seed noise bound")


def validate_gate_record(record: Mapping[str, Any]) -> None:
    """Apply fixed acceptance thresholds; record-declared thresholds are informational."""

    gate = record.get("gate")
    if gate not in {"sft", "grpo"}:
        raise ValueError("correctness record has an invalid gate")
    seeds = list(record.get("seeds", []))
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError("correctness gate requires exactly three seeds: 41, 42, 43")
    if not record.get("model_digest") or not record.get("dataset_digest"):
        raise ValueError("correctness record is missing source digests")
    sources = record.get("source_records")
    if not isinstance(sources, list) or len(sources) < 6:
        raise ValueError("correctness gate requires at least six traceable source records")
    for source in sources:
        if not source.get("path") or len(str(source.get("sha256", ""))) != 64:
            raise ValueError("source records require paths and SHA-256 digests")

    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("correctness record is missing metrics")
    if float(metrics["first_loss_abs_error_max"]) > FIRST_LOSS_ATOL:
        raise ValueError("first loss exceeds the 5e-3 absolute tolerance")
    if float(metrics["grad_norm_relative_error_max"]) > GRAD_NORM_RTOL:
        raise ValueError("gradient norm exceeds the 1e-2 relative tolerance")
    if float(metrics["update_cosine_min"]) < UPDATE_COSINE_MIN:
        raise ValueError("selected update cosine must be at least 0.999")
    _validate_curve(metrics["loss_curve_points"], label="loss")
    if gate == "grpo":
        if float(metrics["fixed_rollout_loss_abs_error_max"]) > FIXED_ROLLOUT_ATOL:
            raise ValueError("fixed-rollout loss exceeds the 5e-4 tolerance")
        if float(metrics["resume_next_loss_abs_error_max"]) > RESUME_NEXT_LOSS_ATOL:
            raise ValueError("resume next-step loss exceeds the 1e-6 tolerance")
        _validate_curve(metrics["reward_curve_points"], label="reward")
    if record.get("status") != "pass":
        raise ValueError("a published correctness gate must have pass status")


def _run_gate(
    gate: GateName,
    comparisons_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    comparisons_file = Path(comparisons_path)
    try:
        comparisons = json.loads(comparisons_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid comparison input: {comparisons_file}") from error
    if not isinstance(comparisons, list):
        raise ValueError("comparison input must be a JSON list")
    record = build_gate_record(gate, comparisons)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"gate record already exists: {destination}")
    destination.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def run_sft_gate(
    comparisons_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return _run_gate("sft", comparisons_path, output_path)


def run_grpo_gate(
    comparisons_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    return _run_gate("grpo", comparisons_path, output_path)


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_correctness_args(argv)
    if config.execute:
        execute_correctness_gate(config)
        return
    runner = run_sft_gate if config.gate == "sft" else run_grpo_gate
    record = runner(config.comparisons or "", config.output or "")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
