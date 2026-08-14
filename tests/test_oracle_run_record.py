from __future__ import annotations

import json

from scripts.train_grpo_gsm8k import (
    _raw_dataset_digest,
    build_training_config_kwargs,
    parse_args,
    write_oracle_run_record,
)


def test_oracle_training_config_receives_the_explicit_seed() -> None:
    args = parse_args(["--output_dir", "/run", "--seed", "43"])

    config = build_training_config_kwargs(args)

    assert config["seed"] == 43
    assert config["num_generations"] == 4
    assert config["gradient_accumulation_steps"] == 4


def test_oracle_record_contains_resolved_config_and_log_history(tmp_path) -> None:
    path = tmp_path / "oracle.json"
    args = parse_args(
        ["--output_dir", "/run", "--seed", "41", "--run_record", str(path)]
    )

    written = write_oracle_run_record(
        args.run_record,
        args=args,
        resolved_config={"seed": 41, "loss_type": "dr_grpo"},
        log_history=[{"step": 1, "loss": 0.25}],
        dataset_digest="fixed-data",
        model_digest="fixed-model",
    )

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "args": vars(args),
        "dataset_digest": "fixed-data",
        "log_history": [{"loss": 0.25, "step": 1}],
        "model_digest": "fixed-model",
        "resolved_config": {"loss_type": "dr_grpo", "seed": 41},
    }


def test_oracle_record_is_opt_in(tmp_path) -> None:
    assert write_oracle_run_record(
        None,
        args=parse_args(["--output_dir", "/run"]),
        resolved_config={},
        log_history=[],
        dataset_digest="data",
        model_digest="model",
    ) is None
    assert list(tmp_path.iterdir()) == []


def test_oracle_source_digest_matches_raw_question_answer_rows() -> None:
    rows = [
        {"question": "2+2?", "answer": "work #### 4"},
        {"question": "3+3?", "answer": "work #### 6"},
    ]

    assert _raw_dataset_digest(rows) == (
        "d24301521cd065cd989ee850f935589578e769ba0249e480ab0baf1bf9cf89dc"
    )
