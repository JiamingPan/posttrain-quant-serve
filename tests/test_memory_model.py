import pytest

from train.memory_model import (
    assert_memory_fits,
    preflight_launch,
    predict_grpo_peak,
    predict_sft_peak,
)


QWEN3_8B_PARAMETERS = 8_190_735_360


@pytest.mark.parametrize(
    ("world_size", "allocated_gib", "reserved_gib"),
    [
        (1, 66.03, 71.31),
        (2, 38.22, 41.28),
        (4, 22.96, 24.80),
        (8, 15.33, 16.56),
    ],
)
def test_native_bf16_sft_predictions_match_the_approved_ledger(
    world_size: int,
    allocated_gib: float,
    reserved_gib: float,
) -> None:
    prediction = predict_sft_peak(QWEN3_8B_PARAMETERS, world_size, checkpointing=True)

    assert prediction.allocated_gib == pytest.approx(allocated_gib, abs=0.02)
    assert prediction.reserved_gib == pytest.approx(reserved_gib, abs=0.02)


def test_disabling_activation_checkpointing_adds_ten_gib() -> None:
    checkpointed = predict_sft_peak(QWEN3_8B_PARAMETERS, 4, checkpointing=True)
    uncheckpointed = predict_sft_peak(QWEN3_8B_PARAMETERS, 4, checkpointing=False)

    assert uncheckpointed.activations_gib - checkpointed.activations_gib == pytest.approx(10.0)
    assert uncheckpointed.allocated_gib - checkpointed.allocated_gib == pytest.approx(10.0)


def test_beta_zero_grpo_allocates_no_reference_model() -> None:
    prediction = predict_grpo_peak(
        QWEN3_8B_PARAMETERS,
        world_size=4,
        beta=0.0,
        rollout_mode="keep_unsharded",
    )

    assert prediction.reference_gib == 0.0
    assert prediction.training_phase_gib == pytest.approx(22.46, abs=0.02)
    assert prediction.rollout_phase_gib == pytest.approx(30.89, abs=0.02)
    assert prediction.peak_gib == prediction.rollout_phase_gib


def test_positive_beta_adds_exactly_one_frozen_bf16_reference_shard() -> None:
    without_reference = predict_grpo_peak(
        QWEN3_8B_PARAMETERS,
        world_size=4,
        beta=0.0,
        rollout_mode="keep_unsharded",
    )
    with_reference = predict_grpo_peak(
        QWEN3_8B_PARAMETERS,
        world_size=4,
        beta=0.1,
        rollout_mode="keep_unsharded",
    )

    assert with_reference.reference_gib == pytest.approx(3.8141, abs=0.001)
    assert with_reference.peak_gib - without_reference.peak_gib == pytest.approx(
        with_reference.reference_gib
    )


def test_memory_preflight_rejects_a_prediction_over_ninety_five_percent_capacity() -> None:
    prediction = predict_sft_peak(QWEN3_8B_PARAMETERS, 2, checkpointing=True)

    with pytest.raises(ValueError, match=r"Qwen/Qwen3-8B.*world_size=2.*bf16.*41\.28.*40\.00"):
        assert_memory_fits(
            prediction,
            capacity_gib=40.0,
            model_name="Qwen/Qwen3-8B",
            state_precision="bf16",
        )


def test_memory_preflight_accepts_a_prediction_with_headroom() -> None:
    prediction = predict_sft_peak(QWEN3_8B_PARAMETERS, 4, checkpointing=True)

    assert_memory_fits(
        prediction,
        capacity_gib=48.0,
        model_name="Qwen/Qwen3-8B",
        state_precision="bf16",
    )


def test_launch_preflight_honors_the_sft_checkpointing_ablation() -> None:
    result = preflight_launch(
        stage="sft",
        world_size=4,
        stage_args=("--model", "Qwen/Qwen3-8B", "--no_activation_checkpointing"),
        device_name="NVIDIA A100-SXM4-80GB",
        capacity_gib=79.2,
    )

    assert result["status"] == "fit"
    assert result["checkpointing"] is False
    assert result["predicted_reserved_gib"] == pytest.approx(35.60, abs=0.03)


def test_launch_preflight_rejects_non_a100_world_size_one_qwen3() -> None:
    with pytest.raises(ValueError, match="world-size-1.*A100 80"):
        preflight_launch(
            stage="sft",
            world_size=1,
            stage_args=("--model", "Qwen/Qwen3-8B"),
            device_name="NVIDIA A40",
            capacity_gib=44.4,
        )


def test_launch_preflight_rejects_fp32_resident_world_size_one() -> None:
    with pytest.raises(ValueError, match="fp32-resident.*world size one"):
        preflight_launch(
            stage="sft",
            world_size=1,
            stage_args=(
                "--model",
                "Qwen/Qwen3-8B",
                "--resident_precision",
                "fp32",
            ),
            device_name="NVIDIA A100-SXM4-80GB",
            capacity_gib=79.2,
        )


def test_grpo_auto_preflight_selects_reshard_when_full_policy_is_unsafe() -> None:
    result = preflight_launch(
        stage="grpo",
        world_size=4,
        stage_args=(
            "--model",
            "Qwen/Qwen3-8B",
            "--rollout_mode",
            "auto",
            "--beta",
            "0.0",
        ),
        device_name="NVIDIA A40",
        capacity_gib=26.0,
    )

    assert result["status"] == "fit"
    assert result["rollout_mode"] == "reshard"
    assert result["predicted_reserved_gib"] < 26.0 * 0.95


def test_launch_preflight_skips_unknown_model_sizes_without_guessing() -> None:
    result = preflight_launch(
        stage="sft",
        world_size=1,
        stage_args=("--model", "/models/tiny-qwen"),
        device_name="NVIDIA A40",
        capacity_gib=44.4,
    )

    assert result == {
        "capacity_gib": 44.4,
        "device_name": "NVIDIA A40",
        "model": "/models/tiny-qwen",
        "stage": "sft",
        "status": "skipped_unknown_model_size",
        "world_size": 1,
    }
