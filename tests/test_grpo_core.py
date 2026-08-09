from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.gsm8k_reward import gsm8k_exact_match_reward
from train.grpo_core import (
    RolloutBatch,
    grpo_loss_sum,
    group_advantages,
    select_token_logps,
)


FIXTURE = Path(__file__).parent / "fixtures" / "grpo_fixed_rollout.json"


def _fixture_arguments(payload: dict[str, object]) -> dict[str, object]:
    tensor_names = {
        "current_logps",
        "old_logps",
        "ref_logps",
        "advantages",
        "completion_mask",
    }
    return {
        key: torch.tensor(value, dtype=torch.float64)
        if key in tensor_names
        else value
        for key, value in payload.items()
        if not key.startswith("expected_")
    }


def test_scale_rewards_none_centers_without_standardizing() -> None:
    rewards = torch.tensor([[1.0, 0.0, -0.25, 0.0]])

    actual = group_advantages(rewards, scale_rewards="none")

    assert torch.equal(actual, rewards - rewards.mean(dim=1, keepdim=True))


def test_group_and_batch_reward_scaling_use_explicit_denominators() -> None:
    rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    centered = rewards - rewards.mean(dim=1, keepdim=True)

    grouped = group_advantages(rewards, scale_rewards="group")
    batched = group_advantages(rewards, scale_rewards="batch")

    assert torch.allclose(
        grouped,
        centered / rewards.std(dim=1, keepdim=True, unbiased=False),
    )
    assert torch.allclose(batched, centered / rewards.std(unbiased=False))


def test_select_token_logps_selects_next_tokens_and_completion_suffix() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -1.0],
                [-2.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ]
    )
    token_ids = torch.tensor([[0, 1, 2, 0]])
    expected_all = logits[:, :-1].log_softmax(dim=-1).gather(
        -1,
        token_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)

    actual = select_token_logps(logits, token_ids, completion_length=2)

    assert torch.allclose(actual, expected_all[:, -2:])


def test_dr_grpo_matches_independent_scalar_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    result = grpo_loss_sum(**_fixture_arguments(payload))

    assert result.loss_sum.item() == pytest.approx(payload["expected_loss_sum"], abs=1e-12)
    assert result.policy_loss_sum.item() == pytest.approx(
        payload["expected_policy_loss_sum"], abs=1e-12
    )
    assert result.kl_sum.item() == pytest.approx(payload["expected_kl_sum"], abs=1e-12)
    assert result.clip_ratio.item() == pytest.approx(
        payload["expected_clip_ratio"], abs=1e-12
    )
    assert result.normalized_loss.item() == pytest.approx(
        payload["expected_normalized_loss"], abs=1e-12
    )
    assert result.normalizer == 6
    assert result.valid_tokens == 4


def test_beta_zero_removes_reference_kl_from_both_loss_and_metrics() -> None:
    arguments = dict(
        current_logps=torch.tensor([[-0.2, -0.4]]),
        old_logps=torch.tensor([[-0.3, -0.3]]),
        advantages=torch.tensor([1.0]),
        completion_mask=torch.tensor([[1.0, 1.0]]),
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.0,
        max_completion_length=2,
    )

    without_reference = grpo_loss_sum(ref_logps=None, **arguments)
    with_reference = grpo_loss_sum(
        ref_logps=torch.tensor([[-10.0, -10.0]]),
        **arguments,
    )

    assert torch.equal(with_reference.loss_sum, without_reference.loss_sum)
    assert with_reference.kl_sum.item() == 0.0


def test_completion_mask_excludes_padding_and_dr_grpo_uses_fixed_denominator() -> None:
    result = grpo_loss_sum(
        current_logps=torch.zeros(2, 4),
        old_logps=torch.zeros(2, 4),
        ref_logps=None,
        advantages=torch.tensor([1.0, -1.0]),
        completion_mask=torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]]),
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.0,
        max_completion_length=4,
    )

    assert result.loss_sum.item() == 1.0
    assert result.valid_tokens == 3
    assert result.normalizer == 8
    assert result.normalized_loss.item() == pytest.approx(1 / 8)


def test_rollout_batch_requires_cpu_backing_and_consistent_shapes() -> None:
    batch = RolloutBatch(
        prompt_input_ids=torch.ones(2, 3, dtype=torch.long),
        prompt_attention_mask=torch.ones(2, 3, dtype=torch.long),
        completion_input_ids=torch.ones(2, 4, dtype=torch.long),
        completion_mask=torch.ones(2, 4, dtype=torch.bool),
        old_logps=torch.zeros(2, 4),
        rewards=torch.tensor([1.0, 0.0]),
        advantages=torch.tensor([0.5, -0.5]),
        ref_logps=None,
        completions=("#### 4", "#### 5"),
        answers=("work #### 4", "work #### 6"),
    )

    assert batch.batch_size == 2
    assert batch.max_completion_length == 4

    with pytest.raises(ValueError, match="batch dimension"):
        RolloutBatch(
            prompt_input_ids=torch.ones(1, 3, dtype=torch.long),
            prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
            completion_input_ids=torch.ones(2, 4, dtype=torch.long),
            completion_mask=torch.ones(2, 4, dtype=torch.bool),
            old_logps=torch.zeros(2, 4),
            rewards=torch.tensor([1.0, 0.0]),
            advantages=torch.tensor([0.5, -0.5]),
            ref_logps=None,
            completions=("a", "b"),
            answers=("a", "b"),
        )


def test_fsdp_track_reuses_the_existing_verifiable_reward_unchanged() -> None:
    rewards = gsm8k_exact_match_reward(
        completions=[
            "Reasoning.\n#### 12",
            "Reasoning.\n#### 12\nHuman: next question",
            "Reasoning.\n#### 9",
            "Reasoning.\n#### 9\nProblem: next question",
        ],
        answer=["work #### 12"] * 4,
    )

    assert rewards == [1.0, 0.75, 0.0, -0.25]
