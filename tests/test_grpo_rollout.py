from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train.grpo_core import (
    RolloutConfig,
    choose_rollout_mode,
    generate_rollout_batch,
    teacher_forced_logps,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 4
        assert self.padding_side == "left"
        rows = [[2, 10], [2, 11, 12]][: len(texts)]
        width = max(map(len, rows))
        ids = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }

    def batch_decode(self, token_ids, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return ["#### 4" if int(row[0]) == 20 else "#### 9" for row in token_ids]


class TinyGeneratingPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.generate_calls = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        vocab_size = 32
        logits = torch.arange(
            vocab_size,
            device=input_ids.device,
            dtype=torch.float32,
        ).view(1, 1, vocab_size)
        logits = logits.expand(input_ids.size(0), input_ids.size(1), -1) * self.scale
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        self.generate_calls += 1
        assert self.training is False
        assert kwargs["max_new_tokens"] == 3
        assert kwargs["temperature"] == 1.0
        assert kwargs["synced_gpus"] is False
        completions = torch.tensor(
            [[20 if index % 2 == 0 else 21, 3, 0] for index in range(input_ids.size(0))],
            device=input_ids.device,
        )
        return torch.cat((input_ids, completions), dim=1)


def test_auto_rollout_mode_falls_back_when_full_policy_will_not_fit() -> None:
    assert choose_rollout_mode(
        predicted_keep_unsharded_gib=42.0,
        capacity_gib=40.0,
        requested="auto",
    ) == "reshard"
    assert choose_rollout_mode(
        predicted_keep_unsharded_gib=32.0,
        capacity_gib=40.0,
        requested="auto",
    ) == "keep_unsharded"


def test_explicit_keep_unsharded_refuses_an_unsafe_peak() -> None:
    with pytest.raises(MemoryError, match="42.00 GiB.*40.00 GiB"):
        choose_rollout_mode(
            predicted_keep_unsharded_gib=42.0,
            capacity_gib=40.0,
            requested="keep_unsharded",
        )


def test_generation_left_pads_repeats_and_moves_completed_rollout_to_cpu() -> None:
    policy = TinyGeneratingPolicy()
    tokenizer = TinyTokenizer()
    config = RolloutConfig(
        num_generations=2,
        max_prompt_length=4,
        max_completion_length=3,
        rollout_mode="reshard",
    )

    batch = generate_rollout_batch(
        policy,
        tokenizer,
        prompt_texts=["first", "second"],
        answers=["work #### 4", "work #### 4"],
        config=config,
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
    )

    assert tokenizer.padding_side == "right"
    assert policy.generate_calls == 1
    assert batch.prompt_input_ids.tolist() == [
        [0, 2, 10],
        [0, 2, 10],
        [2, 11, 12],
        [2, 11, 12],
    ]
    assert batch.completion_mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
    ]
    assert batch.rewards.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert batch.advantages.tolist() == [0.5, -0.5, 0.5, -0.5]
    assert batch.old_logps.device.type == "cpu"
    assert batch.ref_logps is None
    assert batch.rollout_mode == "reshard"
    assert policy.training is True
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_rollout_applies_the_configured_reward_scaling() -> None:
    batch = generate_rollout_batch(
        TinyGeneratingPolicy(),
        TinyTokenizer(),
        prompt_texts=["first"],
        answers=["work #### 4"],
        config=RolloutConfig(
            num_generations=2,
            max_prompt_length=4,
            max_completion_length=3,
            rollout_mode="reshard",
            scale_rewards="group",
        ),
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
    )

    assert batch.advantages.tolist() == [1.0, -1.0]


def test_teacher_forced_reference_scoring_is_inference_only_and_cpu_backed() -> None:
    reference = TinyGeneratingPolicy()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    result = teacher_forced_logps(
        reference,
        prompt_input_ids=torch.tensor([[2, 10], [2, 11]]),
        prompt_attention_mask=torch.ones(2, 2, dtype=torch.long),
        completion_input_ids=torch.tensor([[20, 3, 0], [21, 3, 0]]),
        completion_mask=torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool),
        device=torch.device("cpu"),
        microbatch_size=1,
    )

    assert result.shape == (2, 3)
    assert result.device.type == "cpu"
    assert result.requires_grad is False
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_reference_scores_are_attached_only_after_policy_rollout_finishes() -> None:
    policy = TinyGeneratingPolicy()
    reference = TinyGeneratingPolicy()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    batch = generate_rollout_batch(
        policy,
        TinyTokenizer(),
        prompt_texts=["first"],
        answers=["work #### 4"],
        config=RolloutConfig(
            num_generations=2,
            max_prompt_length=4,
            max_completion_length=3,
            rollout_mode="reshard",
        ),
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        reference=reference,
    )

    assert batch.ref_logps is not None
    assert batch.ref_logps.device.type == "cpu"
    assert policy.generate_calls == 1
    assert reference.generate_calls == 0


@pytest.mark.cuda
@pytest.mark.distributed
def test_keep_unsharded_rollout_reshards_in_finally(torchrun_result) -> None:
    row = torchrun_result("tests/workers/grpo_rollout_worker.py", nproc=2)

    assert row["all_groups_sharded_after_success"] is True
    assert row["all_groups_sharded_after_forced_error"] is True
    assert row["rollout_cpu_backed"] is True
    assert row["same_collective_schedule"] is True
