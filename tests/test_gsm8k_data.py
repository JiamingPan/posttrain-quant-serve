from __future__ import annotations

import pytest
import torch

from scripts.gsm8k_reward import build_gsm8k_chat_text
from train.gsm8k_data import (
    CheckpointableDistributedSampler,
    SFTFeature,
    build_grpo_prompt,
    build_sft_feature,
    collate_token_batches,
    gsm8k_user_text,
    pack_sft_features,
)


class FakeQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    @staticmethod
    def _text_ids(text: str) -> list[int]:
        return [20 + (ord(character) % 71) for character in text]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        if not tokenize:
            rendered = "".join(
                f"{message['role']}: {message['content']} <eos> " for message in messages
            )
            if add_generation_prompt:
                rendered += "assistant: "
            return rendered

        token_ids: list[int] = []
        for message in messages:
            role_id = 10 if message["role"] == "user" else 11
            token_ids.extend([role_id, *self._text_ids(message["content"]), self.eos_token_id])
        if add_generation_prompt:
            token_ids.append(11)
        return token_ids


class TransformersV5Tokenizer(FakeQwenTokenizer):
    """Mirror the BatchEncoding return adopted by Transformers v5."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = True,
    ):
        result = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if not tokenize:
            return result
        return {
            "input_ids": result,
            "attention_mask": [1] * len(result),
        }


def test_only_assistant_suffix_is_supervised() -> None:
    feature = build_sft_feature(
        FakeQwenTokenizer(),
        "2+2?",
        "reasoning\n#### 4",
        max_length=256,
    )

    first_label = next(i for i, value in enumerate(feature.labels) if value != -100)
    assert feature.labels[:first_label] == [-100] * first_label
    assert feature.labels[first_label:] == feature.input_ids[first_label:]


def test_sft_feature_extracts_ids_from_transformers_v5_batch_encoding() -> None:
    feature = build_sft_feature(
        TransformersV5Tokenizer(),
        "2+2?",
        "#### 4",
        max_length=256,
    )

    assert feature.input_ids[0] == 10
    assert all(isinstance(token_id, int) for token_id in feature.input_ids)


def test_non_prefix_preserving_chat_template_is_rejected() -> None:
    class BrokenTokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
            result = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
            if tokenize and len(messages) > 1:
                result[0] = 99
            return result

    with pytest.raises(ValueError, match="not prefix-preserving"):
        build_sft_feature(BrokenTokenizer(), "2+2?", "#### 4", max_length=256)


def test_grpo_prompt_reuses_existing_reward_prompt() -> None:
    tokenizer = FakeQwenTokenizer()

    assert build_grpo_prompt(tokenizer, "2+2?") == build_gsm8k_chat_text(
        tokenizer,
        "2+2?",
    )


def test_sft_and_reward_paths_share_the_exact_gsm8k_instruction() -> None:
    assert gsm8k_user_text("2+2?") == (
        "Solve the math problem. Show the reasoning briefly. End with exactly one final line "
        "in the form #### <answer>, then stop. Do not write another problem or dialogue "
        "after the answer.\n\nProblem: 2+2?"
    )


def test_packing_is_fixed_length_and_preserves_masks() -> None:
    features = [
        SFTFeature(input_ids=[1, 2], labels=[-100, 2]),
        SFTFeature(input_ids=[3, 4, 5], labels=[-100, 4, 5]),
    ]

    packed = pack_sft_features(
        features,
        sequence_length=4,
        eos_token_id=99,
        pad_token_id=0,
        pad_final=True,
    )

    assert packed == [
        SFTFeature(input_ids=[1, 2, 99, 3], labels=[-100, 2, 99, -100]),
        SFTFeature(input_ids=[4, 5, 0, 0], labels=[4, 5, -100, -100]),
    ]


def test_packing_does_not_duplicate_an_existing_eos_and_can_drop_tail() -> None:
    packed = pack_sft_features(
        [
            SFTFeature(input_ids=[1, 99], labels=[-100, 99]),
            SFTFeature(input_ids=[2, 3, 4], labels=[-100, 3, 4]),
        ],
        sequence_length=4,
        eos_token_id=99,
        pad_token_id=0,
        pad_final=False,
    )

    assert packed == [SFTFeature(input_ids=[1, 99, 2, 3], labels=[-100, 99, -100, 3])]


def test_collation_builds_attention_mask_and_counts_supervised_tokens() -> None:
    batch = collate_token_batches(
        [
            SFTFeature(input_ids=[1, 2], labels=[-100, 2]),
            SFTFeature(input_ids=[3], labels=[3]),
        ],
        pad_token_id=0,
        pad_to_length=3,
    )

    assert torch.equal(batch.input_ids, torch.tensor([[1, 2, 0], [3, 0, 0]]))
    assert torch.equal(batch.labels, torch.tensor([[-100, 2, -100], [3, -100, -100]]))
    assert torch.equal(batch.attention_mask, torch.tensor([[1, 1, 0], [1, 0, 0]]))
    assert batch.supervised_tokens == 2


def test_rank_slices_are_disjoint_and_come_from_one_global_permutation() -> None:
    rank_zero = CheckpointableDistributedSampler(16, rank=0, world_size=2, seed=7)
    rank_one = CheckpointableDistributedSampler(16, rank=1, world_size=2, seed=7)

    zero_indices = rank_zero.next_indices(3)
    one_indices = rank_one.next_indices(3)

    assert set(zero_indices).isdisjoint(one_indices)
    assert zero_indices + one_indices == rank_zero.global_permutation[:6]


def test_sampler_rolls_to_a_new_epoch_without_returning_a_partial_batch() -> None:
    sampler = CheckpointableDistributedSampler(
        5,
        rank=0,
        world_size=2,
        seed=11,
        shuffle=False,
    )

    assert sampler.next_indices(2) == [0, 1]
    assert sampler.next_indices(2) == [0, 1]
    assert sampler.epoch == 1
    assert sampler.global_cursor == 4


def test_sampler_can_fill_a_full_batch_across_an_epoch_boundary() -> None:
    rank_zero = CheckpointableDistributedSampler(
        5,
        rank=0,
        world_size=2,
        seed=11,
        shuffle=False,
        drop_last=False,
    )
    rank_one = CheckpointableDistributedSampler(
        5,
        rank=1,
        world_size=2,
        seed=11,
        shuffle=False,
        drop_last=False,
    )

    assert rank_zero.next_indices(2) == [0, 1]
    assert rank_one.next_indices(2) == [2, 3]
    assert rank_zero.next_indices(2) == [4, 0]
    assert rank_one.next_indices(2) == [1, 2]
    assert rank_zero.epoch == rank_one.epoch == 1
    assert rank_zero.global_cursor == rank_one.global_cursor == 3


def test_sampler_resume_returns_exact_next_local_indices() -> None:
    sampler = CheckpointableDistributedSampler(16, rank=1, world_size=2, seed=7, shuffle=True)
    first = [sampler.next_indices(2) for _ in range(3)]
    state = sampler.state_dict()
    expected = sampler.next_indices(2)
    resumed = CheckpointableDistributedSampler(16, rank=1, world_size=2, seed=7, shuffle=True)
    resumed.load_state_dict(state)

    assert resumed.next_indices(2) == expected
    assert len(first) == 3


def test_sampler_checkpoint_state_is_identical_on_every_rank() -> None:
    rank_zero = CheckpointableDistributedSampler(16, rank=0, world_size=2, seed=7)
    rank_one = CheckpointableDistributedSampler(16, rank=1, world_size=2, seed=7)
    rank_zero.next_indices(2)
    rank_one.next_indices(2)

    assert rank_zero.state_dict() == rank_one.state_dict()

    restored_rank_one = CheckpointableDistributedSampler(
        16,
        rank=1,
        world_size=2,
        seed=7,
    )
    restored_rank_one.load_state_dict(rank_zero.state_dict())
    assert restored_rank_one.next_indices(2) == rank_one.next_indices(2)


def test_sampler_can_preserve_global_cursor_when_world_size_changes() -> None:
    original = CheckpointableDistributedSampler(
        16,
        rank=0,
        world_size=2,
        seed=7,
        shuffle=False,
    )
    original.next_indices(2)
    resumed = CheckpointableDistributedSampler(
        16,
        rank=0,
        world_size=1,
        seed=7,
        shuffle=False,
    )

    resumed.load_state_dict(original.state_dict(), allow_world_size_change=True)

    assert resumed.next_indices(2) == [4, 5]
