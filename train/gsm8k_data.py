"""Shared deterministic GSM8K preparation for FSDP2 SFT and GRPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from scripts.gsm8k_reward import build_gsm8k_chat_text, gsm8k_user_text


IGNORE_INDEX = -100


@dataclass(frozen=True)
class SFTFeature:
    input_ids: list[int]
    labels: list[int]


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    supervised_tokens: int


def build_sft_feature(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    max_length: int,
) -> SFTFeature:
    """Tokenize one conversation and supervise only its assistant suffix."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    user = [{"role": "user", "content": gsm8k_user_text(question)}]
    conversation = [*user, {"role": "assistant", "content": answer}]
    prompt_ids = list(
        tokenizer.apply_chat_template(
            user,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = list(
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "The chat template is not prefix-preserving for assistant-only SFT labels"
        )

    input_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]
    return SFTFeature(input_ids=input_ids, labels=labels)


def build_grpo_prompt(tokenizer: Any, question: str) -> str:
    """Render the prompt through the existing verifiable-reward path."""

    return build_gsm8k_chat_text(tokenizer, question)


def _validate_feature(feature: SFTFeature) -> None:
    if not feature.input_ids:
        raise ValueError("SFT features must contain at least one token")
    if len(feature.input_ids) != len(feature.labels):
        raise ValueError("input_ids and labels must have the same length")


def pack_sft_features(
    features: Sequence[SFTFeature],
    *,
    sequence_length: int,
    eos_token_id: int,
    pad_token_id: int,
    pad_final: bool,
) -> list[SFTFeature]:
    """Pack examples in input order into deterministic fixed-length rows."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    stream_ids: list[int] = []
    stream_labels: list[int] = []
    for feature_index, feature in enumerate(features):
        _validate_feature(feature)
        stream_ids.extend(feature.input_ids)
        stream_labels.extend(feature.labels)
        if feature_index + 1 < len(features) and stream_ids[-1] != eos_token_id:
            stream_ids.append(eos_token_id)
            stream_labels.append(eos_token_id)

    packed: list[SFTFeature] = []
    full_length = len(stream_ids) - (len(stream_ids) % sequence_length)
    for start in range(0, full_length, sequence_length):
        stop = start + sequence_length
        packed.append(
            SFTFeature(
                input_ids=stream_ids[start:stop],
                labels=stream_labels[start:stop],
            )
        )

    if pad_final and full_length < len(stream_ids):
        remaining_ids = stream_ids[full_length:]
        remaining_labels = stream_labels[full_length:]
        padding = sequence_length - len(remaining_ids)
        packed.append(
            SFTFeature(
                input_ids=[*remaining_ids, *([pad_token_id] * padding)],
                labels=[*remaining_labels, *([IGNORE_INDEX] * padding)],
            )
        )
    return packed


def collate_token_batches(
    features: Sequence[SFTFeature],
    *,
    pad_token_id: int,
    pad_to_length: int | None = None,
) -> TokenBatch:
    """Right-pad local features and report the true loss-token count."""

    if not features:
        raise ValueError("cannot collate an empty feature list")
    for feature in features:
        _validate_feature(feature)
    target_length = pad_to_length or max(len(feature.input_ids) for feature in features)
    if target_length <= 0:
        raise ValueError("pad_to_length must be positive")
    if any(len(feature.input_ids) > target_length for feature in features):
        raise ValueError("pad_to_length is shorter than at least one feature")

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for feature in features:
        padding = target_length - len(feature.input_ids)
        input_rows.append([*feature.input_ids, *([pad_token_id] * padding)])
        label_rows.append([*feature.labels, *([IGNORE_INDEX] * padding)])
        mask_rows.append([*([1] * len(feature.input_ids)), *([0] * padding)])

    labels = torch.tensor(label_rows, dtype=torch.long)
    return TokenBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        labels=labels,
        attention_mask=torch.tensor(mask_rows, dtype=torch.long),
        supervised_tokens=int(labels.ne(IGNORE_INDEX).sum().item()),
    )


class CheckpointableDistributedSampler:
    """Slice one deterministic global permutation and persist its exact cursor."""

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset_size = dataset_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.global_cursor = 0

    @property
    def global_permutation(self) -> list[int]:
        return self._permutation_for_epoch(self.epoch)

    def _permutation_for_epoch(self, epoch: int) -> list[int]:
        if not self.shuffle:
            return list(range(self.dataset_size))
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(self.dataset_size, generator=generator).tolist()

    def next_indices(self, local_batch_size: int) -> list[int]:
        if local_batch_size <= 0:
            raise ValueError("local_batch_size must be positive")
        global_batch_size = local_batch_size * self.world_size
        if self.drop_last:
            if global_batch_size > self.dataset_size:
                raise ValueError("global batch is larger than the dataset with drop_last enabled")
            if self.global_cursor + global_batch_size > self.dataset_size:
                self.epoch += 1
                self.global_cursor = 0
            global_indices = self.global_permutation[
                self.global_cursor : self.global_cursor + global_batch_size
            ]
            self.global_cursor += global_batch_size
        else:
            global_indices = []
            while len(global_indices) < global_batch_size:
                if self.global_cursor == self.dataset_size:
                    self.epoch += 1
                    self.global_cursor = 0
                take = min(
                    global_batch_size - len(global_indices),
                    self.dataset_size - self.global_cursor,
                )
                global_indices.extend(
                    self.global_permutation[
                        self.global_cursor : self.global_cursor + take
                    ]
                )
                self.global_cursor += take

        local_start = self.rank * local_batch_size
        local_indices = global_indices[local_start : local_start + local_batch_size]
        return local_indices

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "dataset_size": self.dataset_size,
            "world_size": self.world_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "global_cursor": self.global_cursor,
        }

    def load_state_dict(
        self,
        state: Mapping[str, object],
        *,
        allow_world_size_change: bool = False,
    ) -> None:
        expected = {
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
        }
        if not allow_world_size_change:
            expected["world_size"] = self.world_size
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(f"sampler state does not match this sampler: {actual!r}")
        epoch = int(state["epoch"])
        global_cursor = int(state["global_cursor"])
        if epoch < 0 or not 0 <= global_cursor <= self.dataset_size:
            raise ValueError("sampler state has an invalid epoch or global cursor")
        self.epoch = epoch
        self.global_cursor = global_cursor
