from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from train.fsdp_sft import (
    assert_adamw_moment_dtype,
    backward_scale,
    prepare_step_batches,
    sft_optimizer_step,
)
from train.gsm8k_data import (
    CheckpointableDistributedSampler,
    SFTFeature,
    TokenBatch,
)


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


class SyncTrackingCausalLM(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.sync_states: list[bool] = []

    def set_requires_gradient_sync(self, enabled: bool) -> None:
        self.sync_states.append(enabled)


def _batch(input_ids: list[list[int]], labels: list[list[int]]) -> TokenBatch:
    input_tensor = torch.tensor(input_ids)
    label_tensor = torch.tensor(labels)
    return TokenBatch(
        input_ids=input_tensor,
        labels=label_tensor,
        attention_mask=torch.ones_like(input_tensor),
        supervised_tokens=int(label_tensor.ne(-100).sum().item()),
    )


def _loss_sum(model: nn.Module, batches: list[TokenBatch]) -> tuple[torch.Tensor, int]:
    loss_sum = torch.zeros(())
    token_count = 0
    for batch in batches:
        logits = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
        ).logits
        labels = batch.labels[:, 1:]
        loss_sum = loss_sum + F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        token_count += int(labels.ne(-100).sum().item())
    return loss_sum, token_count


def test_sft_backward_scale_uses_global_step_token_count() -> None:
    assert backward_scale(world_size=4, global_step_tokens=8192) == pytest.approx(
        4 / 8192
    )


def test_accumulation_uses_one_global_token_mean_not_mean_of_microbatch_means() -> None:
    torch.manual_seed(3)
    model = TinyCausalLM()
    reference = copy.deepcopy(model)
    batches = [
        _batch([[1, 2, 3, 4]], [[-100, 2, 3, 4]]),
        _batch([[5, 6, 7, 8]], [[-100, -100, -100, 8]]),
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    expected_loss_sum, expected_tokens = _loss_sum(reference, batches)
    (expected_loss_sum / expected_tokens).backward()
    reference_optimizer.step()
    metrics = sft_optimizer_step(
        model,
        optimizer,
        scheduler,
        batches,
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        max_grad_norm=100.0,
        accumulation_sync="reduce_scatter",
    )

    assert metrics.global_step_tokens == 4
    assert metrics.global_loss_sum == pytest.approx(expected_loss_sum.item())
    assert metrics.loss == pytest.approx(expected_loss_sum.item() / expected_tokens)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_no_sync_is_disabled_only_for_nonfinal_microbatches() -> None:
    model = SyncTrackingCausalLM()
    batches = [
        _batch([[1, 2, 3]], [[-100, 2, 3]]),
        _batch([[4, 5, 6]], [[-100, 5, 6]]),
        _batch([[7, 8, 9]], [[-100, 8, 9]]),
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    sft_optimizer_step(
        model,
        optimizer,
        scheduler,
        batches,
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        max_grad_norm=10.0,
        accumulation_sync="no_sync",
    )

    assert model.sync_states == [False, False, True, True]


def test_no_sync_rejects_a_plain_model() -> None:
    model = TinyCausalLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    with pytest.raises(TypeError, match="requires an FSDP2 model"):
        sft_optimizer_step(
            model,
            optimizer,
            scheduler,
            [_batch([[1, 2, 3]], [[-100, 2, 3]])],
            ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
            max_grad_norm=1.0,
            accumulation_sync="no_sync",
        )


def test_prepare_step_batches_consumes_exact_sampler_indices() -> None:
    features = [
        SFTFeature(input_ids=[index, index + 1], labels=[-100, index + 1])
        for index in range(8)
    ]
    sampler = CheckpointableDistributedSampler(
        8,
        rank=0,
        world_size=1,
        seed=9,
        shuffle=False,
    )

    batches = prepare_step_batches(
        features,
        sampler,
        local_microbatch_size=2,
        gradient_accumulation_steps=2,
        pad_token_id=0,
    )

    assert [batch.input_ids[:, 0].tolist() for batch in batches] == [[0, 1], [2, 3]]
    assert sampler.global_cursor == 4


def test_adamw_moment_dtype_check_rejects_an_unexpected_state_dtype() -> None:
    model = nn.Linear(2, 1).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()

    assert_adamw_moment_dtype(optimizer, torch.float32)
    with pytest.raises(RuntimeError, match="exp_avg.*torch.float32.*torch.bfloat16"):
        assert_adamw_moment_dtype(optimizer, torch.bfloat16)


@pytest.mark.cuda
@pytest.mark.distributed
def test_two_rank_step_uses_global_token_count_and_bf16_adamw(torchrun_result) -> None:
    row = torchrun_result("tests/workers/sft_step_worker.py", nproc=2)

    assert row["reported_loss_matches_independent_global_mean"] is True
    assert row["metrics_equal_on_all_ranks"] is True
    assert row["adamw_moments_bfloat16"] is True
