from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from train.grpo_core import RolloutBatch, grpo_loss_sum, select_token_logps
from train.fsdp_grpo import (
    build_memory_layout_record,
    grpo_optimizer_step,
    maybe_build_reference,
    resolve_policy_source,
    tensor_moments,
)


class TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def _rollout_for(model: nn.Module) -> RolloutBatch:
    prompt_ids = torch.tensor([[2, 4], [2, 5]])
    prompt_mask = torch.ones_like(prompt_ids)
    completion_ids = torch.tensor([[6, 7], [8, 0]])
    completion_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        old_logps = select_token_logps(logits, input_ids, completion_length=2)
    return RolloutBatch(
        prompt_input_ids=prompt_ids,
        prompt_attention_mask=prompt_mask,
        completion_input_ids=completion_ids,
        completion_mask=completion_mask,
        old_logps=old_logps,
        rewards=torch.tensor([1.0, 0.0]),
        advantages=torch.tensor([1.0, -1.0]),
        ref_logps=None,
        completions=("#### 1", "#### 0"),
        answers=("work #### 1", "work #### 1"),
    )


def test_memory_record_names_policy_reference_and_rollout_locations() -> None:
    row = build_memory_layout_record(beta=0.0, rollout_mode="reshard")

    assert row["policy"] == "fsdp2_sharded_gpu"
    assert row["reference"] == "absent_beta_zero"
    assert row["rollout_records"] == "cpu_after_group"
    assert row["rollout_generation"] == "same_policy_layerwise_all_gather"


def test_memory_record_states_beta_positive_and_keep_unsharded_costs() -> None:
    row = build_memory_layout_record(beta=0.02, rollout_mode="keep_unsharded")

    assert row["reference"] == "independent_frozen_fsdp2_shard_gpu"
    assert row["rollout_generation"] == "same_policy_full_bf16_replica"
    assert row["peak_memory_implications"] == {
        "keep_unsharded": "one_full_bf16_policy_per_gpu_during_rollout",
        "beta_positive": "one_additional_frozen_bf16_reference_shard_per_gpu",
    }


def test_beta_zero_does_not_build_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        "train.fsdp_grpo.build_reference_model",
        Mock(side_effect=AssertionError("reference must stay absent")),
    )

    assert maybe_build_reference(beta=0.0, source="sft", ctx=object()) is None


def test_beta_positive_builds_a_separate_reference(monkeypatch) -> None:
    expected = object()
    builder = Mock(return_value=expected)
    monkeypatch.setattr("train.fsdp_grpo.build_reference_model", builder)

    actual = maybe_build_reference(beta=0.02, source="sft", ctx="mesh")

    assert actual is expected
    builder.assert_called_once_with(source="sft", ctx="mesh")


def test_sft_dcp_source_resolves_architecture_and_checkpoint_digest(tmp_path) -> None:
    base = tmp_path / "base-hf"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"base")
    run = tmp_path / "sft-run"
    checkpoint = run / "checkpoints" / "step-00000003"
    checkpoint.mkdir(parents=True)
    progress = {
        "global_step": 3,
        "consumed_tokens": 30,
        "sampler_state": {},
        "rng_states": [],
        "config": {"stage": "sft"},
        "source_digests": {"model": "base", "data": "fixed"},
    }
    (checkpoint / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "format_version": 1,
                "global_step": 3,
                "world_size": 2,
                "progress": progress,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (checkpoint / "_SUCCESS").write_text("step=3\n", encoding="utf-8")
    (run / "run_config.json").write_text(
        __import__("json").dumps({"model": str(base), "resolved_revision": None}),
        encoding="utf-8",
    )

    source = resolve_policy_source(checkpoint, revision=None)

    assert source.kind == "dcp"
    assert source.weights_path == checkpoint
    assert source.architecture_path == base.resolve()
    assert len(source.model_revision) == 64
    assert len(source.checkpoint_digest) == 64


def test_tensor_moments_stay_on_the_input_device_without_scalar_round_trips() -> None:
    values = torch.tensor([1.0, 2.0, 3.0])

    moments = tensor_moments(values)

    assert moments.tolist() == [6.0, 14.0, 3.0]
    assert moments.device == values.device


def test_grpo_step_matches_one_global_fixed_length_denominator() -> None:
    torch.manual_seed(7)
    model = TinyCausalLM()
    reference = copy.deepcopy(model)
    rollout = _rollout_for(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    input_ids = torch.cat(
        (rollout.prompt_input_ids, rollout.completion_input_ids),
        dim=1,
    )
    current_logps = select_token_logps(
        reference(input_ids=input_ids, use_cache=False).logits,
        input_ids,
        completion_length=2,
    )
    expected = grpo_loss_sum(
        current_logps=current_logps,
        old_logps=rollout.old_logps,
        ref_logps=None,
        advantages=rollout.advantages,
        completion_mask=rollout.completion_mask,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.0,
        max_completion_length=2,
    )
    (expected.loss_sum / 4).backward()
    reference_optimizer.step()

    metrics = grpo_optimizer_step(
        model,
        optimizer,
        scheduler,
        [rollout],
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        policy_microbatch_size=1,
        max_grad_norm=100.0,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.0,
        max_completion_length=2,
        accumulation_sync="reduce_scatter",
    )

    assert metrics.global_completion_count == 2
    assert metrics.normalizer == 4
    assert metrics.global_valid_tokens == 3
    assert metrics.loss == pytest.approx(expected.loss_sum.item() / 4)
    assert all(parameter.grad is None for parameter in model.parameters())
    for actual, wanted in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, wanted, atol=1e-7, rtol=1e-6)


def test_grpo_step_records_policy_and_optimizer_phase_memory() -> None:
    model = TinyCausalLM()
    rollout = _rollout_for(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    metrics = grpo_optimizer_step(
        model,
        optimizer,
        scheduler,
        [rollout],
        ctx=SimpleNamespace(world_size=1, device=torch.device("cpu")),
        policy_microbatch_size=2,
        max_grad_norm=10.0,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.0,
        max_completion_length=2,
        accumulation_sync="reduce_scatter",
    )

    assert metrics.policy_phase.elapsed_seconds > 0
    assert metrics.optimizer_phase.elapsed_seconds > 0
    assert metrics.policy_phase.peak_allocated_bytes == 0
    assert metrics.optimizer_phase.peak_reserved_bytes == 0
