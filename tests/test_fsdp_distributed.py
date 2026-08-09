from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from train.fsdp_utils import (
    FSDPSettings,
    apply_qwen_activation_checkpointing,
    clip_global_grad_norm_,
    fsdp_modules,
    rollout_parameter_state,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values).relu()


class QwenShapedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 4)
        self.model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.config = SimpleNamespace(use_cache=True)


def test_disabled_activation_checkpointing_preserves_block_identity() -> None:
    model = QwenShapedModel()
    original_blocks = tuple(model.model.layers)

    wrapped = apply_qwen_activation_checkpointing(model, enabled=False)

    assert wrapped == []
    assert tuple(model.model.layers) == original_blocks
    assert model.config.use_cache is False


def test_enabled_activation_checkpointing_wraps_each_block_and_preserves_forward() -> None:
    model = QwenShapedModel()
    original_types = tuple(type(block) for block in model.model.layers)
    sample = torch.randn(2, 4, requires_grad=True)

    wrapped = apply_qwen_activation_checkpointing(model, enabled=True)
    output = model.model.layers[1](model.model.layers[0](sample))
    output.sum().backward()

    assert wrapped == ["layer.0", "layer.1"]
    assert tuple(type(block) for block in model.model.layers) != original_types
    assert sample.grad is not None
    assert model.config.use_cache is False


def test_global_clipping_returns_preclip_norm_and_modifies_real_gradients() -> None:
    model = nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]])

    norm = clip_global_grad_norm_(model, max_norm=1.0)

    assert norm.item() == pytest.approx(5.0)
    assert torch.linalg.vector_norm(model.weight.grad).item() == pytest.approx(1.0)


def test_unsharded_model_has_no_fsdp_groups() -> None:
    assert fsdp_modules(QwenShapedModel()) == []


def test_rollout_state_rejects_unknown_mode_before_mutating_model() -> None:
    model = QwenShapedModel()

    with pytest.raises(ValueError, match="keep_unsharded.*reshard"):
        with rollout_parameter_state(model, "replicate"):
            pass

    assert fsdp_modules(model) == []


def test_fsdp_settings_reject_non_bfloat16_compute() -> None:
    with pytest.raises(ValueError, match="param_dtype.*bfloat16"):
        FSDPSettings(param_dtype=torch.float32)


@pytest.mark.cuda
@pytest.mark.distributed
def test_qwen_groups_are_explicit_and_parameters_are_dtensors(torchrun_result) -> None:
    payload = torchrun_result("tests/workers/fsdp_layout_worker.py", nproc=2)

    assert payload["groups"] == [
        "embed_tokens",
        "layer.0",
        "layer.1",
        "lm_head",
        "root",
    ]
    assert payload["all_parameters_dtensor_after_forward"] is True
    assert payload["resident_numel_sum"] == payload["global_numel"]
    assert payload["clip_norm_matches_global_shards"] is True
    assert payload["clip_norm_equal_on_all_ranks"] is True
    assert payload["rollout_parameters_are_unsharded_inside_context"] is True
    assert payload["rollout_parameters_are_resharded_after_context"] is True
    assert payload["process_group_destroyed"] is True
