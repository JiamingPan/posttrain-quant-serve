"""Pinned Dr. GRPO math and CPU-backed rollout records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


RewardScaling = Literal["none", "group", "batch"]


@dataclass(frozen=True)
class RolloutBatch:
    """One completed rollout group, kept on CPU between GRPO phases."""

    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_input_ids: torch.Tensor
    completion_mask: torch.Tensor
    old_logps: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    ref_logps: torch.Tensor | None
    completions: tuple[str, ...]
    answers: tuple[str, ...]

    def __post_init__(self) -> None:
        tensors = {
            "prompt_input_ids": self.prompt_input_ids,
            "prompt_attention_mask": self.prompt_attention_mask,
            "completion_input_ids": self.completion_input_ids,
            "completion_mask": self.completion_mask,
            "old_logps": self.old_logps,
            "rewards": self.rewards,
            "advantages": self.advantages,
        }
        if self.ref_logps is not None:
            tensors["ref_logps"] = self.ref_logps
        non_cpu = [name for name, value in tensors.items() if value.device.type != "cpu"]
        if non_cpu:
            raise ValueError(
                "completed rollout tensors must be CPU-backed; found "
                + ", ".join(non_cpu)
            )
        matrix_names = (
            "prompt_input_ids",
            "prompt_attention_mask",
            "completion_input_ids",
            "completion_mask",
            "old_logps",
        )
        if any(tensors[name].ndim != 2 for name in matrix_names):
            raise ValueError("prompt, completion, mask, and log-probability tensors must be 2D")
        if self.rewards.ndim != 1 or self.advantages.ndim != 1:
            raise ValueError("rewards and advantages must be 1D")

        batch_size = self.completion_input_ids.size(0)
        batch_values = {
            name: value.size(0)
            for name, value in tensors.items()
        }
        batch_values.update(
            completions=len(self.completions),
            answers=len(self.answers),
        )
        if any(value != batch_size for value in batch_values.values()):
            raise ValueError(f"rollout batch dimension mismatch: {batch_values}")
        if self.prompt_input_ids.shape != self.prompt_attention_mask.shape:
            raise ValueError("prompt IDs and attention mask shapes must match")
        completion_shape = self.completion_input_ids.shape
        if self.completion_mask.shape != completion_shape:
            raise ValueError("completion IDs and completion mask shapes must match")
        if self.old_logps.shape != completion_shape:
            raise ValueError("old log-probability and completion shapes must match")
        if self.ref_logps is not None and self.ref_logps.shape != completion_shape:
            raise ValueError("reference log-probability and completion shapes must match")

    @property
    def batch_size(self) -> int:
        return self.completion_input_ids.size(0)

    @property
    def max_completion_length(self) -> int:
        return self.completion_input_ids.size(1)


@dataclass(frozen=True)
class GRPOMetrics:
    """Differentiable objective sums plus detached accounting values."""

    loss_sum: torch.Tensor
    normalized_loss: torch.Tensor
    policy_loss_sum: torch.Tensor
    kl_sum: torch.Tensor
    clip_ratio: torch.Tensor
    valid_tokens: int
    completion_count: int
    normalizer: int


def group_advantages(
    rewards: torch.Tensor,
    *,
    scale_rewards: RewardScaling = "none",
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Center rewards per prompt and optionally standardize explicitly."""

    if rewards.ndim != 2:
        raise ValueError("grouped rewards must have shape [prompts, generations]")
    if rewards.size(1) < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if not rewards.is_floating_point():
        raise TypeError("rewards must use a floating-point dtype")
    if scale_rewards not in {"none", "group", "batch"}:
        raise ValueError("scale_rewards must be 'none', 'group', or 'batch'")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if scale_rewards == "none":
        return centered
    if scale_rewards == "group":
        denominator = rewards.std(dim=1, keepdim=True, unbiased=False)
    else:
        denominator = rewards.std(unbiased=False)
    return centered / denominator.clamp_min(epsilon)


def select_token_logps(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    completion_length: int,
) -> torch.Tensor:
    """Select causal next-token log probabilities for a fixed suffix."""

    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("logits and token IDs must be [batch, sequence, vocabulary] and [batch, sequence]")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits and token IDs must share batch and sequence dimensions")
    if completion_length <= 0 or completion_length > token_ids.size(1) - 1:
        raise ValueError("completion_length must select a non-empty causal suffix")

    next_token_logps = logits[:, :-1].float().log_softmax(dim=-1).gather(
        dim=-1,
        index=token_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    return next_token_logps[:, -completion_length:]


def _validate_grpo_inputs(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor | None,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float,
    beta: float,
    max_completion_length: int,
) -> None:
    if current_logps.ndim != 2:
        raise ValueError("completion log probabilities must be 2D")
    if old_logps.shape != current_logps.shape:
        raise ValueError("current and old log-probability shapes must match")
    if completion_mask.shape != current_logps.shape:
        raise ValueError("completion mask and log-probability shapes must match")
    if advantages.shape != (current_logps.size(0),):
        raise ValueError("advantages must have one value per completion")
    if ref_logps is not None and ref_logps.shape != current_logps.shape:
        raise ValueError("reference and current log-probability shapes must match")
    if beta > 0 and ref_logps is None:
        raise ValueError("beta-positive GRPO requires reference log probabilities")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if not 0 <= epsilon_low < 1 or epsilon_high < 0:
        raise ValueError("clipping bounds must satisfy 0 <= epsilon_low < 1 and epsilon_high >= 0")
    if max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")
    if current_logps.size(1) > max_completion_length:
        raise ValueError("log-probability width exceeds max_completion_length")
    if not current_logps.is_floating_point() or not old_logps.is_floating_point():
        raise TypeError("log probabilities must use floating-point dtypes")


def grpo_loss_sum(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float | None = None,
    beta: float,
    max_completion_length: int,
    ref_logps: torch.Tensor | None = None,
) -> GRPOMetrics:
    """Compute the masked clipped Dr. GRPO objective without rank averaging."""

    resolved_epsilon_high = epsilon_low if epsilon_high is None else epsilon_high
    _validate_grpo_inputs(
        current_logps,
        old_logps,
        ref_logps,
        advantages,
        completion_mask,
        epsilon_low=epsilon_low,
        epsilon_high=resolved_epsilon_high,
        beta=beta,
        max_completion_length=max_completion_length,
    )
    mask = completion_mask.to(dtype=current_logps.dtype)
    token_advantages = advantages.to(dtype=current_logps.dtype).unsqueeze(1)
    log_ratio = current_logps - old_logps.detach()
    ratio = log_ratio.exp()
    unclipped = ratio * token_advantages
    clipped_ratio = ratio.clamp(
        min=1.0 - epsilon_low,
        max=1.0 + resolved_epsilon_high,
    )
    clipped = clipped_ratio * token_advantages
    per_token_policy_loss = -torch.minimum(unclipped, clipped)
    policy_loss_sum = (per_token_policy_loss * mask).sum()

    if beta == 0:
        kl_sum = current_logps.new_zeros(())
        loss_sum = policy_loss_sum
    else:
        assert ref_logps is not None
        delta = ref_logps.detach() - current_logps
        per_token_kl = delta.exp() - delta - 1.0
        kl_sum = (per_token_kl * mask).sum()
        loss_sum = policy_loss_sum + beta * kl_sum

    valid_tokens = int(completion_mask.count_nonzero().item())
    if valid_tokens == 0:
        raise ValueError("completion mask contains no valid tokens")
    completion_count = current_logps.size(0)
    normalizer = completion_count * max_completion_length
    outside_clip = (ratio < 1.0 - epsilon_low) | (
        ratio > 1.0 + resolved_epsilon_high
    )
    clip_ratio = (outside_clip.to(mask.dtype) * mask).sum() / valid_tokens
    return GRPOMetrics(
        loss_sum=loss_sum,
        normalized_loss=loss_sum / normalizer,
        policy_loss_sum=policy_loss_sum,
        kl_sum=kl_sum,
        clip_ratio=clip_ratio,
        valid_tokens=valid_tokens,
        completion_count=completion_count,
        normalizer=normalizer,
    )
