"""Pinned Dr. GRPO math and CPU-backed rollout records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import torch

from scripts.gsm8k_reward import gsm8k_exact_match_reward
from train.fsdp_utils import rollout_parameter_state


RewardScaling = Literal["none", "group", "batch"]
RequestedRolloutMode = Literal["auto", "reshard", "keep_unsharded"]
ResolvedRolloutMode = Literal["reshard", "keep_unsharded"]


class RolloutContext(Protocol):
    world_size: int
    device: torch.device


@dataclass(frozen=True)
class RolloutConfig:
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    scale_rewards: RewardScaling = "none"
    rollout_mode: RequestedRolloutMode = "auto"
    teacher_forcing_microbatch_size: int = 1

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise ValueError("num_generations must be at least two")
        if self.max_prompt_length <= 0 or self.max_completion_length <= 0:
            raise ValueError("prompt and completion lengths must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.scale_rewards not in {"none", "group", "batch"}:
            raise ValueError("invalid reward scaling mode")
        if self.teacher_forcing_microbatch_size <= 0:
            raise ValueError("teacher_forcing_microbatch_size must be positive")


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
    rollout_mode: ResolvedRolloutMode = "reshard"

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
        if self.rollout_mode not in {"reshard", "keep_unsharded"}:
            raise ValueError("rollout_mode must be resolved before storing a rollout")

    @property
    def batch_size(self) -> int:
        return self.completion_input_ids.size(0)

    @property
    def max_completion_length(self) -> int:
        return self.completion_input_ids.size(1)


def choose_rollout_mode(
    *,
    predicted_keep_unsharded_gib: float,
    capacity_gib: float,
    requested: RequestedRolloutMode,
) -> ResolvedRolloutMode:
    """Resolve the rollout communication/memory trade-off before generation."""

    if predicted_keep_unsharded_gib <= 0 or capacity_gib <= 0:
        raise ValueError("predicted peak and GPU capacity must be positive")
    if requested not in {"auto", "reshard", "keep_unsharded"}:
        raise ValueError("requested rollout mode is invalid")
    fits = predicted_keep_unsharded_gib <= capacity_gib
    if requested == "auto":
        return "keep_unsharded" if fits else "reshard"
    if requested == "keep_unsharded" and not fits:
        raise MemoryError(
            "keep_unsharded rollout predicts "
            f"{predicted_keep_unsharded_gib:.2f} GiB on a {capacity_gib:.2f} GiB GPU"
        )
    return requested


def teacher_forced_logps(
    model: torch.nn.Module,
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    completion_input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
) -> torch.Tensor:
    """Score fixed completions without gradients and return CPU log probabilities."""

    if prompt_input_ids.ndim != 2 or completion_input_ids.ndim != 2:
        raise ValueError("prompt and completion IDs must be 2D")
    if prompt_input_ids.shape != prompt_attention_mask.shape:
        raise ValueError("prompt IDs and attention mask shapes must match")
    if completion_input_ids.shape != completion_mask.shape:
        raise ValueError("completion IDs and mask shapes must match")
    if prompt_input_ids.size(0) != completion_input_ids.size(0):
        raise ValueError("prompt and completion batch sizes must match")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")

    was_training = model.training
    model.eval()
    rows: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, prompt_input_ids.size(0), microbatch_size):
                stop = min(start + microbatch_size, prompt_input_ids.size(0))
                prompt_ids = prompt_input_ids[start:stop].to(device)
                prompt_mask = prompt_attention_mask[start:stop].to(device)
                completion_ids = completion_input_ids[start:stop].to(device)
                valid_completion = completion_mask[start:stop].to(device)
                input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
                attention_mask = torch.cat(
                    (prompt_mask, valid_completion.to(prompt_mask.dtype)),
                    dim=1,
                )
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                logps = select_token_logps(
                    logits,
                    input_ids,
                    completion_length=completion_ids.size(1),
                )
                rows.append(logps.cpu())
    finally:
        model.train(was_training)
    return torch.cat(rows, dim=0)


def _completion_mask(
    completion_ids: torch.Tensor,
    *,
    produced_width: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> torch.Tensor:
    positions = torch.arange(
        completion_ids.size(1),
        device=completion_ids.device,
    )
    mask = positions.unsqueeze(0) < produced_width
    if eos_token_id is not None:
        is_eos = completion_ids.eq(eos_token_id)
        has_prior_eos = is_eos.cumsum(dim=1) - is_eos.to(torch.int64) > 0
        mask = mask & ~has_prior_eos
    if eos_token_id != pad_token_id:
        mask = mask & completion_ids.ne(pad_token_id)
    return mask


def _resolved_generation_mode(
    config: RolloutConfig,
    *,
    ctx: RolloutContext,
    predicted_keep_unsharded_gib: float | None,
    capacity_gib: float | None,
) -> ResolvedRolloutMode:
    if config.rollout_mode == "reshard":
        return "reshard"
    if predicted_keep_unsharded_gib is None:
        if config.rollout_mode == "auto":
            return "reshard"
        raise ValueError("keep_unsharded rollout requires a predicted peak")
    if capacity_gib is None:
        if ctx.device.type != "cuda":
            raise ValueError("keep_unsharded rollout requires a GPU capacity")
        capacity_gib = torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3
    return choose_rollout_mode(
        predicted_keep_unsharded_gib=predicted_keep_unsharded_gib,
        capacity_gib=capacity_gib,
        requested=config.rollout_mode,
    )


def generate_rollout_batch(
    policy: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt_texts: Sequence[str],
    answers: Sequence[str],
    config: RolloutConfig,
    ctx: RolloutContext,
    reference: torch.nn.Module | None = None,
    predicted_keep_unsharded_gib: float | None = None,
    capacity_gib: float | None = None,
) -> RolloutBatch:
    """Generate and score one synchronized prompt group using the sharded policy."""

    if not prompt_texts or len(prompt_texts) != len(answers):
        raise ValueError("prompt_texts and answers must be non-empty and equally sized")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("rollout tokenizer requires a pad token")
    mode = _resolved_generation_mode(
        config,
        ctx=ctx,
        predicted_keep_unsharded_gib=predicted_keep_unsharded_gib,
        capacity_gib=capacity_gib,
    )

    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(
            list(prompt_texts),
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    prompt_ids = encoded["input_ids"].repeat_interleave(
        config.num_generations,
        dim=0,
    )
    prompt_mask = encoded["attention_mask"].repeat_interleave(
        config.num_generations,
        dim=0,
    )
    repeated_answers = tuple(
        answer for answer in answers for _ in range(config.num_generations)
    )

    policy_was_training = policy.training
    policy.eval()
    try:
        with rollout_parameter_state(policy, mode):
            with torch.inference_mode():
                generated = policy.generate(
                    input_ids=prompt_ids.to(ctx.device),
                    attention_mask=prompt_mask.to(ctx.device),
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_new_tokens=config.max_completion_length,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    synced_gpus=ctx.world_size > 1,
                    use_cache=True,
                )
            prompt_width = prompt_ids.size(1)
            if generated.size(1) < prompt_width:
                raise RuntimeError("generation returned fewer tokens than the prompt width")
            completion_ids = generated[:, prompt_width:]
            produced_width = min(completion_ids.size(1), config.max_completion_length)
            completion_ids = completion_ids[:, : config.max_completion_length]
            if completion_ids.size(1) < config.max_completion_length:
                padding = torch.full(
                    (
                        completion_ids.size(0),
                        config.max_completion_length - completion_ids.size(1),
                    ),
                    pad_token_id,
                    dtype=completion_ids.dtype,
                    device=completion_ids.device,
                )
                completion_ids = torch.cat((completion_ids, padding), dim=1)
            completion_mask = _completion_mask(
                completion_ids,
                produced_width=produced_width,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )
            old_logps = teacher_forced_logps(
                policy,
                prompt_input_ids=prompt_ids,
                prompt_attention_mask=prompt_mask,
                completion_input_ids=completion_ids.cpu(),
                completion_mask=completion_mask.cpu(),
                device=ctx.device,
                microbatch_size=config.teacher_forcing_microbatch_size,
            )
    finally:
        policy.train(policy_was_training)

    completion_ids_cpu = completion_ids.cpu()
    completion_mask_cpu = completion_mask.cpu()
    completions = tuple(
        tokenizer.batch_decode(
            completion_ids_cpu,
            skip_special_tokens=True,
        )
    )
    rewards = torch.tensor(
        gsm8k_exact_match_reward(
            completions=list(completions),
            answer=list(repeated_answers),
        ),
        dtype=torch.float32,
    )
    advantages = group_advantages(
        rewards.view(len(prompt_texts), config.num_generations),
        scale_rewards=config.scale_rewards,
    ).reshape(-1)
    ref_logps = None
    if reference is not None:
        ref_logps = teacher_forced_logps(
            reference,
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            completion_input_ids=completion_ids_cpu,
            completion_mask=completion_mask_cpu,
            device=ctx.device,
            microbatch_size=config.teacher_forcing_microbatch_size,
        )
    return RolloutBatch(
        prompt_input_ids=prompt_ids.cpu(),
        prompt_attention_mask=prompt_mask.cpu(),
        completion_input_ids=completion_ids_cpu,
        completion_mask=completion_mask_cpu,
        old_logps=old_logps,
        rewards=rewards,
        advantages=advantages,
        ref_logps=ref_logps,
        completions=completions,
        answers=repeated_answers,
        rollout_mode=mode,
    )


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

    suffix_logits = logits[:, -(completion_length + 1) : -1]
    suffix_token_ids = token_ids[:, -completion_length:]
    return suffix_logits.float().log_softmax(dim=-1).gather(
        dim=-1,
        index=suffix_token_ids.unsqueeze(-1),
    ).squeeze(-1)


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
