# GRPO Notes

GRPO is Group Relative Policy Optimization. It is an RL post-training method used for reasoning-style models when rewards can be computed from generated outputs.

## Mental Model

For each prompt:

1. Generate several completions from the current policy model.
2. Score each completion with a reward function.
3. Compare completions inside the same group.
4. Update the policy to make better completions more likely.
5. Penalize the policy if it drifts too far from the reference model.

The key savings versus PPO is that GRPO does not require a separate learned value model/critic. The group of sampled completions provides the baseline.

## GSM8K Reward

For Day 0, the reward is rule-based:

```text
reward = 1.0 if extracted final numeric answer equals the GSM8K target
reward = 0.0 otherwise
```

The prompt asks the model to end with:

```text
#### <answer>
```

The reward code still falls back to the last number in the completion if the format is missing.

The parser gives priority to explicit final-answer patterns before using that fallback:

- `#### 312`
- `<answer>312</answer>`
- `#### <answer>312</answer>`
- `Final answer: 8`
- `the answer is 312`
- `direct numeric answer ... 8`

This matters because Qwen2.5-0.5B-Instruct often emits a valid final number without following the
requested `#### <answer>` format exactly during early smoke runs.

## Metrics To Watch

- `reward`: should be noisy but not broken; on 10 examples it may not rise smoothly.
- `reward_std`: if always zero, all completions are scoring the same and GRPO has no useful signal.
- `kl`: if too high, the model is drifting too far from the reference.
- `completions/mean_length`: tells whether outputs are becoming too long or getting clipped.
- GPU peak memory: confirms whether the environment can support the next scale step.

Use `notebooks/grpo_smoke_analysis.ipynb` after a run to inspect artifacts, completions, parser
behavior, and Slurm logs.

## Batch Size Rule

TRL requires the effective batch size to be divisible by `num_generations`:

```text
num_processes * per_device_train_batch_size * gradient_accumulation_steps
```

Day 0 uses:

```text
1 process * batch 1 * grad_accum 4 = 4
num_generations = 4
```

So the smoke config satisfies the rule.

## Failure Modes

| Symptom | Likely Cause | Response |
| --- | --- | --- |
| CUDA unavailable in job | Slurm did not allocate GPU or module/env mismatch | Check `scripts/cluster_check.py` output first |
| Reward always zero | Prompt format bad, model not producing final number, or answer parser too strict | Inspect sample completions before changing training |
| Reward std always zero | All group completions get identical reward | Increase completions or improve reward signal |
| OOM on 0.5B | Environment/config error | Stop and investigate; do not shrink model further |
| Checkpoint cannot resume | Wrong checkpoint path or max_steps not greater than checkpoint step | Resume from `checkpoint-N` and set max steps above N |
| `GRPOConfig` unexpected keyword | TRL changed config field names across versions | `train_grpo_gsm8k.py` filters unsupported config keys and prints what it dropped |
| Correct-looking answer gets reward 0 | Answer parser is missing the model's format | Add a parser case to `scripts/check_reward_parser.py`, then patch `scripts/gsm8k_reward.py` |

## Interview Explanation

"I used GRPO because GSM8K gives verifiable rewards, so I could run RL post-training without training a reward model. For each math prompt, the trainer samples several completions, scores final-answer correctness, normalizes rewards within the group to estimate advantages, and applies a KL penalty against the reference model. The project then tests whether that RL update changes the model's quantization behavior."
