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

After the 100-step smoke run, prompt leakage was added as a small penalty:

```text
reward -= 0.25 if prompt-like text appears after the parsed answer
```

Examples:

- clean correct answer: `1.0`
- leaky correct answer: `0.75`
- clean wrong answer: `0.0`
- leaky wrong answer: `-0.25`

This keeps answer correctness as the main signal while teaching the model that continuing into
`Human:`, `Problem:`, `Question:`, or similar prompt-like text after the answer is worse than
stopping cleanly.

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


## Day 3 GRPO Stability Notes

The Day 2 regression is more consistent with GRPO signal collapse than simple undertraining:

```text
100-step eval: base 0.60 -> trained 0.50
300-step leak-penalty eval: base 0.60 -> trained 0.20
300-step reference PPL: 1.878 -> 1.934
final train_loss: about 7.7e-6
```

The mechanism to check first is within-group reward collapse. If all completions for a prompt get
identical reward, GRPO has no useful relative advantage for that prompt. With `beta > 0`, the KL term
can still be active even when the policy-gradient term is effectively dead. The notebook now plots
reward standard deviation, zero-reward-variance group fraction, advantage magnitude, and KL when the
trainer saved those series.

## TRL 1.5.0 Knobs To Know

The training script keeps the old defaults unless an override is passed. It exposes these optional
TRL `GRPOConfig` fields when the installed TRL version supports them:

| Flag | Purpose | Default in this repo |
| --- | --- | --- |
| `--num_generations` | group size; larger groups are more likely to contain reward contrast | `4` |
| `--temperature` | sampling exploration during rollout generation | TRL default unless set |
| `--top_p`, `--top_k`, `--min_p`, `--repetition_penalty` | generation exploration/shape controls | TRL default unless set |
| `--scale_rewards` | TRL reward scaling mode; use `group`, `batch`, or `none` | TRL default unless set |
| `--loss_type` | GRPO-family loss variant, e.g. `grpo`, `bnpo`, `dr_grpo`, or `dapo` when supported | TRL default unless set |
| `--epsilon_high` | DAPO-style higher clip range when supported | TRL default unless set |
| `--mask_truncated_completions` | ignore completions truncated by max length when supported | off unless set |
| `--beta` | KL coefficient; lower values weaken the pull toward the reference model | `0.02` |

Important limitation: this repo does not implement true DAPO dynamic sampling. Dropping or resampling
all-equal-reward prompt groups requires changing trainer-side sampling/loss construction. A reward
function can diagnose all-equal groups, but it cannot honestly remove their KL contribution after the
trainer has already sampled them. Use the notebook diagnostics before deciding whether to modify the
trainer itself.

## Interview Explanation

"I used GRPO because GSM8K gives verifiable rewards, so I could run RL post-training without training a reward model. For each math prompt, the trainer samples several completions, scores final-answer correctness, normalizes rewards within the group to estimate advantages, and applies a KL penalty against the reference model. The project then tests whether that RL update changes the model's quantization behavior."
