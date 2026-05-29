"""GRPO smoke training on GSM8K with a rule-based final-answer reward."""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import Any

from datasets import load_dataset
from gsm8k_reward import gsm8k_exact_match_reward
from trl import GRPOConfig, GRPOTrainer


def format_prompt(question: str) -> str:
    return (
        "Solve the math problem. Show the reasoning briefly. End with exactly one final line "
        "in the form #### <answer>, then stop. Do not write another problem or dialogue "
        "after the answer.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def build_dataset(split: str, limit: int | None):
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(
        lambda row: {"prompt": format_prompt(row["question"]), "answer": row["answer"]},
        remove_columns=[col for col in dataset.column_names if col not in {"answer"}],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_limit", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", default="none")

    # Optional TRL GRPOConfig knobs. Defaults are None unless the earlier smoke config already used
    # the field, so current behavior is preserved unless a flag/environment override is passed.
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--scale_rewards", choices=("group", "batch", "none"), default=None)
    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--mask_truncated_completions", action="store_true")
    return parser.parse_args()


def normalize_scale_rewards(value: str | None) -> str | bool | None:
    if value == "none":
        return False
    return value


def add_optional_config(kwargs: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        kwargs[key] = value


def build_grpo_config(**kwargs: Any) -> GRPOConfig:
    """Build GRPOConfig while tolerating small TRL API differences."""
    valid_fields = {field.name for field in fields(GRPOConfig)}
    filtered = {key: value for key, value in kwargs.items() if key in valid_fields}
    dropped = sorted(set(kwargs) - valid_fields)
    if dropped:
        print(f"Dropping unsupported GRPOConfig fields for this TRL version: {dropped}")
    interesting = [
        "beta",
        "num_generations",
        "temperature",
        "scale_rewards",
        "loss_type",
        "mask_truncated_completions",
        "epsilon_high",
    ]
    print("Accepted GRPOConfig diagnostic/optimization fields:")
    for key in interesting:
        if key in filtered:
            print(f"  {key}={filtered[key]!r}")
        elif key in valid_fields:
            print(f"  {key}=<TRL default>")
        else:
            print(f"  {key}=<unsupported by installed TRL>")
    return GRPOConfig(**filtered)


def main() -> None:
    args = parse_args()
    train_dataset = build_dataset(args.split, args.dataset_limit)

    config_kwargs: dict[str, Any] = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=2,
        report_to=args.report_to,
        log_on_each_node=False,
    )
    add_optional_config(config_kwargs, "temperature", args.temperature)
    add_optional_config(config_kwargs, "top_p", args.top_p)
    add_optional_config(config_kwargs, "top_k", args.top_k)
    add_optional_config(config_kwargs, "min_p", args.min_p)
    add_optional_config(config_kwargs, "repetition_penalty", args.repetition_penalty)
    add_optional_config(config_kwargs, "scale_rewards", normalize_scale_rewards(args.scale_rewards))
    add_optional_config(config_kwargs, "loss_type", args.loss_type)
    add_optional_config(config_kwargs, "epsilon_high", args.epsilon_high)
    if args.mask_truncated_completions:
        config_kwargs["mask_truncated_completions"] = True

    training_args = build_grpo_config(**config_kwargs)

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=gsm8k_exact_match_reward,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
