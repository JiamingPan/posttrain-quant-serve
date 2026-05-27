"""GRPO smoke training on GSM8K with a rule-based final-answer reward."""

from __future__ import annotations

import argparse
from dataclasses import fields
import re
from typing import Any

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace(",", "").strip()


def extract_reference_answer(answer: str) -> str | None:
    match = FINAL_ANSWER_RE.search(answer)
    if match:
        return normalize_number(match.group(1))
    numbers = NUMBER_RE.findall(answer)
    return normalize_number(numbers[-1]) if numbers else None


def completion_to_text(completion: Any) -> str:
    """Handle both standard string completions and chat-message completions."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def extract_model_answer(completion: Any) -> str | None:
    text = completion_to_text(completion)
    final_match = FINAL_ANSWER_RE.search(text)
    if final_match:
        return normalize_number(final_match.group(1))
    numbers = NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def gsm8k_exact_match_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, reference in zip(completions, answer, strict=False):
        pred = extract_model_answer(completion)
        target = extract_reference_answer(reference)
        rewards.append(1.0 if pred is not None and target is not None and pred == target else 0.0)
    return rewards


def format_prompt(question: str) -> str:
    return (
        "Solve the math problem. Show the reasoning briefly, then put the final numeric answer "
        "on its own line in the form #### <answer>.\n\n"
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
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", default="none")
    return parser.parse_args()


def build_grpo_config(**kwargs: Any) -> GRPOConfig:
    """Build GRPOConfig while tolerating small TRL API differences."""
    valid_fields = {field.name for field in fields(GRPOConfig)}
    filtered = {key: value for key, value in kwargs.items() if key in valid_fields}
    dropped = sorted(set(kwargs) - valid_fields)
    if dropped:
        print(f"Dropping unsupported GRPOConfig fields for this TRL version: {dropped}")
    return GRPOConfig(**filtered)


def main() -> None:
    args = parse_args()
    train_dataset = build_dataset(args.split, args.dataset_limit)

    training_args = build_grpo_config(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
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
