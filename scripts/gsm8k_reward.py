"""GSM8K answer extraction and exact-match reward helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from typing import Any


NUMBER_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?"
NUMBER_RE = re.compile(NUMBER_PATTERN)
ANSWER_TAG_RE = re.compile(rf"<answer>\s*({NUMBER_PATTERN})\s*</answer>", re.IGNORECASE)
HASH_ANSWER_RE = re.compile(rf"####\s*(?:<[^>]+>\s*)*({NUMBER_PATTERN})", re.IGNORECASE)
FINAL_PHRASE_RE = re.compile(
    rf"(?:final\s+(?:numeric\s+)?answer|direct\s+numeric\s+answer|the\s+answer\s+is|answer\s+is)"
    rf"[^-\d+]{{0,80}}({NUMBER_PATTERN})",
    re.IGNORECASE,
)


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").strip()
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return str(number.normalize())


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


def last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return normalize_number(matches[-1].group(1))


def extract_reference_answer(answer: str) -> str | None:
    for pattern in (HASH_ANSWER_RE, ANSWER_TAG_RE):
        value = last_match(pattern, answer)
        if value is not None:
            return value
    numbers = NUMBER_RE.findall(answer)
    return normalize_number(numbers[-1]) if numbers else None


def extract_model_answer(completion: Any) -> str | None:
    text = completion_to_text(completion)

    # Prefer explicit answer containers or final-answer language over a blind last number.
    for pattern in (ANSWER_TAG_RE, HASH_ANSWER_RE, FINAL_PHRASE_RE):
        value = last_match(pattern, text)
        if value is not None:
            return value

    numbers = NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def gsm8k_exact_match_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, reference in zip(completions, answer, strict=False):
        pred = extract_model_answer(completion)
        target = extract_reference_answer(reference)
        rewards.append(1.0 if pred is not None and target is not None and pred == target else 0.0)
    return rewards
