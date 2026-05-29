"""Print installed TRL GRPOConfig fields relevant to Day 3 diagnostics."""

from __future__ import annotations

from dataclasses import fields

from trl import GRPOConfig

INTERESTING = [
    "num_generations",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "beta",
    "scale_rewards",
    "loss_type",
    "epsilon_high",
    "mask_truncated_completions",
    "log_completions",
    "num_completions_to_print",
]


def main() -> None:
    all_fields = {field.name: field for field in fields(GRPOConfig)}
    print("GRPOConfig fields:")
    for name in sorted(all_fields):
        print(name)
    print("\nRelevant Day 3 knobs:")
    for name in INTERESTING:
        if name in all_fields:
            field = all_fields[name]
            print(f"{name}: supported, default={field.default!r}")
        else:
            print(f"{name}: not supported by installed TRL")


if __name__ == "__main__":
    main()
