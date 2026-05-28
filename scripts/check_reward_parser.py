"""Smoke tests for the GSM8K reward parser."""

from __future__ import annotations

from gsm8k_reward import extract_model_answer, gsm8k_exact_match_reward, has_prompt_leak_after_answer


def main() -> None:
    cases = [
        ("#### 312", "312"),
        ("Final answer: 8 pounds", "8"),
        ("the answer is 312.", "312"),
        ("#### <answer>312</answer> Pages/Year", "312"),
        ("Therefore, the direct numeric answer to the problem is 8 pounds.", "8"),
        ("work... 24*2=48\nFinal answer: 8 pounds", "8"),
        ([{"role": "assistant", "content": "Final answer: 1,024"}], "1024"),
    ]
    for completion, expected in cases:
        actual = extract_model_answer(completion)
        assert actual == expected, f"{completion!r}: expected {expected!r}, got {actual!r}"

    rewards = gsm8k_exact_match_reward(
        completions=[
            "Final answer: 312",
            "Final answer: 8",
            "#### <answer>72</answer>",
            "Final answer: 312\n\nHuman: Solve the next problem.",
            "Final answer: 8\n\nProblem: another task",
        ],
        answer=["work #### 312", "work #### 16", "work #### 72", "work #### 312", "work #### 16"],
    )
    assert rewards == [1.0, 0.0, 1.0, 0.75, -0.25], rewards

    assert has_prompt_leak_after_answer("Final answer: 312\n\nHuman: next task")
    assert not has_prompt_leak_after_answer("Human: context\n\nFinal answer: 312")
    print("reward parser checks passed")


if __name__ == "__main__":
    main()
