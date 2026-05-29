"""Smoke tests for the GSM8K reward parser."""

from __future__ import annotations

from gsm8k_reward import extract_model_answer, group_reward_variance_stats, gsm8k_exact_match_reward, has_prompt_leak_after_answer


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
            "Final answer: 312 pounds.\n\nYou are given a list of integers.",
            "#### 28.Select the correct answer: Among the multiples of 12...",
            "Final answer: 108.\n\nIf alligators are particularly well adapted to life in water...",
        ],
        answer=[
            "work #### 312",
            "work #### 16",
            "work #### 72",
            "work #### 312",
            "work #### 16",
            "work #### 312",
            "work #### 28",
            "work #### 990",
        ],
    )
    assert rewards == [1.0, 0.0, 1.0, 0.75, -0.25, 0.75, 0.75, -0.25], rewards

    assert has_prompt_leak_after_answer("Final answer: 312\n\nHuman: next task")
    assert has_prompt_leak_after_answer("Final answer: 312\n\nYou are given a list of integers.")
    assert has_prompt_leak_after_answer("#### 28.Select the correct answer: ...")
    stats = group_reward_variance_stats(
        completions=["Final answer: 1", "Final answer: 2", "Final answer: 3", "Final answer: 4"],
        answer=["#### 9", "#### 9", "#### 9", "#### 9"],
        num_generations=4,
    )
    assert stats[0]["zero_reward_variance"] is True
    assert not has_prompt_leak_after_answer("Human: context\n\nFinal answer: 312")
    print("reward parser checks passed")


if __name__ == "__main__":
    main()
