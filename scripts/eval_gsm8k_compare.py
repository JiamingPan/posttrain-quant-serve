"""Compare base vs GRPO-trained checkpoints on GSM8K exact-match accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from gsm8k_reward import extract_model_answer, extract_reference_answer, has_prompt_leak_after_answer
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(question: str) -> str:
    return (
        "Solve the math problem. Show the reasoning briefly, then put the final numeric answer "
        "on its own line in the form #### <answer>.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def parse_args() -> argparse.Namespace:
    pqs_root = os.environ.get("PQS_ROOT", "/scratch/huterer_root/huterer0/jiamingp/pqs")
    default_trained = f"{pqs_root}/ckpts/smoke_qwen2_5_0_5b_grpo_100step"
    default_output = f"{pqs_root}/evals/gsm8k_compare_100step"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trained_model", default=default_trained)
    parser.add_argument("--output_dir", default=default_output)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_reference_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--skip_reference_ppl", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def load_eval_rows(split: str, limit: int) -> list[dict[str, str]]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows: list[dict[str, str]] = []
    for i, row in enumerate(dataset):
        target = extract_reference_answer(row["answer"])
        rows.append(
            {
                "index": i,
                "question": row["question"],
                "reference": row["answer"],
                "target_answer": target or "",
                "prompt": format_prompt(row["question"]),
            }
        )
    return rows


def load_model(model_name_or_path: str, trust_remote_code: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def generate_one(model, tokenizer, device, prompt: str, max_prompt_length: int, max_new_tokens: int) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    output = model.generate(**generation_kwargs)
    prompt_tokens = inputs["input_ids"].shape[-1]
    completion_tokens = output[0, prompt_tokens:]
    return tokenizer.decode(completion_tokens, skip_special_tokens=True)


@torch.inference_mode()
def score_reference_solution(
    model,
    tokenizer,
    device,
    prompt: str,
    reference: str,
    max_prompt_length: int,
    max_reference_tokens: int,
) -> tuple[float, int]:
    """Teacher-force the GSM8K reference solution and return NLL/token."""
    prompt_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    full_inputs = tokenizer(
        prompt + "\n" + reference,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length + max_reference_tokens,
    )

    input_ids = full_inputs["input_ids"].to(device)
    attention_mask = full_inputs["attention_mask"].to(device)
    labels = input_ids.clone()
    prompt_len = min(prompt_inputs["input_ids"].shape[-1], labels.shape[-1])
    labels[:, :prompt_len] = -100

    token_count = int((labels != -100).sum().item())
    if token_count == 0:
        return float("nan"), 0

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(outputs.loss.item()), token_count


def evaluate_model(
    label: str,
    model_name_or_path: str,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | int | str]:
    print(f"\n=== Evaluating {label}: {model_name_or_path} ===", flush=True)
    model, tokenizer, device = load_model(model_name_or_path, args.trust_remote_code)

    predictions_path = output_dir / f"{label}_predictions.jsonl"
    correct = 0
    parsed = 0
    prompt_leaks = 0
    completion_chars: list[int] = []
    nll_weighted_sum = 0.0
    nll_token_count = 0

    with predictions_path.open("w") as f:
        for row in rows:
            reference_nll = None
            reference_tokens = 0
            if not args.skip_reference_ppl:
                reference_nll, reference_tokens = score_reference_solution(
                    model,
                    tokenizer,
                    device,
                    row["prompt"],
                    row["reference"],
                    args.max_prompt_length,
                    args.max_reference_tokens,
                )
                if math.isfinite(reference_nll) and reference_tokens > 0:
                    nll_weighted_sum += reference_nll * reference_tokens
                    nll_token_count += reference_tokens

            completion = generate_one(
                model,
                tokenizer,
                device,
                row["prompt"],
                args.max_prompt_length,
                args.max_new_tokens,
            )
            pred = extract_model_answer(completion)
            is_correct = pred is not None and pred == row["target_answer"]
            leak = has_prompt_leak_after_answer(completion)
            correct += int(is_correct)
            parsed += int(pred is not None)
            prompt_leaks += int(leak)
            completion_chars.append(len(completion))

            record = {
                **row,
                "model_label": label,
                "model_path": model_name_or_path,
                "completion": completion,
                "parsed_answer": pred,
                "exact_match": is_correct,
                "prompt_leak": leak,
                "completion_chars": len(completion),
                "reference_nll_per_token": reference_nll,
                "reference_tokens": reference_tokens,
            }
            f.write(json.dumps(record) + "\n")

            print(
                f"[{label}] {row['index'] + 1}/{len(rows)} "
                f"pred={pred} target={row['target_answer']} correct={is_correct}",
                flush=True,
            )

    n = len(rows)
    reference_nll_per_token = nll_weighted_sum / nll_token_count if nll_token_count else None
    reference_ppl = math.exp(reference_nll_per_token) if reference_nll_per_token is not None else None

    summary = {
        "label": label,
        "model": model_name_or_path,
        "num_examples": n,
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "parse_rate": parsed / n if n else 0.0,
        "prompt_leak_rate": prompt_leaks / n if n else 0.0,
        "completion_chars_mean": sum(completion_chars) / n if n else 0.0,
        "reference_nll_per_token": reference_nll_per_token,
        "reference_ppl": reference_ppl,
        "reference_ppl_token_count": nll_token_count,
        "predictions_path": str(predictions_path),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def write_paired_comparison(output_dir: Path) -> dict[str, int | str]:
    base_rows = [json.loads(line) for line in (output_dir / "base_predictions.jsonl").read_text().splitlines()]
    trained_rows = [
        json.loads(line) for line in (output_dir / "trained_predictions.jsonl").read_text().splitlines()
    ]

    paired_path = output_dir / "paired_comparison.csv"
    improved = worsened = unchanged = 0
    with paired_path.open("w", newline="") as f:
        fieldnames = [
            "index",
            "target_answer",
            "base_answer",
            "trained_answer",
            "base_correct",
            "trained_correct",
            "change",
            "question",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for base, trained in zip(base_rows, trained_rows):
            if not base["exact_match"] and trained["exact_match"]:
                change = "improved"
                improved += 1
            elif base["exact_match"] and not trained["exact_match"]:
                change = "worsened"
                worsened += 1
            else:
                change = "unchanged"
                unchanged += 1
            writer.writerow(
                {
                    "index": base["index"],
                    "target_answer": base["target_answer"],
                    "base_answer": base["parsed_answer"],
                    "trained_answer": trained["parsed_answer"],
                    "base_correct": base["exact_match"],
                    "trained_correct": trained["exact_match"],
                    "change": change,
                    "question": base["question"],
                }
            )

    return {
        "paired_comparison_path": str(paired_path),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(args.split, args.limit)
    run_config = vars(args) | {"num_examples": len(rows)}
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    summaries = [
        evaluate_model("base", args.base_model, rows, args, output_dir),
        evaluate_model("trained", args.trained_model, rows, args, output_dir),
    ]
    comparison = write_paired_comparison(output_dir)
    base_nll = summaries[0]["reference_nll_per_token"]
    trained_nll = summaries[1]["reference_nll_per_token"]
    base_ppl = summaries[0]["reference_ppl"]
    trained_ppl = summaries[1]["reference_ppl"]

    summary = {
        "split": args.split,
        "limit": args.limit,
        "base": summaries[0],
        "trained": summaries[1],
        "delta_accuracy": summaries[1]["accuracy"] - summaries[0]["accuracy"],
        "delta_reference_nll_per_token": (
            trained_nll - base_nll if trained_nll is not None and base_nll is not None else None
        ),
        "delta_reference_ppl": trained_ppl - base_ppl if trained_ppl is not None and base_ppl is not None else None,
        "reference_ppl_ratio": trained_ppl / base_ppl if trained_ppl is not None and base_ppl else None,
        **comparison,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
