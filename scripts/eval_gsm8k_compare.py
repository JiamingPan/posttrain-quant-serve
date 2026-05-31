"""Compare FP16 and bnb-NF4 W4 base/GRPO checkpoints on GSM8K.

Default limit is 50 examples. Use --limit 100 when you want a less noisy W4
delta after the smoke path is working.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from gsm8k_reward import extract_model_answer, extract_reference_answer, has_prompt_leak_after_answer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


HONESTY_NOTE = (
    "W4 = bitsandbytes NF4 double-quant, a load-time 4-bit scheme. "
    "This is NOT AWQ W4G128; an AWQ run is needed to confirm the result is not "
    "specific to the bnb scheme. Greedy decoding (do_sample=False); accuracy is "
    "exact-match on the parsed #### answer."
)


def format_prompt(question: str) -> str:
    return (
        "Solve the math problem. Show the reasoning briefly. End with exactly one final line "
        "in the form #### <answer>, then stop. Do not write another problem or dialogue "
        "after the answer.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def parse_args() -> argparse.Namespace:
    pqs_root = os.environ.get("PQS_ROOT", "/scratch/huterer_root/huterer0/jiamingp/pqs")
    default_trained = f"{pqs_root}/ckpts/qwen2_5_0_5b_grpo_g8_dr100"
    default_output = f"{pqs_root}/evals/gsm8k_compare_test50_g8_dr100_bnb_w4"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--trained_model",
        default=default_trained,
        help="GRPO checkpoint to evaluate; defaults to the Day 3 g8_dr100 checkpoint under PQS_ROOT.",
    )
    parser.add_argument("--output_dir", default=default_output)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=50, help="Use 50 for smoke; use 100 to reduce W4 delta noise.")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_reference_tokens", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--precisions",
        default="fp16,w4",
        help="Comma-separated precisions to run: fp16, w4, or fp16,w4.",
    )
    parser.add_argument("--skip_reference_ppl", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    args.precisions = parse_precisions(args.precisions, parser)
    return args


def parse_precisions(value: str, parser: argparse.ArgumentParser) -> list[str]:
    allowed = {"fp16", "w4"}
    precisions = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not precisions:
        parser.error("--precisions must include at least one of: fp16,w4")
    invalid = [item for item in precisions if item not in allowed]
    if invalid:
        parser.error(f"unsupported --precisions entries: {invalid}; use fp16,w4")
    if len(set(precisions)) != len(precisions):
        parser.error(f"--precisions contains duplicates: {precisions}")
    return precisions


def validate_runtime_for_precisions(precisions: list[str]) -> None:
    if "w4" not in precisions:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("precision=w4 requires CUDA because bitsandbytes 4-bit loading needs a GPU.")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "precision=w4 requires bitsandbytes, but it is not installed in this Python environment. "
            "Install it in the active Great Lakes venv with `python -m pip install -r requirements.txt` "
            "or `python -m pip install 'bitsandbytes>=0.43.0'`."
        )


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


def load_model(model_name_or_path: str, precision: str, trust_remote_code: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if precision == "fp16":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
    elif precision == "w4":
        if device.type != "cuda":
            raise RuntimeError("precision=w4 requires CUDA because bitsandbytes 4-bit loading needs a GPU.")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("precision=w4 requires bitsandbytes. Install it with `pip install bitsandbytes`.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            device_map={"": 0},
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(f"Unsupported precision: {precision}")

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


def accuracy_stderr(correct: int, n: int) -> float | None:
    if n <= 0:
        return None
    p = correct / n
    return math.sqrt(p * (1.0 - p) / n)


def evaluate_model(
    variant: str,
    precision: str,
    model_name_or_path: str,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | int | str]:
    label = f"{variant}_{precision}"
    print(f"\n=== Evaluating {label}: {model_name_or_path} ===", flush=True)
    model, tokenizer, device = load_model(model_name_or_path, precision, args.trust_remote_code)

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
                "variant": variant,
                "precision": precision,
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
        "variant": variant,
        "precision": precision,
        "model": model_name_or_path,
        "num_examples": n,
        "accuracy": correct / n if n else 0.0,
        "accuracy_stderr": accuracy_stderr(correct, n),
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
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def write_results_matrix(output_dir: Path, summaries: list[dict[str, float | int | str]]) -> str:
    matrix_path = output_dir / "results_matrix.csv"
    fieldnames = [
        "variant",
        "precision",
        "label",
        "num_examples",
        "accuracy",
        "accuracy_stderr",
        "correct",
        "parse_rate",
        "prompt_leak_rate",
        "completion_chars_mean",
        "reference_nll_per_token",
        "reference_ppl",
        "reference_ppl_token_count",
        "model",
        "predictions_path",
    ]
    with matrix_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})
    return str(matrix_path)


def read_predictions(output_dir: Path, label: str) -> list[dict[str, object]]:
    path = output_dir / f"{label}_predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def classify_change(before_correct: bool, after_correct: bool) -> str:
    if not before_correct and after_correct:
        return "improved"
    if before_correct and not after_correct:
        return "worsened"
    return "unchanged"


def write_paired_quant_effect(output_dir: Path, variant_names: list[str]) -> dict[str, object]:
    paired_path = output_dir / "paired_quant_effect.csv"
    fieldnames = [
        "variant",
        "index",
        "target_answer",
        "fp16_answer",
        "w4_answer",
        "fp16_correct",
        "w4_correct",
        "quant_change",
        "question",
    ]
    quant_effect: dict[str, dict[str, int]] = {}

    with paired_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant in variant_names:
            fp16_rows = read_predictions(output_dir, f"{variant}_fp16")
            w4_rows = read_predictions(output_dir, f"{variant}_w4")
            if not fp16_rows or not w4_rows:
                continue

            counts = {"improved": 0, "worsened": 0, "unchanged": 0}
            for fp16, w4 in zip(fp16_rows, w4_rows):
                change = classify_change(bool(fp16["exact_match"]), bool(w4["exact_match"]))
                counts[change] += 1
                writer.writerow(
                    {
                        "variant": variant,
                        "index": fp16["index"],
                        "target_answer": fp16["target_answer"],
                        "fp16_answer": fp16["parsed_answer"],
                        "w4_answer": w4["parsed_answer"],
                        "fp16_correct": fp16["exact_match"],
                        "w4_correct": w4["exact_match"],
                        "quant_change": change,
                        "question": fp16["question"],
                    }
                )
            quant_effect[variant] = counts

    return {
        "paired_quant_effect_path": str(paired_path),
        "quant_effect": quant_effect,
    }


def accuracy_delta(summaries: dict[str, dict[str, object]], positive_label: str, negative_label: str) -> float | None:
    positive = summaries.get(positive_label)
    negative = summaries.get(negative_label)
    if positive is None or negative is None:
        return None
    return float(positive["accuracy"]) - float(negative["accuracy"])


def legacy_fp16_aliases(summaries: dict[str, dict[str, object]], delta_fp16: float | None) -> dict[str, object]:
    aliases: dict[str, object] = {}
    if "base_fp16" in summaries:
        aliases["base"] = summaries["base_fp16"]
    if "g8_dr100_fp16" in summaries:
        aliases["trained"] = summaries["g8_dr100_fp16"]
    if delta_fp16 is not None:
        aliases["delta_accuracy"] = delta_fp16
    return aliases


def format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def print_verdict(delta_fp16: float | None, delta_w4: float | None, gain_survival: float | None) -> None:
    if delta_fp16 is not None and delta_w4 is None:
        print(
            "\nVerdict: Only FP16 was evaluated, so W4 survival cannot be measured in this run. "
            f"RL gain at FP16 = {format_delta(delta_fp16)}.",
            flush=True,
        )
        return
    if delta_fp16 is None and delta_w4 is not None:
        print(
            "\nVerdict: Only W4 was evaluated, so W4 survival relative to FP16 cannot be measured in this run. "
            f"RL gain at W4 = {format_delta(delta_w4)}.",
            flush=True,
        )
        return
    if delta_fp16 is None or delta_w4 is None or gain_survival is None:
        print(
            "\nVerdict: Need both FP16 and W4 rows to measure whether the GRPO gain survives quantization.",
            flush=True,
        )
        return

    if gain_survival < -0.05:
        conclusion = "W4 appears to erase part of the GRPO benefit under bnb-NF4."
    else:
        conclusion = "GRPO benefit survives bnb-NF4 W4, pending AWQ confirmation."
    print(
        "\nVerdict: "
        f"RL gain at FP16 = {format_delta(delta_fp16)}, "
        f"at W4 = {format_delta(delta_w4)}, "
        f"survival = {format_delta(gain_survival)} -> {conclusion}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    validate_runtime_for_precisions(args.precisions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(args.split, args.limit)
    run_config = vars(args) | {"num_examples": len(rows)}
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    variants = []
    for precision in args.precisions:
        variants.append(("base", args.base_model, precision))
        variants.append(("g8_dr100", args.trained_model, precision))

    summaries = [
        evaluate_model(variant, precision, model_path, rows, args, output_dir)
        for variant, model_path, precision in variants
    ]
    results_matrix_path = write_results_matrix(output_dir, summaries)
    quant_comparison = write_paired_quant_effect(output_dir, ["base", "g8_dr100"])
    variant_summaries = {str(summary["label"]): summary for summary in summaries}

    delta_fp16 = accuracy_delta(variant_summaries, "g8_dr100_fp16", "base_fp16")
    delta_w4 = accuracy_delta(variant_summaries, "g8_dr100_w4", "base_w4")
    gain_survival = delta_w4 - delta_fp16 if delta_w4 is not None and delta_fp16 is not None else None
    quant_drop_base = accuracy_delta(variant_summaries, "base_fp16", "base_w4")
    quant_drop_g8 = accuracy_delta(variant_summaries, "g8_dr100_fp16", "g8_dr100_w4")
    summary = {
        "split": args.split,
        "limit": args.limit,
        "precisions": args.precisions,
        "notes": [HONESTY_NOTE],
        "variants": variant_summaries,
        "results_matrix_path": results_matrix_path,
        "delta_fp16": delta_fp16,
        "delta_w4": delta_w4,
        "gain_survival": gain_survival,
        "quant_drop_base": quant_drop_base,
        "quant_drop_g8": quant_drop_g8,
        **variant_summaries,
        **legacy_fp16_aliases(variant_summaries, delta_fp16),
        **quant_comparison,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Honesty note ===")
    print(HONESTY_NOTE)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")
    print(f"Wrote {results_matrix_path}")
    print(f"Wrote {quant_comparison['paired_quant_effect_path']}")
    print_verdict(delta_fp16, delta_w4, gain_survival)


if __name__ == "__main__":
    main()
