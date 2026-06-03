"""Benchmark one checkpoint with vLLM offline generation.

The benchmark is intentionally self-contained: it loads a checkpoint with vLLM,
builds realistic GSM8K chat-template prompts, runs sequential offline requests,
and writes one result row to JSONL and CSV. It does not require a separately
managed server process.

Example:

    python scripts/benchmark.py \
      --model $PQS_ROOT/ckpts/qwen2_5_1_5b_grpo_data1000_chat \
      --label grpo_fp16 \
      --quantization none \
      --output-dir $PQS_ROOT/results/serving/qwen2_5_1_5b
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from gsm8k_reward import build_gsm8k_chat_text


RESULT_FIELDS = [
    "label",
    "model",
    "quantization",
    "num_prompts",
    "max_new_tokens",
    "dtype",
    "max_model_len",
    "generated_tokens",
    "wall_time_s",
    "throughput_tok_s",
    "latency_p50_s",
    "latency_p95_s",
    "peak_mem_gb",
    "peak_reserved_gb",
    "split",
    "output_dir",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument("--label", default=None, help="Short row label, e.g. base_fp16 or grpo_awq.")
    parser.add_argument(
        "--quantization",
        choices=["none", "awq"],
        default="none",
        help="Use 'awq' for saved AWQ W4G128 checkpoints; use 'none' for dense checkpoints.",
    )
    parser.add_argument("--num-prompts", type=int, default=64, help="Number of GSM8K prompts to run.")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="GSM8K split.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="vLLM dtype. Default is float16 because AWQ kernels used here reject bf16.",
    )
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model length.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization cap.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/serving_benchmark_runs"),
        help="Directory for serving_benchmark.jsonl and serving_benchmark.csv.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code in tokenizer/vLLM.")
    return parser.parse_args(argv)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def load_prompts(model_name_or_path: str, split: str, limit: int, trust_remote_code: bool) -> list[str]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Benchmark prompt loading requires datasets and transformers. Install project deps first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.select(range(min(limit, len(dataset))))
    return [build_gsm8k_chat_text(tokenizer, row["question"]) for row in dataset]


def count_generated_tokens(output: Any) -> int:
    completion = output.outputs[0]
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is None:
        return 0
    return len(token_ids)


def write_result(output_dir: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "serving_benchmark.jsonl"
    csv_path = output_dir / "serving_benchmark.csv"

    with jsonl_path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in RESULT_FIELDS})

    return jsonl_path, csv_path


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Benchmarking requires vLLM and torch. Install serving deps with "
            "`python -m pip install -r requirements-serving.txt`."
        ) from exc

    prompts = load_prompts(args.model, args.split, args.num_prompts, args.trust_remote_code)
    if not prompts:
        raise SystemExit("No prompts loaded; check --split and --num-prompts.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=args.model,
        quantization=None if args.quantization == "none" else args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    latencies: list[float] = []
    generated_tokens = 0
    wall_start = time.perf_counter()

    for i, prompt in enumerate(prompts, start=1):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        latency = time.perf_counter() - start
        latencies.append(latency)
        generated_tokens += sum(count_generated_tokens(output) for output in outputs)
        print(
            f"[{args.label or Path(args.model).name}] {i}/{len(prompts)} "
            f"latency_s={latency:.3f} generated_tokens={generated_tokens}",
            flush=True,
        )

    wall_time = time.perf_counter() - wall_start
    peak_mem_gb = None
    peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9

    row = {
        "label": args.label or Path(args.model).name,
        "model": args.model,
        "quantization": args.quantization,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "generated_tokens": generated_tokens,
        "wall_time_s": wall_time,
        "throughput_tok_s": generated_tokens / wall_time if wall_time > 0 else 0.0,
        "latency_p50_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_s": percentile(latencies, 0.95),
        "peak_mem_gb": peak_mem_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "split": args.split,
        "output_dir": str(args.output_dir),
    }
    return row


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    row = run_benchmark(args)
    jsonl_path, csv_path = write_result(args.output_dir, row)
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
