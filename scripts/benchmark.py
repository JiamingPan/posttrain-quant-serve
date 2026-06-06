"""Benchmark one checkpoint with vLLM offline generation.

The benchmark is intentionally self-contained: it loads a checkpoint with vLLM,
builds realistic GSM8K chat-template prompts, runs offline requests, and writes
one result row to JSONL and CSV. It can run either sequential one-prompt requests
or batched requests that send many prompts to vLLM together. It does not require
a separately managed server process.

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
import os
import statistics
import subprocess
import threading
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
    "request_mode",
    "batch_size",
    "num_batches",
    "max_num_seqs",
    "max_num_batched_tokens",
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
    parser.add_argument(
        "--request-mode",
        choices=["sequential", "batch"],
        default="sequential",
        help=(
            "sequential sends one prompt per llm.generate call; batch sends chunks of "
            "--batch-size prompts per call to test vLLM batch pressure."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prompts per llm.generate call in --request-mode batch. Defaults to --num-prompts.",
    )
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
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Optional vLLM max_num_seqs override.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM max_num_batched_tokens override.",
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


def _visible_gpu_id() -> str | None:
    """Return the first Slurm-visible GPU id for nvidia-smi, if available."""
    for env_name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS"):
        value = os.environ.get(env_name)
        if not value:
            continue
        first = value.split(",")[0].strip()
        if first and first.lower() not in {"none", "nodevfiles"}:
            return first
    return None


def _query_gpu_used_gb(gpu_id: str | None) -> float | None:
    cmd_base = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    cmds = [cmd_base]
    if gpu_id:
        cmds.insert(0, ["nvidia-smi", "--id", gpu_id, *cmd_base[1:]])

    completed = None
    for cmd in cmds:
        try:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            break
        except FileNotFoundError:
            return None
        except subprocess.CalledProcessError:
            continue
    if completed is None:
        return None

    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line.split()[0]) / 1024.0)
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


class GpuMemorySampler:
    """Poll nvidia-smi so vLLM worker/custom-allocator memory is visible."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.gpu_id = _visible_gpu_id()
        self.baseline_gb: float | None = None
        self.peak_gb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.baseline_gb = _query_gpu_used_gb(self.gpu_id)
        self.peak_gb = self.baseline_gb
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            current = _query_gpu_used_gb(self.gpu_id)
            if current is None:
                continue
            if self.peak_gb is None or current > self.peak_gb:
                self.peak_gb = current

    def stop(self) -> tuple[float | None, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        current = _query_gpu_used_gb(self.gpu_id)
        if current is not None and (self.peak_gb is None or current > self.peak_gb):
            self.peak_gb = current
        return self.baseline_gb, self.peak_gb


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


def build_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "quantization": None if args.quantization == "none" else args.quantization,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    return LLM(**llm_kwargs)


def generate_sequential(llm: Any, sampling: Any, prompts: list[str], label: str) -> tuple[list[float], int, int]:
    latencies: list[float] = []
    generated_tokens = 0

    for i, prompt in enumerate(prompts, start=1):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        latency = time.perf_counter() - start
        latencies.append(latency)
        generated_tokens += sum(count_generated_tokens(output) for output in outputs)
        print(
            f"[{label}] sequential {i}/{len(prompts)} "
            f"latency_s={latency:.3f} generated_tokens={generated_tokens}",
            flush=True,
        )

    return latencies, generated_tokens, len(prompts)


def generate_batched(
    llm: Any,
    sampling: Any,
    prompts: list[str],
    label: str,
    batch_size: int,
) -> tuple[list[float], int, int]:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    latencies: list[float] = []
    generated_tokens = 0
    num_batches = 0
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for start_index in range(0, len(prompts), batch_size):
        num_batches += 1
        batch_prompts = prompts[start_index : start_index + batch_size]
        start = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling)
        latency = time.perf_counter() - start
        latencies.extend([latency] * len(batch_prompts))
        generated_tokens += sum(count_generated_tokens(output) for output in outputs)
        print(
            f"[{label}] batch {num_batches}/{total_batches} "
            f"batch_size={len(batch_prompts)} latency_s={latency:.3f} "
            f"generated_tokens={generated_tokens}",
            flush=True,
        )

    return latencies, generated_tokens, num_batches


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from vllm import SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Benchmarking requires vLLM and torch. Install serving deps with "
            "`python -m pip install -r requirements-serving.txt`."
        ) from exc

    prompts = load_prompts(args.model, args.split, args.num_prompts, args.trust_remote_code)
    if not prompts:
        raise SystemExit("No prompts loaded; check --split and --num-prompts.")
    batch_size = args.batch_size if args.batch_size is not None else len(prompts)
    if args.request_mode == "sequential":
        batch_size = 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    memory_sampler = GpuMemorySampler()
    memory_sampler.start()

    llm = build_llm(args)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    wall_start = time.perf_counter()

    label = args.label or Path(args.model).name
    if args.request_mode == "batch":
        latencies, generated_tokens, num_batches = generate_batched(llm, sampling, prompts, label, batch_size)
    else:
        latencies, generated_tokens, num_batches = generate_sequential(llm, sampling, prompts, label)

    wall_time = time.perf_counter() - wall_start
    peak_mem_gb = None
    peak_reserved_gb = None
    if torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    gpu_mem_baseline_gb, peak_gpu_used_gb = memory_sampler.stop()
    if peak_gpu_used_gb is not None and (peak_mem_gb is None or peak_mem_gb == 0.0):
        peak_mem_gb = peak_gpu_used_gb
        print(
            "Using nvidia-smi peak device memory for peak_mem_gb "
            f"(baseline_gb={gpu_mem_baseline_gb}, peak_gb={peak_gpu_used_gb}).",
            flush=True,
        )

    row = {
        "label": args.label or Path(args.model).name,
        "model": args.model,
        "quantization": args.quantization,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "request_mode": args.request_mode,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
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
