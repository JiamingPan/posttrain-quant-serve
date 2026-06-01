"""Quantize one causal LM checkpoint with AutoAWQ W4G128.

This creates a saved AWQ checkpoint. It is intentionally separate from
bitsandbytes NF4, which is load-time quantization and does not write a quantized
model to disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from gsm8k_reward import build_gsm8k_chat_text
from transformers import AutoTokenizer


def str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--calib_split", default="train", choices=["train", "test"])
    parser.add_argument("--calib_limit", type=int, default=128)
    parser.add_argument("--max_calib_seq_len", type=int, default=512)
    parser.add_argument("--n_parallel_calib_samples", type=int, default=8)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--zero_point", type=str_to_bool, default=True)
    parser.add_argument("--version", default="GEMM")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def require_autoawq() -> Any:
    if importlib.util.find_spec("awq") is None:
        raise RuntimeError(
            "AutoAWQ is not installed. Install optional AWQ deps with "
            "`python -m pip install -r requirements-awq.txt`."
        )
    from awq import AutoAWQForCausalLM

    return AutoAWQForCausalLM


def load_calibration_texts(split: str, limit: int, tokenizer: Any) -> list[str]:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    texts = []
    for row in dataset:
        # AWQ calibrates activation scales, so include the task prompt and the
        # reference solution in the same format used by evaluation.
        prompt = build_gsm8k_chat_text(tokenizer, row["question"])
        texts.append(f"{prompt}\n{row['answer']}")
    return texts


def main() -> None:
    args = parse_args()
    AutoAWQForCausalLM = require_autoawq()

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calib_data = load_calibration_texts(args.calib_split, args.calib_limit, tokenizer)

    print("=== AWQ quantization config ===", flush=True)
    print(json.dumps(quant_config, indent=2), flush=True)
    print(f"model_name_or_path={args.model_name_or_path}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"calibration_samples={len(calib_data)} split={args.calib_split}", flush=True)

    model = AutoAWQForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        trust_remote_code=args.trust_remote_code,
    )
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data,
        max_calib_samples=len(calib_data),
        max_calib_seq_len=args.max_calib_seq_len,
        n_parallel_calib_samples=args.n_parallel_calib_samples,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = {
        "model_name_or_path": args.model_name_or_path,
        "quant_method": "awq",
        "quant_config": quant_config,
        "calib_split": args.calib_split,
        "calib_limit": args.calib_limit,
        "max_calib_seq_len": args.max_calib_seq_len,
        "n_parallel_calib_samples": args.n_parallel_calib_samples,
        "note": "AWQ W4G128 quantized with AutoAWQ on chat-formatted GSM8K calibration text.",
    }
    (args.output_dir / "awq_quantize_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f'AWQ model saved at "{args.output_dir}"', flush=True)


if __name__ == "__main__":
    main()
