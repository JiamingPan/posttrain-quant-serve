"""Serve or smoke-test a checkpoint with vLLM.

Examples:

Start an OpenAI-compatible vLLM server for a dense FP16 checkpoint:

    python scripts/serve.py \
      --mode server \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --quantization none \
      --port 8000

Start an OpenAI-compatible vLLM server for a saved AWQ W4G128 checkpoint:

    python scripts/serve.py \
      --mode server \
      --model $PQS_ROOT/ckpts_awq/qwen2_5_1_5b_data1000_chat_awq_w4g128 \
      --quantization awq \
      --port 8000

Smoke the server from another shell:

    curl http://127.0.0.1:8000/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "$PQS_ROOT/ckpts_awq/qwen2_5_1_5b_data1000_chat_awq_w4g128",
        "prompt": "Solve the math problem. Show the reasoning briefly. Problem: Janet has 3 bags with 4 marbles each. How many marbles does she have? End with #### <answer>.",
        "max_tokens": 128,
        "temperature": 0
      }'

Run an offline vLLM generation smoke without holding a port:

    python scripts/serve.py \
      --mode offline \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

from gsm8k_reward import build_gsm8k_chat_text


DEFAULT_PROMPT = (
    "Janet has 3 bags with 4 marbles in each bag. She gives 5 marbles to her "
    "friend. How many marbles does Janet have left?"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local checkpoint path.")
    parser.add_argument(
        "--quantization",
        choices=["none", "awq"],
        default="none",
        help="Use 'awq' for saved AWQ W4G128 checkpoints; use 'none' for dense checkpoints.",
    )
    parser.add_argument(
        "--mode",
        choices=["server", "offline"],
        default="server",
        help="Start the OpenAI-compatible API server or run one offline generation smoke.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host.")
    parser.add_argument("--port", type=int, default=8000, help="API server port.")
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model length.")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="vLLM dtype. Default is float16 because AWQ kernels used here reject bf16.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization cap.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/vLLM where supported.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Offline smoke GSM8K-style question.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Offline smoke generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling top-p.")
    return parser.parse_args(argv)


def server_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.quantization == "awq":
        cmd.extend(["--quantization", "awq"])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def run_server(args: argparse.Namespace) -> None:
    cmd = server_command(args)
    print("Starting vLLM OpenAI-compatible server:", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_offline(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Offline serving requires vLLM and transformers. Install serving deps with "
            "`python -m pip install -r requirements-serving.txt`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    prompt = build_gsm8k_chat_text(tokenizer, args.prompt)

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
    outputs = llm.generate([prompt], sampling)
    completion = outputs[0].outputs[0].text
    print("=== Prompt ===")
    print(prompt)
    print("\n=== Completion ===")
    print(completion)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "server":
        run_server(args)
    else:
        run_offline(args)


if __name__ == "__main__":
    main()
