"""Deterministic, download-free Qwen3 fixture for CUDA integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def tiny_qwen3_config() -> Any:
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=False,
    )


def build_tiny_qwen3(*, device: str | torch.device | None = None) -> Any:
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_qwen3_config())
    return model.to(device) if device is not None else model


def write_tiny_qwen3(output_dir: str | Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    vocab.update({f"tok{index}": index + 4 for index in range(124)})
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{{ message['content'] }} <eos> {% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    model = build_tiny_qwen3()
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_tiny_qwen3(args.output_dir)


if __name__ == "__main__":
    main()
