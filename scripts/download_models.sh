#!/usr/bin/env bash
set -euo pipefail

# Keep large model files outside git. HF_HOME can be pointed at cluster scratch.
: "${HF_HOME:=$PWD/.hf_cache}"

export HF_HOME

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
