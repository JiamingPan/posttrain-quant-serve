#!/usr/bin/env bash
# Benchmark one checkpoint with vLLM offline generation on a single A40.
#
# Submit from the repo root, for example:
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct \
#   LABEL=base_fp16 \
#   QUANTIZATION=none \
#   OUTPUT_DIR=$PQS_ROOT/results/serving/qwen2_5_1_5b \
#   sbatch --account=huterer2 --time=00:45:00 --export=ALL slurm/serve_benchmark.sbatch
#
# If vLLM is not installed, install it once first with:
#   INSTALL_SERVING_DEPS=1 NUM_PROMPTS=1 MAX_NEW_TOKENS=1 ...

#SBATCH --job-name=pqs-serve-bench
#SBATCH --account=huterer2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/activate_great_lakes.sh

python scripts/cluster_check.py

if ! python -c "import vllm" >/dev/null 2>&1; then
  if [[ "${INSTALL_SERVING_DEPS:-0}" != "1" ]]; then
    cat >&2 <<'EOT'
vLLM is not importable in the active environment.

Install serving dependencies once, not from several benchmark jobs at the same time:

  INSTALL_SERVING_DEPS=1 NUM_PROMPTS=1 MAX_NEW_TOKENS=1 \
  sbatch --job-name=pqs-serve-install --account=huterer2 --time=00:45:00 --export=ALL slurm/serve_benchmark.sbatch

This wrapper redirects pip/Cargo/Rust temp/cache directories under PQS_ROOT.
EOT
    exit 2
  fi

  echo "vLLM is not importable; installing requirements-serving.txt into the active env."
  lock_file="$PQS_ROOT/serving_deps_install.lock"
  (
    flock 9
    if ! python -c "import vllm" >/dev/null 2>&1; then
      python -m pip install -r requirements-serving.txt
    fi
  ) 9>"$lock_file"
fi

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LABEL="${LABEL:-$(basename "$MODEL")}"
QUANTIZATION="${QUANTIZATION:-none}"
OUTPUT_DIR="${OUTPUT_DIR:-$PQS_ROOT/results/serving/qwen2_5_1_5b}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
REQUEST_MODE="${REQUEST_MODE:-sequential}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
SPLIT="${SPLIT:-test}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

mkdir -p "$OUTPUT_DIR"

cmd=(
  python scripts/benchmark.py
  --model "$MODEL"
  --label "$LABEL"
  --quantization "$QUANTIZATION"
  --num-prompts "$NUM_PROMPTS"
  --request-mode "$REQUEST_MODE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --split "$SPLIT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$BATCH_SIZE" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "$MAX_NUM_SEQS" ]]; then
  cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

echo "=== vLLM serving benchmark config ==="
echo "MODEL=$MODEL"
echo "LABEL=$LABEL"
echo "QUANTIZATION=$QUANTIZATION"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "REQUEST_MODE=$REQUEST_MODE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "DTYPE=$DTYPE"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "SPLIT=$SPLIT"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
