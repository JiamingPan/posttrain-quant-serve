#!/usr/bin/env bash
# Start a Jupyter notebook server on a Great Lakes GPU node.
#
# Submit from the repo root:
#   sbatch --account=cavestru0 --time=06:00:00 slurm/jupyter_gpu.sbatch
#
# Then read jupyter_info_gpu.txt for the SSH tunnel command.

#SBATCH --job-name=pqs-jupyter
#SBATCH --account=huterer0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs notebooks

source scripts/activate_great_lakes.sh

unset XDG_RUNTIME_DIR

if ! python -m jupyter --version >/dev/null 2>&1; then
  cat >&2 <<'EOT'
Jupyter is not installed in the active project environment.

Install it once from the repo root:

  source scripts/activate_great_lakes.sh
  python -m pip install 'notebook>=7.0.0' 'ipykernel>=6.29.0'

Then resubmit this Slurm job.
EOT
  exit 1
fi

NOTEBOOK_DIR="$SLURM_SUBMIT_DIR/notebooks"
INFO_FILE="$SLURM_SUBMIT_DIR/jupyter_info_gpu.txt"

nport="$(shuf -i 8000-9000 -n 1)"
ip="$(hostname -i)"
hn="$(hostname)"
me="$(whoami)"

cat > "$INFO_FILE" <<EOT
Jupyter notebook running on host ${hn}.

Copy this command into your local Mac terminal:
ssh -N -L ${nport}:${ip}:${nport} ${me}@greatlakes.arc-ts.umich.edu

Then open:
http://localhost:${nport}

Notebook directory:
${NOTEBOOK_DIR}

Useful smoke run path:
${PQS_ROOT}/ckpts/smoke_qwen2_5_0_5b_grpo_100step
EOT

echo "Wrote connection info to ${INFO_FILE}"
cat "$INFO_FILE"

cd "$NOTEBOOK_DIR"

python -m jupyter notebook \
  --no-browser \
  --port="$nport" \
  --ip=0.0.0.0
