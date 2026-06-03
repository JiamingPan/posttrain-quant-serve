# Cluster Setup Notes

Run these on the Slurm cluster. The repo, virtual environment, Hugging Face cache, and checkpoints should live on scratch, not in `$HOME`.

## Environment

Great Lakes path used for this project:

```bash
export PQS_ROOT=$PQS_ROOT
```

Create the environment:

```bash
module purge
module load python/3.11.5
module load cuda/12.8.1

mkdir -p "$PQS_ROOT/envs" "$PQS_ROOT/pip-cache" "$PQS_ROOT/hf_cache" "$PQS_ROOT/ckpts"
python -m venv "$PQS_ROOT/envs/posttrain-quant-serve"
source "$PQS_ROOT/envs/posttrain-quant-serve/bin/activate"

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$PQS_ROOT/pip-cache"
export HF_HOME="$PQS_ROOT/hf_cache"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify CUDA:

```bash
nvidia-smi
python - <<'PY'
import torch
from trl import GRPOTrainer
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("GRPOTrainer import OK")
PY
```

## Hugging Face

## Day 0 GRPO Smoke

Submit the 5-step shape check:

```bash
sbatch --export=ALL,MAX_STEPS=5,DATASET_LIMIT=10 slurm/smoke_single_gpu.sbatch
```

If clean, submit the 100-step smoke:

```bash
sbatch --export=ALL,MAX_STEPS=100,DATASET_LIMIT=10 slurm/smoke_single_gpu.sbatch
```
