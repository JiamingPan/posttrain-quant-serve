# Cluster Setup Notes

Run these on the Slurm cluster, not on a login shell without GPUs.

## Environment

Create a fresh environment and install the PyTorch build that matches the cluster CUDA module.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install torch/torchvision/torchaudio from the cluster-recommended CUDA index.
# Then:
python -m pip install -r requirements.txt
```

Verify CUDA:

```bash
nvidia-smi
python scripts/cluster_check.py
```

## Hugging Face

```bash
huggingface-cli login
bash scripts/download_models.sh
```

## torchtune Config

Use torchtune's official config as the source of truth:

```bash
bash scripts/prepare_torchtune_config.sh
```

Then launch the smoke run:

```bash
CONFIG=configs/smoke_qwen2_5_0_5b.yaml MAX_STEPS=100 bash scripts/finetune.sh
```
