# Configs

Training and serving remain script-driven; this repository does not require a
parallel YAML configuration system.

Primary entry points:

```bash
# Existing single-GPU GRPO oracle
bash scripts/run_grpo_smoke.sh

# Direct PyTorch FSDP2 trainers
python -m train.fsdp_sft --help
python -m train.fsdp_grpo --help

# Single-node Slurm/torchrun wrapper
bash scripts/launch_slurm.sh --help
```

Command-line arguments are the canonical configuration. Slurm wrappers map
environment variables into explicit flags, print the resolved command, and
write the resolved run configuration beside training records.
