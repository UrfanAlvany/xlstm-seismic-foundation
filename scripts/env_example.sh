#!/usr/bin/env bash
set -euo pipefail

# Example environment setup for pretraining/finetuning runs.
# Copy to env.sh and edit as needed.

# Base path containing SeisBench datasets as subfolders (ETHZ, GEOFON, STEAD, ...)
export SEISBENCH_DATA=${SEISBENCH_DATA:-"$HOME/seis_data"}
# Cache for SeisBench (optional)
export SEISBENCH_CACHE=${SEISBENCH_CACHE:-"$SEISBENCH_DATA/seis_cache"}

# W&B (optional)
export WANDB_PROJECT=${WANDB_PROJECT:-"final-seismology"}
export WANDB_MODE=${WANDB_MODE:-"online"}  # set to offline if needed

# PyTorch tuning
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}

# Distributed defaults (overridden by torchrun/SLURM)
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=1

echo "SEISBENCH_DATA=$SEISBENCH_DATA"
echo "SEISBENCH_CACHE=$SEISBENCH_CACHE"
echo "WANDB_PROJECT=$WANDB_PROJECT (mode=$WANDB_MODE)"

