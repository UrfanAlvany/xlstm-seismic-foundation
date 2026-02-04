#!/usr/bin/env bash
set -euo pipefail

# Example fine-tuning run for ETHZ phase picking.
# Adjust config/overrides as needed.

NUM_GPUS=${1:-${NUM_GPUS:-1}}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${SEISBENCH_DATA:=$HOME/seis_data}"
: "${SEISBENCH_CACHE:=$SEISBENCH_DATA/seis_cache}"
export SEISBENCH_DATA SEISBENCH_CACHE

echo "[info] Using SEISBENCH_DATA=$SEISBENCH_DATA"

# Choose a finetune config that uses a large phase-pick decoder
EXPR=phase_picking/fine_tune_hydra_contrastive_ethz

OVERRIDES=(
  trainer.strategy=ddp
  trainer.devices=$NUM_GPUS
  trainer.accelerator=gpu
  loader.num_workers=12
  loader.prefetch_factor=4
  decoder._name_=large-phase-pick-decoder
  train.clip_grad_norm=1.0
)

if command -v torchrun >/dev/null 2>&1; then
  torchrun --standalone --nproc_per_node=$NUM_GPUS simple_train.py experiment=$EXPR "${OVERRIDES[@]}"
else
  python simple_train.py experiment=$EXPR "${OVERRIDES[@]}"
fi

