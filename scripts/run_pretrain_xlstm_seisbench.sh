#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_pretrain_xlstm_seisbench.sh [NUM_GPUS]
#
# Requires: export SEISBENCH_DATA (or defaults to $HOME/seis_data)

NUM_GPUS=${1:-${NUM_GPUS:-1}}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

: "${SEISBENCH_DATA:=$HOME/seis_data}"
: "${SEISBENCH_CACHE:=$SEISBENCH_DATA/seis_cache}"
export SEISBENCH_DATA SEISBENCH_CACHE

echo "[info] Using SEISBENCH_DATA=$SEISBENCH_DATA"

# Quick data sanity check
python scripts/verify_seisbench_data.py --base "$SEISBENCH_DATA" || true

EXPR=contrastive/xlstm_unet_seisbench

# Multi-dataset mix (enable full set once installed)
DATASETS="[ETHZ,GEOFON,STEAD,INSTANCE,MLAAPDE,Iquique,PNW,OBST2024]"

# Overrides for stability and parity
OVERRIDES=(
  dataset.dataset_name="$DATASETS"
  dataset.training_fraction=0.25
  trainer.strategy=ddp
  trainer.devices=$NUM_GPUS
  trainer.accelerator=gpu
  trainer.accumulate_grad_batches=1
  loader.num_workers=12
  loader.prefetch_factor=4
  model.only_masked=false
  encoder.bert_style=false
  train.gumbel_min_temperature=1.0
  train.gumbel_max_temperature=2.0
)

if command -v torchrun >/dev/null 2>&1; then
  torchrun --standalone --nproc_per_node=$NUM_GPUS simple_train.py experiment=$EXPR "${OVERRIDES[@]}"
else
  # Fallback to single-process; Lightning will spawn DDP workers
  python simple_train.py experiment=$EXPR "${OVERRIDES[@]}"
fi
