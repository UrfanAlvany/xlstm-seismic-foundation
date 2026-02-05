#!/bin/bash
# Foreshock fine-tuning command (repo-root friendly).
#
# Prereqs:
#   export SEIS_DATA_DIR=/path/to/foreshock/data
#   python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=/path/to/pretrained.ckpt

export WANDB_MODE=offline

python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  trainer.devices=1 \
  trainer.strategy=auto \
  dataset.data_dir=${SEIS_DATA_DIR:?set SEIS_DATA_DIR} \
  dataset.batch_size=32 \
  trainer.precision=bf16-mixed \
  dataset.dimension_order=NWC 
