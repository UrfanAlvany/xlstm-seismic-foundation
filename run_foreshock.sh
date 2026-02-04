#!/bin/bash
# Foreshock finetuning command - pretrained path is now in config file

export WANDB_MODE=offline

python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  trainer.devices=1 \
  trainer.strategy=auto \
  dataset.data_dir=/scicore/home/dokman0000/alvani0000/seis_data \
  dataset.batch_size=32 \
  trainer.precision=bf16-mixed \
  dataset.dimension_order=NWC 