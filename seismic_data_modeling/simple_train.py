import datetime
import uuid
import os
import threading
import time
import traceback
import pickle

import hydra
import omegaconf
import re
import math

import psutil
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, BaseFinetuning
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import torch
import numpy as np
import pandas as pd
from torch.optim import Optimizer

from tasks.encoders import instantiate_encoder, load_encoder_from_file, instantiate_encoder_simple
from tasks.decoders import instantiate_decoder, load_decoder_from_file, instantiate_decoder_simple
from tasks.task import task_registry
from dataloaders.base import SeisbenchDataLit
from dataloaders.foreshock_aftershock_lit import ForeshockAftershockLitDataset

from torch.utils.data import DataLoader

from utils.config_utils import instantiate
from utils import registry
from utils.optim_utils import print_optim, add_optimizer_hooks
from omegaconf import DictConfig, OmegaConf
from seisbench.util import worker_seeding
import seisbench
import logging

# ignore INFO level logging
seisbench.logger.setLevel(logging.WARNING)

# Stronger reproducibility across dataloader workers and modules
pl.seed_everything(42, workers=True)
torch.manual_seed(42)
np.random.seed(42)


class ModelFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=0):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.model)
        self.freeze(pl_module.encoder)

    def finetune_function(self, pl_module, current_epoch, optimizer) -> None:
        if current_epoch == self._unfreeze_at_epoch:
            print(f'Unfreezing model at epoch {current_epoch}')
            # Unfreeze backbone/model
            self.unfreeze_and_add_param_group(
                modules=pl_module.model,
                optimizer=optimizer,
                train_bn=True
            )
            # Unfreeze encoder
            self.unfreeze_and_add_param_group(
                modules=pl_module.encoder,
                optimizer=optimizer,
                train_bn=True
            )
            # Optional: lower LR for newly unfrozen params to reduce validation spikes
            try:
                lr_mult = float(pl_module.hparams.train.get('unfrozen_lr_mult', None))
            except Exception:
                lr_mult = None
            if lr_mult is not None and lr_mult > 0 and lr_mult < 1.0:
                try:
                    base_lr = float(optimizer.param_groups[0].get('lr', 0.0))
                    # Last two groups belong to the just-unfrozen encoder and model
                    for g in optimizer.param_groups[-2:]:
                        g['lr'] = base_lr * lr_mult
                    print(f'[finetune] Applied unfrozen LR multiplier {lr_mult} -> new LR {base_lr * lr_mult:.2e} for unfrozen groups')
                except Exception:
                    print('[finetune] Could not adjust LR for unfrozen groups; proceeding with base LR')


class SimpleSeqModel(pl.LightningModule):
    # Class-level switch to skip pretraining init during exact restore
    SKIP_PRETRAINED_INIT: bool = False

    def __init__(self, config, d_data: int = 3, skip_pretrained_init: bool = False):
        super().__init__()
        # Save only the config to preserve Lightning's expected hparams; skip flag is runtime-only
        self.save_hyperparameters(config)
        self.d_data = d_data

        self.l2_norm = config.train.get('l2', False)
        self.random_sample_len = config.train.get('random_sample_len', False)

        # Be defensive: ensure encoder sub-config exists for exact restore paths
        try:
            OmegaConf.set_struct(config, False)
            if not hasattr(config, 'encoder') or getattr(config.encoder, '_name_', None) is None:
                # Provide defaults consistent with training
                config.encoder = OmegaConf.create({
                    '_name_': 'conv-down-encoder-contrastive',
                    'kernel_size': 3,
                    'n_layers': 2,
                    'dim': 256,
                    'stride': 2,
                })
        except Exception:
            pass
        finally:
            try:
                OmegaConf.set_struct(config, True)
            except Exception:
                pass

        # Honor both local flag and class-level toggle for exact restore
        _skip = bool(skip_pretrained_init or getattr(SimpleSeqModel, 'SKIP_PRETRAINED_INIT', False))

        if config.model.get('pretrained', None) is not None and not _skip:
            # load pretrained model
            print('\nLoading pretrained model\n')

            # extract checkpoint path
            ckpt_path = config.model.pretrained

            # check if the model should be randomly initialized (for sanity checks)
            rand_init = config.model.get('rand_init', False)
            print(f'model random initialization: {rand_init}')

            # look for updated model parameters
            # for example: updated dropout value for fine-tuning
            if config.get('model_update', None) is not None:
                update_configs = config.model_update

                # load model from checkpoint
                ckpt_tuple = load_checkpoint(
                    ckpt_path,
                    updated_model_config=update_configs,
                    d_data=d_data,
                    rand_init=rand_init
                )
                if ckpt_tuple is None:
                    print(f"Pretrained checkpoint not found: {ckpt_path}. Proceeding with fresh init.")
                    ckpt = None
                else:
                    ckpt, _ = ckpt_tuple
            else:
                ckpt_tuple = load_checkpoint(
                    checkpoint_path=ckpt_path,
                    d_data=d_data,
                    rand_init=rand_init,
                )
                if ckpt_tuple is None:
                    print(f"Pretrained checkpoint not found: {ckpt_path}. Proceeding with fresh init.")
                    ckpt = None
                else:
                    ckpt, _ = ckpt_tuple

            # extract main model from checkpoint
            if ckpt is not None:
                # If the target model is a backbone (e.g., xlstm-unet), initialize it
                # and load only the backbone weights from the pretrained wrapper.
                model_cfg = OmegaConf.create(OmegaConf.to_container(self.hparams.model, resolve=True))
                with omegaconf.open_dict(model_cfg):
                    if 'pretrained' in model_cfg:
                        del model_cfg['pretrained']
                    if 'rand_init' in model_cfg:
                        del model_cfg['rand_init']

                target_name = getattr(model_cfg, '_name_', None)
                pretrained_model = ckpt.model
                # Most pretraining uses ContrastiveModel with attribute `._net` holding the backbone
                if target_name == 'xlstm-unet' and hasattr(pretrained_model, '_net'):
                    print('[finetune] Initializing new xlstm-unet backbone and loading weights from pretrained wrapper _net.')
                    self.model = instantiate(registry.model, model_cfg)
                    try:
                        self.model.load_state_dict(pretrained_model._net.state_dict(), strict=False)
                    except Exception as e:
                        print(f'[finetune] Warning: could not fully load backbone weights: {e}')
                else:
                    # Fallback: reuse the checkpoint model as-is
                    self.model = pretrained_model
            else:
                # sanitize model config to avoid passing pretrained/rand_init to constructor
                model_cfg = OmegaConf.create(OmegaConf.to_container(self.hparams.model, resolve=True))
                with omegaconf.open_dict(model_cfg):
                    if 'pretrained' in model_cfg:
                        del model_cfg['pretrained']
                    if 'rand_init' in model_cfg:
                        del model_cfg['rand_init']
                self.model = instantiate(registry.model, model_cfg)
            # freeze model parameters
            if config.model.get('freeze', None) is not None and config.model.get('freeze', False):
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

            if config.train.get('only_final_layer', False):
                num_layers_to_train = config.train.get('num_layers', 1)
                self.model.fix_all_but_last_layer(num_layers=num_layers_to_train)

            # save parameters for L2 norm
            if self.l2_norm:
                # TODO: check if there is a better way to clone the model
                self.l2_lambda = config.train.get('l2_lambda', 0.1)
                ref_ckpt, _ = load_checkpoint(config.model.pretrained, d_data=d_data)
                self.reference_model = ref_ckpt.model.eval()
                for param in self.reference_model.parameters():
                    param.requires_grad = False
        else:
            # initialize new model, sanitize config
            model_cfg = OmegaConf.create(OmegaConf.to_container(self.hparams.model, resolve=True))
            with omegaconf.open_dict(model_cfg):
                if 'pretrained' in model_cfg:
                    del model_cfg['pretrained']
                if 'rand_init' in model_cfg:
                    del model_cfg['rand_init']
            self.model = instantiate(registry.model, model_cfg)

        try:
            d_model = self.model.d_model
        except:
            print('could not infer d_model from model')
            d_model = 0

        freeze_encoder = config.encoder.get('freeze', False)
        encoder_config = self.hparams.encoder
        if config.encoder.get('freeze', None) is not None:
            del encoder_config.freeze

        if config.encoder.get('pretrained', None) is not None and 'ckpt' in locals() and ckpt is not None:
            print('\nLoading pretrained encoder\n')
            self.encoder = ckpt.encoder
        else:
            print('\nInitialize new encoder\n')
            self.encoder = instantiate_encoder_simple(encoder_config, d_data=self.d_data, d_model=d_model)

        if freeze_encoder:
            print('Freezing encoder\n')
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if config.decoder.get('pretrained', None) is not None and 'ckpt' in locals() and ckpt is not None:
            print('\nLoading pretrained decoder\n')
            self.decoder = ckpt.decoder
            if config.decoder.get('freeze', None) is not None and config.decoder.get('freeze', False):
                self.decoder.eval()
                for param in self.decoder.parameters():
                    param.requires_grad = False
        else:
            self.decoder = instantiate_decoder_simple(self.hparams.decoder, d_data=self.d_data, d_model=d_model)

        if config.train.get("disable_pretraining", False):
            print('\nTurning off pretraining mode for model and encoder\n')
            self.model.pretraining = False
            self.encoder.pretraining = False

        # Optional: disable TFLA/Triton mLSTM kernels at runtime (force PyTorch fallback)
        if config.train.get('disable_mlstm_kernels', False):
            try:
                # Patch any mLSTMCell backend_fn to official chunkwise_simple
                from xlstm.blocks.mlstm.backends import chunkwise_simple
                from xlstm.blocks.mlstm.cell import mLSTMCell
                from models.xlstm_unet import xLSTMUNetBackbone, UNetConfig
                # If ContrastiveModel with xLSTM UNet backbone, its net is under ._net
                net = getattr(self.model, '_net', None)
                rebuilt = False
                # Safer: rebuild xLSTM backbone without kernels and load weights
                if isinstance(net, xLSTMUNetBackbone):
                    try:
                        old_cfg = net.cfg
                        new_cfg = UNetConfig(**{
                            **vars(old_cfg),
                            'enable_tflakernels': False,
                            'mlstm_backend': 'parallel',
                        })
                        new_net = xLSTMUNetBackbone(new_cfg).to(next(self.model.parameters()).device)
                        new_net.load_state_dict(net.state_dict(), strict=False)
                        self.model._net = new_net
                        rebuilt = True
                        print('[kernels] Rebuilt xLSTMUNetBackbone without TFLA kernels (parallel backend).')
                    except Exception as e:
                        print(f'[kernels] Rebuild without kernels failed: {e}; falling back to backend_fn patching.')
                if not rebuilt:
                    # Fallback: patch cells to chunkwise_simple
                    def _chunk_call_factory(chunk_size: int):
                        def _chunk_call(**kw):
                            return chunkwise_simple(chunk_size=chunk_size, return_last_state=False, **kw)
                        return _chunk_call
                    patched = 0
                    for m in self.model.modules():
                        if isinstance(m, mLSTMCell):
                            chunk_sz = 128
                            try:
                                if hasattr(net, 'cfg') and hasattr(net.cfg, 'chunk_size'):
                                    chunk_sz = int(net.cfg.chunk_size)
                            except Exception:
                                pass
                            m.backend_fn = _chunk_call_factory(chunk_sz)
                            patched += 1
                    if patched > 0:
                        print(f'[kernels] Disabled TFLA kernels; patched {patched} mLSTM cells to chunkwise_simple')
            except Exception as e:
                print(f'[kernels] Could not disable TFLA kernels at runtime: {e}')

        self.task = instantiate(task_registry, self.hparams.task)
        self.criterion = self.task.loss
        self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.quantized_pretraining = hasattr(self.model, 'quantize') and getattr(self.model, 'quantize') and hasattr(self.model, 'pretraining') and getattr(self.model, 'pretraining')

    def forward(self, batch, batch_idx=None):
        masked = False
        if isinstance(batch, dict):
            x = batch['X']
            y = batch['y'] if 'y' in batch else None # for wav2vec, we dont need y
            # Masked pretraining provides (x, mask) as a list/tuple
            if isinstance(x, (list, tuple)):
                masked = True
                x, mask = x
        else:
            x, y = batch

        if self.random_sample_len:
            sample_len = min(x.shape[1], int(16 * torch.randint(180, 512, (1,))))
            start_idx = torch.randint(0, x.shape[1] - sample_len, (1,))
            x = x[:, start_idx:start_idx + sample_len, :]
            if y is not None:
                y = y[:, start_idx:start_idx + sample_len, :]

        # Keep a view of the raw waveform (post any cropping) for decoders that need it
        raw_for_decoder = x

        # encode
        x = self.encoder(x)
        # For contrastive pretraining, pass through the full tuple (x_masked, x_target, mask_indices)
        # Otherwise, unwrap tuple outputs (e.g., bidir-autoreg-encoder returning (x, tokens))
        if isinstance(x, (tuple, list)):
            try:
                task_name = self.hparams.task['_name_'] if isinstance(self.hparams.task, dict) else self.hparams.task._name_
            except Exception:
                task_name = 'default'
            if task_name != 'contrastive':
                x = x[0]

        # forward pass
        x, _ = self.model(x, None)

        # decode (for contrastive pretraining, decoder is usually dummy and will pass through)
        # Some heads may expect both features and raw input (e.g., concat-raw variant)
        if hasattr(self.decoder, 'expects_raw') and getattr(self.decoder, 'expects_raw'):
            x = self.decoder((x, raw_for_decoder), None)
        else:
            x = self.decoder(x, None)

        if masked:
            return x[mask], y[mask]

        return x, y

    def _l2_norm(self):
        # compute l2 norm between current model weights and reference (pretrained) model weights
        l2_norm = 0
        for param, ref_param in zip(self.model.parameters(), self.reference_model.parameters()):
            # |param - ref_param|^2 / 2
            l2_norm += (param - ref_param).norm(2)
        return l2_norm

    def _step_with_metrics(self, batch, batch_idx, prefix='train'):
        x, y = self.forward(batch, batch_idx)
        
        
        if self.quantized_pretraining:
            # Contrastive pretraining (optionally with quantizer/diversity penalty)
            metrics = {}
            # Delegate full tuple to criterion; it handles 5- or 6-tuples
            # and applies diversity_lambda if provided via task.loss config.
            diversity_loss = None
            if isinstance(x, (tuple, list)) and len(x) >= 6:
                diversity_loss = x[5]
            contrastive_loss = self.criterion(x, y)
            metrics['contrastive_loss'] = contrastive_loss
            if diversity_loss is not None:
                metrics['diversity_loss'] = diversity_loss
                # Derive codebook perplexity from diversity ratio when quantizer exists
                try:
                    q = getattr(self.model, 'quantizer', None)
                    if q is not None and hasattr(q, 'num_groups') and hasattr(q, 'num_vars'):
                        num_codevectors = float(q.num_groups) * float(q.num_vars)
                        metrics['codebook_perplexity'] = num_codevectors * (1.0 - diversity_loss)
                        # Also log SeisLM-style all-frames perplexity and diversity for comparability
                        if hasattr(q, 'last_perplexity_all') and q.last_perplexity_all is not None:
                            metrics['codebook_perplexity_all'] = q.last_perplexity_all
                            metrics['diversity_loss_all'] = 1.0 - (float(q.last_perplexity_all) / num_codevectors)
                except Exception:
                    pass
            loss = contrastive_loss

        else:
            metrics = self.metrics(x, y)

            if prefix == 'train':
                data_loss = self.criterion(x, y)
                if self.l2_norm:
                    l2_loss = self._l2_norm()
                    metrics['data_loss'] = data_loss
                    metrics['l2_norm'] = l2_loss
                    loss = data_loss + self.l2_lambda * l2_loss
                else:
                    loss = data_loss
            else:
                loss = self.loss_val(x, y)

        metrics['loss'] = loss
        metrics = {f'{prefix}/{metric}': val for metric, val in metrics.items()}

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        if self.quantized_pretraining:
            return loss, contrastive_loss, diversity_loss
        else:
            return loss

    def training_step(self, batch, batch_idx=None):
        if self.quantized_pretraining:
            # Cosine anneal Gumbel temperature (seisLM-style)
            try:
                t_min = float(self.hparams.train.get('gumbel_min_temperature', 0.5))
                t_max = float(self.hparams.train.get('gumbel_max_temperature', 2.0))
                # Resolve total anneal steps robustly (works when max_steps is -1/null)
                total_steps = int(getattr(self.trainer, 'max_steps', 0))
                # Treat negative or zero as "unset" so we fall back properly
                if total_steps <= 0:
                    total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0))
                if total_steps <= 0:
                    # Fallback: steps_per_epoch * max_epochs (account for grad accumulation)
                    try:
                        dl = self.trainer.fit_loop.epoch_loop._data_source.dataloader()  # type: ignore
                        steps_per_epoch = len(dl)
                        acc = int(getattr(self.trainer, 'accumulate_grad_batches', 1))
                        if acc > 1:
                            steps_per_epoch = max(1, steps_per_epoch // acc)
                        total_steps = steps_per_epoch * int(max(1, getattr(self.trainer, 'max_epochs', 1)))
                    except Exception:
                        total_steps = 0

                if total_steps and hasattr(self, 'trainer'):
                    ratio = min(1.0, max(0.0, self.global_step / max(1, total_steps)))
                    temp = t_min + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (t_max - t_min)
                    if hasattr(self.model, 'quantizer') and (self.model.quantizer is not None):
                        self.model.quantizer.temperature = temp
                        self.log('trainer/gumbel_temperature', temp, prog_bar=False, on_step=True, on_epoch=False)
            except Exception:
                pass
            loss, contrastive_loss, diversity_loss = self._step_with_metrics(batch, batch_idx, prefix='train')
            # logging
            loss_epoch = {
                'trainer/loss': loss,
                'trainer/contrastive_loss': contrastive_loss,
                'trainer/diversity_loss': diversity_loss,
                'trainer/epoch': self.current_epoch
            }
            # Log estimated codebook perplexity from diversity loss if quantizer present
            try:
                q = getattr(self.model, 'quantizer', None)
                if q is not None and hasattr(q, 'num_groups') and hasattr(q, 'num_vars') and diversity_loss is not None:
                    num_codevectors = float(q.num_groups) * float(q.num_vars)
                    codebook_perplexity = num_codevectors * (1.0 - float(diversity_loss))
                    self.log('trainer/codebook_perplexity', codebook_perplexity, prog_bar=False, on_step=True, on_epoch=False)
                    # Also log all-frames variants for SeisLM-style comparison
                    if hasattr(q, 'last_perplexity_all') and q.last_perplexity_all is not None:
                        self.log('trainer/codebook_perplexity_all', q.last_perplexity_all, prog_bar=False, on_step=True, on_epoch=False)
                        self.log('trainer/diversity_loss_all', 1.0 - (float(q.last_perplexity_all) / num_codevectors), prog_bar=False, on_step=True, on_epoch=False)
            except Exception:
                pass
        else:
            loss = self._step_with_metrics(batch, batch_idx, prefix='train')

            # logging
            loss_epoch = {'trainer/loss': loss, 'trainer/epoch': self.current_epoch}
        # Avoid per-step progress bar updates to prevent excessive console I/O on Slurm
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        return loss

    def validation_step(self, batch, batch_idx=None):
        loss = self._step_with_metrics(batch, batch_idx, prefix='val')
        return loss

    def test_step(self, batch, batch_idx=None):
        loss = self._step_with_metrics(batch, batch_idx, prefix='test')
        return loss

    def configure_optimizers(self):
        # Optional param grouping hooks
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Base params
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        # Robust optimizer instantiation (support fused=True on newer torch, fallback otherwise)
        opt_cfg = OmegaConf.create(OmegaConf.to_container(self.hparams.optimizer, resolve=True))
        opt_name = opt_cfg.get('_name_', 'adamw')
        try:
            optimizer = instantiate(registry.optimizer, opt_cfg, params)
        except TypeError as e:
            msg = str(e)
            # Fallback if 'fused' not supported in current torch
            fallback = opt_cfg.copy()
            with omegaconf.open_dict(fallback):
                if 'fused' in fallback:
                    del fallback['fused']
                fallback['foreach'] = True
            optimizer = instantiate(registry.optimizer, fallback, params)
            print("[info] AdamW(fused) unsupported; falling back to foreach=True.")

        print(f"Optimizer: {optimizer}")
        try:
            defaults = getattr(optimizer, 'defaults', {})
            print(f"[optim] defaults: fused={defaults.get('fused', None)} foreach={defaults.get('foreach', None)}")
        except Exception:
            pass

        # Param groups with custom hyperparams
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
        print("Hyperparameter groups", hps)
        for hp in hps:
            group_params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": group_params, **self.hparams.optimizer, **hp})

        # (No additional discriminative LR param group by default)

        keys = set(k for hp in hps for k in hp.keys())
        print_optim(optimizer, keys)
        print('\n\n')

        # Scheduler
        if "scheduler" not in self.hparams:
            return optimizer

        # Dynamically compute training steps if requested/missing (SeisLM-style)
        sched_cfg = OmegaConf.create(OmegaConf.to_container(self.hparams.scheduler, resolve=True))
        name = sched_cfg.get('_name_', '')
        # If num_training_steps not provided, derive from trainer
        if (sched_cfg.get('num_training_steps') in (None, 0, 'null')):
            total_steps = None
            try:
                # Lightning provides an estimate after setup
                total_steps = int(getattr(self.trainer, 'estimated_stepping_batches', 0))
            except Exception:
                total_steps = None
            if not total_steps or total_steps <= 0:
                # Fallback to max_epochs * batches_per_epoch if available
                try:
                    if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs:
                        # Best-effort: use length of train dataloader
                        bpe = len(self.trainer.fit_loop.epoch_loop._data_source.dataloader())  # type: ignore
                        total_steps = int(self.trainer.max_epochs * bpe)
                except Exception:
                    total_steps = None
            if total_steps and total_steps > 0:
                with omegaconf.open_dict(sched_cfg):
                    sched_cfg['num_training_steps'] = total_steps
                    wf = sched_cfg.get('warmup_fraction', None)
                    if wf is not None and sched_cfg.get('num_warmup_steps') in (None, 0, 'null'):
                        sched_cfg['num_warmup_steps'] = int(max(1, round(float(wf) * total_steps)))
                        # Remove helper key not accepted by HF schedulers
                        if 'warmup_fraction' in sched_cfg:
                            del sched_cfg['warmup_fraction']
        # If still missing required ints, set tiny defaults to avoid runtime errors in smoke runs
        with omegaconf.open_dict(sched_cfg):
            # Only fill defaults if truly missing (None/'null'), not when explicitly set to 0
            if sched_cfg.get('num_training_steps') in (None, 'null'):
                sched_cfg['num_training_steps'] = 100
            if sched_cfg.get('num_warmup_steps') in (None, 'null'):
                # default warmup at 10%
                sched_cfg['num_warmup_steps'] = max(1, int(0.1 * int(sched_cfg['num_training_steps'])))
            if 'warmup_fraction' in sched_cfg:
                del sched_cfg['warmup_fraction']

        lr_scheduler = instantiate(registry.scheduler, sched_cfg, optimizer)
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int = 0) -> None:
        """
        Manual gradient clipping compatible with fused optimizers.
        - Unscales grads if fp16 scaler is in use
        - Clips by max_norm if configured in either config.optim.clip_grad_norm,
          config.train.clip_grad_norm, or trainer.gradient_clip_val
        """
        try:
            # Determine max_norm from config or trainer
            max_norm = None
            # Prefer explicit optim.clip_grad_norm if present
            if hasattr(self.hparams, 'optim') and self.hparams.optim is not None and 'clip_grad_norm' in self.hparams.optim:
                max_norm = float(self.hparams.optim.clip_grad_norm)
            elif hasattr(self.hparams, 'train') and self.hparams.train is not None and 'clip_grad_norm' in self.hparams.train:
                max_norm = float(self.hparams.train.clip_grad_norm)
            elif getattr(getattr(self, 'trainer', None), 'gradient_clip_val', 0.0):
                max_norm = float(self.trainer.gradient_clip_val)

            if not max_norm or max_norm <= 0:
                return

            # Unscale if using GradScaler
            scaler = getattr(getattr(self.trainer, 'strategy', None), 'precision_plugin', None)
            scaler = getattr(scaler, 'scaler', None)
            if scaler is not None:
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass

            params_with_grad = [p for p in self.parameters() if p.grad is not None]
            if params_with_grad:
                torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm, foreach=False)
        except Exception:
            if getattr(self, 'global_step', 0) == 0:
                print('[warn] Gradient clipping skipped due to exception.')


def create_trainer(config):
    print('\n\n', '*' * 32)
    print('Initializing trainer')
    print('*' * 32, '\n')

    current_date = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    # Ensure unique version/run-id across concurrent array tasks started in the same second
    job_id = os.environ.get('SLURM_JOB_ID')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    run_suffix = os.environ.get('RUN_SUFFIX')
    suffix_parts = []
    if job_id:
        suffix_parts.append(f"j{job_id}")
    if task_id:
        suffix_parts.append(f"t{task_id}")
    if run_suffix:
        suffix_parts.append(str(run_suffix))
    # If nothing to disambiguate, add a short random token
    if not suffix_parts:
        suffix_parts.append(uuid.uuid4().hex[:8])
    run_version = f"{current_date}__{'_'.join(suffix_parts)}"
    # model_name = config.model['_name_']
    experiment_name = config.experiment_name

    # setup logger
    if config.train.get("ckpt_path", None) is not None:
        # Keep checkpoint-derived version as base, but still disambiguate parallel runs
        base_from_ckpt = config.train.get("ckpt_path").split('/')[-1]
        run_version = f"{base_from_ckpt}__{uuid.uuid4().hex[:8]}"

    logger_t = TensorBoardLogger(
        save_dir='final_seismology_logs',
        name='final-seismology',
        default_hp_metric=False,
        version=run_version,
    )

    logger_wab = WandbLogger(
        project='final-seismology',
        save_dir='final_seismology_logs',
        name=experiment_name,
        version=run_version,
    )

    loggers = [logger_t, logger_wab]

    # monitor learning rate
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        # log_momentum=True,
    )
    top_checkpoints = ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        dirpath=f'wandb_logs/mars/{run_version}/checkpoints',
        filename="callback-{epoch:d}-{step:d}",
    )

    # Also keep best validation accuracy snapshot for downstream comparison
    # (only for tasks that log accuracy, skip for phase picking)
    task_name = config.task.get('_name_', '')
    if 'phase' not in task_name.lower():
        best_acc_checkpoint = ModelCheckpoint(
            save_top_k=1,
            monitor="val/accuracy",
            mode="max",
            dirpath=f'wandb_logs/mars/{run_version}/checkpoints',
            filename="bestacc-{epoch:d}-{step:d}",
        )
        callbacks = [lr_monitor, top_checkpoints, best_acc_checkpoint]
    else:
        # Phase picking tasks don't log val/accuracy
        callbacks = [lr_monitor, top_checkpoints]

    last_checkpoints = ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=f'wandb_logs/mars/{run_version}/checkpoints',
    )

    callbacks.append(last_checkpoints)

    unfreeze_at_epoch = config.train.get('unfreeze_at_epoch', 0)
    if unfreeze_at_epoch > 0:
        print(f'\nadding freeze unfreeze callback for epoch {unfreeze_at_epoch}\n')
        unfreeze_callback = ModelFreezeUnfreeze(unfreeze_at_epoch=unfreeze_at_epoch)
        callbacks.append(unfreeze_callback)

    # initialize trainer
    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **config.trainer)
    return trainer


def _extract_step_number(filename):
    match = re.search(r'step=(\d+)\.ckpt$', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint(
        checkpoint_path: str,
        location: str = 'cpu',
        return_path: bool = False,
        updated_model_config: OmegaConf = None,
        d_data: int = 3,
        rand_init: bool = False,
) -> tuple[SimpleSeqModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu.
    If no checkpoint is specified, the folder is searched for checkpoints and the one with the highest
    step number is returned.

    :param checkpoint_path: path to checkpoint file. The hparams file is extracted automatically
    :param location: device to load to. Defaults to cpu as Lightning handles the devices
    :param return_path: If true, returns the path to the specific checkpoint
    :param updated_model_config: Config with updated parameters. We initially load the configurations from
    the checkpoint and, if provided, use the updated_model_config to overwrite certain parameters. Mostly
    used for fine-tuning e.g. a way to add dropout
    :param d_data: data dimensionality. Passed to the constructor of the model.
    :param rand_init: If True, the model architecture is determined by the provided checkpoint but the trained
    weights are NOT loaded. Instead, the model is returned as is with default initialization.
    :return: LightningSequenceModel, hparams
    """
    # Resolve symlinks to robustly locate adjacent hparams.yaml
    resolved_ckpt_path = os.path.realpath(checkpoint_path)

    if not resolved_ckpt_path.endswith('.ckpt'):
        # the path does not directly lead to checkpoint, we search for checkpoints in directory
        all_files = []

        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(resolved_ckpt_path):
            for file in files:
                file_path = os.path.join(root, file)
                step_number = _extract_step_number(file)
                if step_number is not None:
                    all_files.append((step_number, file_path))
        all_files.sort(key=lambda x: x[0])
        resolved_ckpt_path = all_files[-1][1]

    # Compute hparams path from the resolved checkpoint location (two levels up)
    parts = resolved_ckpt_path.split('/')
    # Default attempt: sibling under wandb_logs/mars/<date>/hparams.yaml
    hparam_path = '/'.join(parts[:-2]) + '/hparams.yaml'

    # Fallback: if not present, try TensorBoard logger location
    if not os.path.isfile(hparam_path):
        try:
            # Find project root and date folder near wandb_logs/mars/<date>
            if 'wandb_logs' in parts:
                idx = parts.index('wandb_logs')
                # pattern: .../wandb_logs/mars/<date>/checkpoints/file.ckpt
                # get date folder safely
                date_folder = parts[idx + 2] if len(parts) > idx + 2 else None
                project_root = '/'.join(parts[:idx])
                if date_folder:
                    alt = os.path.join(project_root, 'final_seismology_logs', 'final-seismology', date_folder, 'hparams.yaml')
                    if os.path.isfile(alt):
                        hparam_path = alt
        except Exception:
            pass

    if not os.path.isfile(resolved_ckpt_path):
        print('NO CHECKPOINT FOUND')
        return None
    if not os.path.isfile(hparam_path):
        print('NO HPARAM FOUND')
        hparams = None
    else:
        with open(hparam_path, 'r') as f:
            hparams = yaml.safe_load(f)

    print(f'Loading hparams from {resolved_ckpt_path}')
    if hparams is not None:
        name = hparams['experiment_name']
        print(f'Experiment name: {name}')

    # initialize model based on loaded resp. updated configuration
    if updated_model_config is not None:
        # create and update omega config
        full_config = OmegaConf.create(hparams) if hparams is not None else OmegaConf.create({})
        full_config.model.update(updated_model_config)
        model = SimpleSeqModel(full_config, d_data=d_data)
        # model.load_state_dict(torch.load(checkpoint_path, map_location=location)['state_dict'])
    else:
        # Fallback: if hparams.yaml was not found, construct a minimal config so caller's config prevails
        base_cfg = OmegaConf.create(hparams) if hparams is not None else OmegaConf.create({})
        model = SimpleSeqModel(base_cfg, d_data=d_data)

    # load state dict or return randomly initialized model
    if not rand_init:
        print(f'Loading state dict from checkpoint')
        model.load_state_dict(torch.load(resolved_ckpt_path, map_location=location, weights_only=False)['state_dict'])
        # model = SimpleSeqModel.load_from_checkpoint(checkpoint_path, map_location=location)
    else:
        print(f'Returning randomly initialized model')

    # optionally return full path
    if return_path:
        return model, hparams, resolved_ckpt_path
    else:
        return model, hparams


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf) -> None:
    try:
        trainer = create_trainer(config)

        print('*' * 32)
        print('CONFIGURATION')
        print(OmegaConf.to_yaml(config))
        print('*' * 32, '\n\n')

        print(f'Cuda available: {torch.cuda.is_available()}')

        # Instantiate dataset directly; avoid passing duplicate 'preload' kw to prevent Hydra partial conflicts
        dataset = instantiate(registry.dataset, config.dataset)

        if isinstance(dataset, SeisbenchDataLit):
            print('\nInitializing Seisbench Loaders\n')
            train_loader = DataLoader(
                dataset.dataset_train,
                shuffle=True,
                worker_init_fn=worker_seeding,
                **config.loader
            )
            val_loader = DataLoader(
                dataset.dataset_val,
                shuffle=False,
                worker_init_fn=worker_seeding,
                **config.loader
            )
        elif isinstance(dataset, ForeshockAftershockLitDataset):
            print('\nInitializing Foreshock-Aftershock Loaders\n')
            train_loader = dataset.train_loader
            val_loader = dataset.val_loader
        else:
            print('\nInitializing Standard Loaders\n')
            train_loader = DataLoader(
                dataset.dataset_train,
                shuffle=True,
                **config.loader
            )
            val_loader = DataLoader(
                dataset.dataset_val,
                shuffle=False,
                **config.loader
            )

        d_data = dataset.d_data

        if config.train.get("ckpt_path", None) is not None:
            print(f'\nLoading checkpoint from {config.train.ckpt_path}\n')
            model, hparams, ckpt_path = load_checkpoint(config.train.ckpt_path, return_path=True)
        else:
            model = SimpleSeqModel(config, d_data=d_data)

        if config.train.get('seq_warmup', None) is not None:
            seq_warmup = True
            final_sample_len = config.train.get('final_sample_len', 4096)
            final_batch_size = config.train.get('final_batch_size', 128)
            num_epochs_warmup = config.train.get('num_epochs_warmup', 2)
            min_seq_len = config.train.get('min_seq_len', 256)
        else:
            seq_warmup = False

        # ===================================================================================================
        # SANITY CHECK
        weights = []
        for param in model.model.parameters():
            weights.extend(param.view(-1).tolist())  # Flatten and convert to list
            if len(weights) > 100:
                break
        encoder_weights = []
        for param in model.encoder.parameters():
            encoder_weights.extend(param.view(-1).tolist())  # Flatten and convert to list
            if len(encoder_weights) > 100:
                break
        print('\n\n', "First 100 model weights:", weights[:100])
        print("First 100 encoder weights:", encoder_weights[:100], '\n\n')
        # ===================================================================================================

        summary = ModelSummary(model, max_depth=1)
        print('\n', '*' * 32, '\n')
        print('SUMMARY')
        print(summary)

        print('\n', '*' * 32, '\n')
        print('ENCODER')
        print(model.encoder)
        print('\n', '*' * 32, '\n')

        print('DECODER')
        print(model.decoder)
        print('*' * 32, '\n\n')

        ############################
        # FIT THE MODEL - CLEAN VERSION FOR xLSTM RESEARCH
        ############################

        # Optional exact resume: if train.resume_fit is True, pass ckpt_path to Trainer.fit so that
        # optimizer/scheduler/global_step and callbacks resume exactly.
        fit_ckpt_path = None
        if config.train.get('resume_fit', False):
            fit_ckpt_path = config.train.get('fit_ckpt_path', None)
            if fit_ckpt_path is None:
                fit_ckpt_path = config.train.get('ckpt_path', None)

        if seq_warmup is False:
            # Standard training (recommended for bidirectional xLSTM)
            print('🚀 Starting standard xLSTM training (bidirectional compatible)')
            if fit_ckpt_path is not None:
                print(f"[resume] Resuming fit from checkpoint: {fit_ckpt_path}")
                trainer.fit(model, train_loader, val_loader, ckpt_path=fit_ckpt_path)
            else:
                trainer.fit(model, train_loader, val_loader)
        else:
            # Advanced sequence length warmup training
            print('\n', '*' * 32, '\n')
            print('Start Training With Sequence Length Warmup')
            print('\n', '*' * 32, '\n')

            # compute batch sizes and sequence lengths
            sample_lengths = []
            batch_sizes = []

            b_size = final_batch_size
            s_len = final_sample_len
            while s_len >= min_seq_len:
                sample_lengths.append(s_len)
                batch_sizes.append(b_size)
                b_size = b_size * 2
                s_len = s_len // 2
            batch_sizes = list(reversed(batch_sizes))
            sample_lengths = list(reversed(sample_lengths))

            # print batch sizes and sequence lengths
            for i in range(len(batch_sizes)):
                print(f'Batch Size: {batch_sizes[i] :3d}, Sample Length: {sample_lengths[i]:6d}')

            # Main training loop
            for i in range(len(batch_sizes)):
                print('\n', '*' * 32, '\n')
                if i == len(batch_sizes) - 1:
                    n_epochs = config.trainer.max_epochs - (i + 1) * num_epochs_warmup
                else:
                    n_epochs = num_epochs_warmup
                print(f'Train for {n_epochs} epochs. Batch_size {batch_sizes[i]}, seq_len: {sample_lengths[i]}')
                print('\n', '*' * 32, '\n')

                dataset.sample_len = sample_lengths[i]
                dataset.setup()

                train_loader = DataLoader(
                    dataset.dataset_train,
                    shuffle=True,
                    worker_init_fn=worker_seeding,
                    pin_memory=config.loader.pin_memory,
                    num_workers=config.loader.num_workers,
                    batch_size=batch_sizes[i],
                )
                val_loader = DataLoader(
                    dataset.dataset_val,
                    shuffle=False,
                    worker_init_fn=worker_seeding,
                    pin_memory=config.loader.pin_memory,
                    num_workers=config.loader.num_workers,
                    batch_size=batch_sizes[i],
                )

                if i > 0:
                    model, _ = load_checkpoint('final_seismology_logs/final-seismology/' + trainer.logger.version)
                    
                if i == len(batch_sizes) - 1:
                    trainer.fit_loop.max_epochs = config.trainer.max_epochs
                else:
                    trainer.fit_loop.max_epochs = (i + 1) * num_epochs_warmup
                    
                trainer.fit(model, train_loader, val_loader)

        print('\n', '*' * 32, '\n')
        print('✅ TRAINING COMPLETED SUCCESSFULLY!')
        print('*' * 32, '\n')
        
    except Exception as e:
        print(f'\n❌ ERROR OCCURRED: {e}')
        traceback.print_exc()
    finally:
        # FIXME: workaround to prevent wandb from blocking the termination of runs on sciCORE slurm
        def aux(pid, timeout=60):
            time.sleep(timeout)
            print(f"Program did not terminate successfully, killing process tree")
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()

        shutdown_cleanup_thread = threading.Thread(target=aux, args=(os.getpid(), 60), daemon=True)
        shutdown_cleanup_thread.start()


if __name__ == '__main__':
    main()
