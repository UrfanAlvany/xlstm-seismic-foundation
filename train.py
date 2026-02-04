import datetime
import os
import threading
import time
import traceback

import hydra
import omegaconf
import re

import psutil
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloaders.MNISTdataloader import MNISTdataset
from dataloaders.simple_waveform import SineWaveLightningDataset
from models.simple_test_models import ConvNet
from tasks.encoders import instantiate_encoder, load_encoder_from_file
from tasks.decoders import instantiate_decoder, load_decoder_from_file
from tasks.task import task_registry
from dataloaders.base import SeisbenchDataLit


from utils.config_utils import instantiate
from utils import registry
from omegaconf import DictConfig, OmegaConf


class LightningSequenceModel(pl.LightningModule):
    def __init__(self, config, preload_data: bool = False):
        super().__init__()
        self.save_hyperparameters(config)
        if config.encoder.get("pretrained", None) is not None:
            self.use_pretrained_encoder = config.encoder.get("pretrained")
        else:
            self.use_pretrained_encoder = False
        if config.decoder.get("pretrained", None) is not None:
            self.use_pretrained_decoder = config.decoder.get("pretrained")
        else:
            self.use_pretrained_decoder = False

            # initialize dataset:
        if preload_data:
            self.dataset = instantiate(registry.dataset, self.hparams.dataset, preload=preload_data)
        else:
            self.dataset = instantiate(registry.dataset, self.hparams.dataset)

        self.setup()

    def setup(self, stage=None):
        self.model = instantiate(registry.model, self.hparams.model)

        if not self.use_pretrained_encoder:
            self.encoder = instantiate_encoder(self.hparams.encoder, self.dataset, self.model)
        else:
            encoder_path = self.hparams.encoder.path
            print(f'\nLoading Encoder from {encoder_path}\n')
            self.encoder, encoder_hparams = load_encoder_from_file(encoder_path, self.dataset, self.model)

        if not self.use_pretrained_decoder:
            self.decoder = instantiate_decoder(self.hparams.decoder, self.dataset, self.model)
        else:
            decoder_path = self.hparams.decoder.path
            print(f'\nLoading Decoder from {decoder_path}\n')
            self.decoder, decoder_hparams = load_decoder_from_file(decoder_path, self.dataset, self.model)

        # if self.hparams.model._name_ == "conv_net":
        #    self.model = ConvNet(self.hparams.model.in_channels, self.hparams.model.img_size)

        self.task = instantiate(task_registry, self.hparams.task, dataset=self.dataset, model=self.model)
        self.criterion = self.task.loss
        self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

        self._initialize_state()

    def _initialize_state(self):
        self._state = None

    def _on_epoch_start(self):
        self._initialize_state()

    def forward(self, batch, batch_idx=None):
        if isinstance(self.dataset, SeisbenchDataLit):
            x = batch['X']
            # Some pretraining tasks (e.g., contrastive) do not provide 'y'
            y = batch.get('y', None)
        else:
            x, y = batch

        # encode
        if self.use_pretrained_encoder:
            with torch.no_grad():
                x, y = self.encoder.prepare_data(x, y)
        else:
            x = self.encoder(x)

        # forward pass
        x, state = self.model(x, state=self._state)
        self._state = state

        # decode
        if self.decoder is not None:
            if self.use_pretrained_decoder:
                with torch.no_grad():
                    x = self.decoder(x, state=state)
            else:
                x = self.decoder(x, state=state)
        return x, y

    def step(self, x_t):
        x_t = self.encoder(x_t)
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t = self.decoder.step(x_t, state=self._state)
        return x_t

    def _step_with_metrics(self, batch, batch_idx, prefix='train'):
        x, y = self.forward(batch, batch_idx)

        if 'context_len' in self.hparams.task:
            args = {'context_len': self.hparams.task.context_len}
        else:
            args = {}

        if prefix == 'train':
            loss = self.criterion(x, y, **args)
        else:
            loss = self.loss_val(x, y, **args)

        metrics = self.metrics(x, y)
        metrics['loss'] = loss
        metrics = {f'{prefix}/{metric}': val for metric, val in metrics.items()}

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return loss

    def on_train_epoch_start(self) -> None:
        self._on_epoch_start()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self._on_epoch_start()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self._on_epoch_start()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

    def training_step(self, batch, batch_idx):
        loss = self._step_with_metrics(batch, batch_idx, prefix='train')

        # logging
        loss_epoch = {'trainer/loss': loss, 'trainer/epoch': self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step_with_metrics(batch, batch_idx, prefix='val')
        print(f'Validation Loss: {loss:.4f}')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step_with_metrics(batch, batch_idx, prefix='test')
        print(f'Test Loss: {loss:.4f}')
        return loss

    def predict_step(self, batch, batch_idx):
        #print(f'predicting on {batch_idx}')
        x, y = self.forward(batch, batch_idx)
        return x, y

    def configure_optimizers(self):
        # Set zero weight decay for some params

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        args = self.hparams.optimizer.copy()
        with omegaconf.open_dict(args):
            del args['_name_']
        optimizer = instantiate(registry.optimizer, self.hparams.optimizer, params)
        print(f"Optimizer: {optimizer}")

        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def val_dataloader(self):
        return self.dataset.val_dataloader(**self.hparams.loader)

    def test_dataloader(self):
        return self.dataset.test_dataloader(**self.hparams.loader)


def create_trainer(config):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    model_name = config.model['_name_']
    experiment_name = config.experiment_name

    # setup logger
    if config.train.get("ckpt_path", None) is not None:
        current_date = config.train.get("ckpt_path").split('/')[-1]

    logger_t = TensorBoardLogger(
        save_dir='final_seismology_logs',
        name='final-seismology',
        default_hp_metric=False,
        version=current_date,
    )

    logger_wab = WandbLogger(
        project='final-seismology',
        save_dir='final_seismology_logs',
        name=experiment_name,
        version=current_date,
    )

    loggers = [logger_t, logger_wab]

    # monitor learning rate
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        # log_momentum=True,
    )
    top_checkpoints = ModelCheckpoint(
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        dirpath=f'final_seismology_logs/final-seismology/{current_date}/checkpoints',
        filename="callback-{epoch:d}-{step:d}",
    )

    last_checkpoints = ModelCheckpoint(
        save_top_k=2,
        monitor="epoch",
        mode="max",
        dirpath=f'final_seismology_logs/final-seismology/{current_date}/checkpoints',
    )

    # initialize trainer
    trainer = pl.Trainer(logger=loggers, callbacks=[lr_monitor, top_checkpoints, last_checkpoints], **config.trainer)
    return trainer


def plot_and_save_training_examples(model: LightningSequenceModel, trainer: pl.Trainer, num_examples: int = 3):
    print('\n', 'Plotting and saving training examples', '\n')
    log_dir = trainer.logger.log_dir

    # create directory for plots
    save_dir = os.path.join(log_dir, 'training_examples')

    # Check if the 'plots' directory exists
    if not os.path.exists(save_dir):
        # If it does not exist, create it
        os.makedirs(save_dir)
        print(f"Directory 'training_examples' created at {save_dir}")
    else:
        print(f"Directory 'training_examples' already exists at {save_dir}")

    if isinstance(model.dataset, SeisbenchDataLit):
        x = next(iter(model.dataset.train_dataloader(batch_size=num_examples)))['X']
    else:
        x, _ = next(iter(model.dataset.train_dataloader(batch_size=num_examples)))

    for i in range(x.shape[0]):
        fig, ax = plt.subplots(figsize=(40, 10))
        ax.plot(x[i].squeeze().cpu().numpy())
        plt.suptitle(f'Training example {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_example_{i}.png'))
        plt.close()


def _extract_step_number(filename):
    match = re.search(r'step=(\d+)\.ckpt$', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint(checkpoint_path: str, location: str = 'cpu', return_path: bool = False) -> tuple[
    LightningSequenceModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu.
    If no checkpoint is specified, the folder is searched for checkpoints and the one with the highest
    step number is returned.
    :param checkpoint_path: path to checkpoint file. The hparams file is extracted automatically
    :return: LightningSequenceModel, hparams
    """
    if not checkpoint_path.endswith('.ckpt'):
        # the path does not directly lead to checkpoint, we search for checkpoints in directory
        all_files = []

        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                file_path = os.path.join(root, file)
                step_number = _extract_step_number(file)
                if step_number is not None:
                    all_files.append((step_number, file_path))
        all_files.sort(key=lambda x: x[0])
        checkpoint_path = all_files[-1][1]

    hparam_path = '/'.join(checkpoint_path.split('/')[:-2]) + '/hparams.yaml'

    if not os.path.isfile(checkpoint_path):
        print('NO CHECKPOINT FOUND')
        return None
    if not os.path.isfile(hparam_path):
        print('NO HPARAM FOUND')
        hparams = None
    else:
        with open(hparam_path, 'r') as f:
            hparams = yaml.safe_load(f)

    print(f'Loading checkpoint from {checkpoint_path}')
    if hparams is not None:
        name = hparams['experiment_name']
        print(f'Experiment name: {name}')

    model = LightningSequenceModel.load_from_checkpoint(checkpoint_path, map_location=location)
    if return_path:
        return model, hparams, checkpoint_path
    else:
        return model, hparams


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf) -> None:
    try:
        print('*' * 32)
        print('CONFIGURATION')
        print(OmegaConf.to_yaml(config))
        print('*' * 32, '\n\n')

        print(f'cuda available: {torch.cuda.is_available()}')

        preload = config.dataset.get('_name_') in registry.preloadable_datasets

        if config.train.get("ckpt_path", None) is not None:
            print(f'loading checkpoint from {config.train.ckpt_path}')
            model, hparams, ckpt_path = load_checkpoint(config.train.ckpt_path, return_path=True)
            # fixes for sequence length warmup
            # reload the dataset, updates the sample length
            if preload:
                model.dataset = instantiate(registry.dataset, config.dataset, preload=preload)
            else:
                model.dataset = instantiate(registry.dataset, config.dataset)
            # update the batch size
            model.hparams.loader.batch_size = config.loader.batch_size
            trainer = create_trainer(config)
        else:
            trainer = create_trainer(config)
            model = LightningSequenceModel(config, preload_data=preload)

        plot_and_save_training_examples(model, trainer, num_examples=16)

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

        if config.train.get("ckpt_path", None) is not None:
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)

        print('\n', '*' * 32, '\n')
        print('DONE')
        print('\n', '*' * 32, '\n')
    except Exception as e:
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
