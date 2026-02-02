import seisbench
import seisbench.models as sbm
import seisbench.generate as sbg

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod, ABC

# Allows to import this file in both jupyter notebook and code
try:
    from .augmentations import DuplicateEvent
except ImportError:
    from dataloaders.data_utils.seisbench_utils.augmentations import DuplicateEvent

from evaluation.eval_utils import load_checkpoint, get_pipeline_components, print_hparams, get_model_summary
from dataloaders.seisbench_auto_reg import get_eval_augmentations
from models.phasenet_wrapper import PhaseNetWrapper
import torch
import torch.nn as nn
from models.benchmark_models import SeisBenchModuleLit
from dataloaders.seisbench_auto_reg import phase_dict


class PhasePickerLit(SeisBenchModuleLit):
    def __init__(self, ckpt_path=None, avg_latent=False, random_init=False, norm_type='peak'):
        """
        Wrapper class to evaluate a phase picker. The encoder, model and decoder are loaded from the provided
        checkpoint. If random_init is True, the hparams of the checkpoint are used to set up a model, which is
        randomly initialized.
        :param ckpt_path: Path to a checkpoint directory or a specific checkpoint
        :param avg_latent: If true, the predict step returns the model output (without decoder) averaged over the
        sequence length of the provided local window.
        :param random_init: If true, a newly initialized model with the same architecture as the checkpoint
        will be returned
        """
        super().__init__()
        self.save_hyperparameters()
        self.avg_latent = avg_latent
        self.norm_type = norm_type
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load checkpoint
        pl_module, hparams, specific_ckpt = load_checkpoint(
            ckpt_path,
            location=device,
            return_path=True,
            simple=True,
            return_random_init=random_init
        )
        try:
            self.sample_len = hparams['dataset']['sample_len']
        except:
            print('Could not determine sample length from checkpoint file')
            self.sample_len = 4096

        self.model = pl_module.model
        self.encoder = pl_module.encoder
        self.decoder = pl_module.decoder

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.model(x, None)
        if self.avg_latent:
            # we want to evaluate the representation learned by the model and skip the decoder
            return x
        x = self.decoder(x, None)
        return x

    def shared_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def get_augmentations(self):
        return get_eval_augmentations(sample_len=self.sample_len, d_data=3, bits=0)

    def get_eval_augmentations(self):
        return get_eval_augmentations(sample_len=self.sample_len, d_data=3, bits=0, norm_type=self.norm_type)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch['X']
        window_borders = batch['window_borders']

        pred = self.forward(x)
        if self.avg_latent:
            m = torch.zeros(pred.shape[0], pred.shape[-1])
            for i in range(pred.shape[0]):
                start_sample, end_sample = window_borders[i]
                local_pred = pred[i, start_sample:end_sample, :]
                m[i] = torch.mean(local_pred, dim=0)

            # m = torch.mean(pred, dim=1)
            return m
        else:
            pred = F.softmax(pred, dim=-1)

            score_detection = torch.zeros(pred.shape[0])
            score_p_or_s = torch.zeros(pred.shape[0])
            p_sample = torch.zeros(pred.shape[0], dtype=int)
            s_sample = torch.zeros(pred.shape[0], dtype=int)

            for i in range(pred.shape[0]):
                start_sample, end_sample = window_borders[i]
                local_pred = pred[i, start_sample:end_sample, :]

                score_detection[i] = torch.max(1 - local_pred[:, -1])
                score_p_or_s[i] = torch.max(local_pred[:, 0]) / torch.max(local_pred[:, 1])
                p_sample[i] = torch.argmax(local_pred[:, 0])
                s_sample[i] = torch.argmax(local_pred[:, 1])
            return score_detection, score_p_or_s, p_sample, s_sample