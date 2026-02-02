from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

from dataloaders.foreshock_aftershock_dataloader import prepare_foreshock_aftershock_dataloaders
from dataloaders.base import SequenceDataset


class ForeshockAftershockLitDataset(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            num_classes: int,
            batch_size: int,
            event_split_method: str,
            component_order: str,
            seed: int = 42,
            remove_class_overlapping_dates: bool = False,
            train_frac: float = 0.70,
            val_frac: float = 0.10,
            test_frac: float = 0.20,
            demean_axis: Optional[int] = -1,
            amp_norm_axis: Optional[int] = -1,
            amp_norm_type: str = "peak",
            num_workers: int = 8,
            dimension_order: str = "NCW",
            collator: Any = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir

        self.d_data = 3
        self.num_classes = num_classes

        loaders = prepare_foreshock_aftershock_dataloaders(
            data_dir=data_dir,
            num_classes=num_classes,
            batch_size=batch_size,
            event_split_method=event_split_method,
            component_order=component_order,
            seed=seed,
            remove_class_overlapping_dates=remove_class_overlapping_dates,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            demean_axis=demean_axis,
            amp_norm_axis=amp_norm_axis,
            amp_norm_type=amp_norm_type,
            num_workers=num_workers,
            dimension_order=dimension_order,
            collator=collator
        )

        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]

    def train_dataloader(self, **kwargs):
        return self.train_loader

    def val_dataloader(self, **kwargs):
        return self.val_loader

    def test_dataloader(self, **kwargs):
        return self.test_loader