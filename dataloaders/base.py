from pathlib import Path

import torch
import pytorch_lightning as pl
from seisbench.util import worker_seeding


class SequenceDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else None

        self.save_hyperparameters()

        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.d_data = 1
        self.num_classes = None

        self.init()

    def init(self):
        pass

    def setup(self):
        pass

    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(getattr(self, 'seed', 42)),
        )

    def _dataloader(self, dataset, shuffle, **kwargs):
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **kwargs)

    def train_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_train, shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_val, shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_test, shuffle=False, **kwargs)


class SeisbenchDataLit(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.d_data = 1
        self.num_classes = None

        self.init()

    def init(self):
        pass

    def setup(self):
        pass

    def _dataloader(self, dataset, shuffle, **kwargs):
        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, worker_init_fn=worker_seeding, **kwargs)

    def train_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_train, shuffle=True, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_val, shuffle=False, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_test, shuffle=False, **kwargs)
