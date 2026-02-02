import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optimizers
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataloaders.base import SequenceDataset


class MNISTdataset(SequenceDataset):
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )
        self.setup()

    def setup(self):
        self.d_data = 1
        self.dataset_train = datasets.MNIST(root=self.data_dir, train=True, transform=self.transform, download=True)
        self.dataset_test = datasets.MNIST(root=self.data_dir, train=False, transform=self.transform, download=True)
        self.split_train_val(self.hparams.val_split)
