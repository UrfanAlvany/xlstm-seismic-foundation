import os
import os.path
import glob
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import decimate

import matplotlib.pyplot as plt

import dataloaders.data_utils.costa_rica_utils as cu
from dataloaders.base import SequenceDataset
from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11
from dataloaders.tokenizers.bpe_tokenizer import BPETokenizer


class CostaRicaBPE(Dataset):
    def __init__(
            self,
            directory: str = None,
            sample_len: int = 2048,
            train: bool = True
    ):
        super().__init__()

        self.directory = directory
        self.sample_len = sample_len
        self.train = train

        self.tokenizer = BPETokenizer()
        self.tokenizer.load_vocab('bpe_vocab_4096_d100_train')

        self.data = torch.load(directory).long()

        total_num_samples = len(self.data)
        num_train_samples = int(total_num_samples * 0.95)

        if self.train:
            self.data = self.data[:num_train_samples]
        else:
            self.data = self.data[num_train_samples:]

        self.num_samples = len(self.data)
        self.num_slices = self.num_samples // self.sample_len

    def __len__(self):
        return self.num_slices

    def get_num_classes(self):
        return self.tokenizer.get_vocab_size()

    def __getitem__(self, idx):
        start_idx = torch.randint(low=0, high=self.num_samples - self.sample_len - 1, size=(1,)).item()
        stop_idx = start_idx + self.sample_len

        x_plus_one = self.data[start_idx:stop_idx + 1]
        return x_plus_one[:-1].unsqueeze(-1), x_plus_one[1:].unsqueeze(-1)

    def get_tokenizer(self):
        return self.tokenizer

    def decode(self, sequence):
        return self.tokenizer.decode(sequence)


class CostaRicaBPELightning(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.d_data = 1

        self.dataset_train = CostaRicaBPE(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            train=True
        )
        # TODO: fix sample length
        self.dataset_test = CostaRicaBPE(
            directory=self.data_dir,
            sample_len=81000,
            train=False
        )
        self.split_train_val(self.hparams.val_split)
        self.num_classes = self.dataset_train.dataset.get_num_classes()


def dataset_test():
    data_path = 'data/costa_rica/bpe_small/cr_small_bpe_encoded_ids.pt'
    sample_len = 16000

    data_config = {
        'sample_len': sample_len,
        'val_split': 0.1,
    }

    loader_config = {
        'batch_size': 2,
        'shuffle': True,
    }

    dataset = CostaRicaBPELightning(data_dir=data_path, **data_config)
    train_loader = dataset.train_dataloader(**loader_config)
    val_loader = dataset.val_dataloader(**loader_config)
    test_loader = dataset.test_dataloader(**loader_config)

    x_train, y_train = next(iter(train_loader))

    fig, ax = plt.subplots(figsize=(40, 10))
    ax.plot(x_train[0])
    plt.suptitle('x_train')
    plt.show()

    tokenizer = dataset.dataset_train.dataset.get_tokenizer()
    x_train_decoded = tokenizer.decode(x_train[0].long().squeeze()).squeeze()
    fig, ax = plt.subplots(figsize=(40, 10))
    ax.plot(x_train_decoded)
    plt.suptitle('x_train decoded')
    plt.show()


if __name__ == '__main__':
    dataset_test()
