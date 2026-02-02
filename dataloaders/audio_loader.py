import os
import os.path
import torch
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import dataloaders.data_utils.costa_rica_utils as cu
import torchaudio
from torchaudio import load

from scipy.signal import decimate
from torch.utils.data import Dataset, DataLoader
from dataloaders.base import SequenceDataset
from evaluation.eval_utils import moving_average
from tqdm import tqdm

from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11

# use the command below to delete ableton live files from dataset (from root folder).
# test by leaving out the -delete
# find . -name "*.asd" -type f -delete
#


class_dict = {
    0: '808',
    1: 'Clap',
    2: 'Cymbal',
    3: 'Kick',
    4: 'Perc',
    5: 'Snare',
}


def list_wav_files(directory):
    # Use the '**' wildcard to include subdirectories
    wav_files = glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True)

    return wav_files


class AudioDataset(Dataset):
    def __init__(
            self,
            directory: str,
            sample_len: int = 8192,
            sample_rate: int = 44100,
            channels: int = 1,
            auto_reg: bool = True,
            return_labels: bool = False,
            train: str = 'train'
    ):
        super().__init__()
        self.directory = directory
        self.sample_len = sample_len
        self.sample_rate = sample_rate
        self.channels = channels
        self.auto_reg = auto_reg
        self.train = train
        self.return_labels = return_labels
        self.train_fraction = 0.9

        self.file_paths = []
        for label in class_dict.keys():
            fp = list_wav_files(os.path.join(self.directory, class_dict[label]))
            num_train_samples = int(len(fp) * self.train_fraction)
            if self.train == 'train':
                fp = fp[:num_train_samples]
            else:
                fp = fp[num_train_samples:]
            self.file_paths += zip(label * torch.ones(len(fp)), fp)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            sample, sr = load(self.file_paths[idx][1], normalize=True, channels_first=False)
        except:
            print(self.file_paths[idx][1])
        label = self.file_paths[idx][0].long()
        # hacky conversion mono to stereo and vice versa
        if self.channels == 1:
            if sample.dim() == 2:
                sample = sample[:, 0]
        else:
            assert self.channels == 2
            if sample.shape[1] == 1:
                sample = sample.squeeze(1)
                sample = torch.stack((sample, sample), dim=-1)

        if sr != self.sample_rate:
            sample = torchaudio.functional.resample(sample.transpose(0, 1), orig_freq=sr, new_freq=self.sample_rate).transpose(0, 1)

        if sample.dim() == 1:
            sample = sample.unsqueeze(-1)
        diff = self.sample_len + 1 - sample.shape[0]
        if diff >= 0:
            z = torch.zeros(diff, sample.shape[1])
            x_plus_one = torch.cat((sample, z))
        else:
            if self.train == 'train':
                start_idx = torch.randint(low=0, high=sample.shape[0] - self.sample_len - 1, size=(1,)).item()
            else:
                start_idx = 0
            x_plus_one = sample[start_idx:start_idx + self.sample_len + 1]

        x = x_plus_one[:self.sample_len]
        if self.auto_reg:
            y = x_plus_one[1:]
        else:
            y = label.unsqueeze(-1)

        if self.return_labels:
            return (x, label.unsqueeze(-1)), y
        else:
            return x, y


class AudioDatasetLit(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.dataset_train = AudioDataset(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            sample_rate=self.hparams.sample_rate,
            channels=self.hparams.channels,
            auto_reg=self.hparams.auto_reg,
            return_labels=self.hparams.return_labels,
            train='train'
        )
        self.dataset_test = AudioDataset(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            sample_rate=self.hparams.sample_rate,
            channels=self.hparams.channels,
            auto_reg=self.hparams.auto_reg,
            return_labels=self.hparams.return_labels,
            train='test'
        )
        self.d_data = self.dataset_train.channels
        self.split_train_val(self.hparams.val_split)
        self.num_classes = len(class_dict)


def audio_dataset_label_test():
    print('\nClassify\n')

    data_config = {
        'directory': 'data/audio',
        'sample_len': 80000,
        'sample_rate': 44100,
        'channels': 1,
        'auto_reg': False
    }
    loader_config = {
        'batch_size': 8,
        'num_workers': 0,
        'shuffle': True
    }
    dataset = AudioDataset(**data_config)
    loader = DataLoader(dataset, **loader_config)

    x, y = next(iter(loader))
    print('x.shape: ', x.shape)
    print('y.shape: ', y.shape)

    for i in range(x.shape[0]):
        plt.plot(x[i])
        plt.title(class_dict[y[i].item()])
        plt.show()
        plt.close()


def audio_dataset_autoreg_test():
    print('\nAuto Reg\n')

    data_config = {
        'directory': 'data/audio',
        'sample_len': 200,
        'sample_rate': 44100,
        'channels': 2,
        'auto_reg': True
    }
    loader_config = {
        'batch_size': 4,
        'num_workers': 0,
        'shuffle': True
    }
    dataset = AudioDataset(**data_config)
    loader = DataLoader(dataset, **loader_config)

    x, y = next(iter(loader))
    print('x.shape: ', x.shape)
    print('y.shape: ', y.shape)

    for i in range(x.shape[0]):
        plt.plot(x[i])
        plt.plot(y[i])
        # plt.title(class_dict[y[i].item()])
        plt.show()
        plt.close()


def audio_lit_test(auto_reg=False):
    print('\nAudio Lightning Test\n')
    print(f'testing auto-reg: {auto_reg}\n')

    data_dir = 'data/audio'
    data_config = {
        'sample_len': 44128,
        'sample_rate': 44100,
        'channels': 2,
        'auto_reg': auto_reg,
        'val_split': 0.1,
        'return_labels': True
    }
    loader_config = {
        'batch_size': 64,
        'num_workers': 0,
    }
    dataset = AudioDatasetLit(data_dir=data_dir, **data_config)
    loader = dataset.train_dataloader(**loader_config)

    x, y = next(iter(loader))
    x, labels = x
    print('x.shape: ', x.shape)
    print('y.shape: ', y.shape)
    print('labels.shape: ', labels.shape)
    print(f'num classes: {dataset.num_classes}')

    for i in range(4):
        plt.plot(x[i])
        if auto_reg:
            plt.plot(y[i])
            plt.title(class_dict[labels[i, 0].item()])
        else:
            plt.title(class_dict[y[i, 0].item()])
        plt.show()
        plt.close()

    for i, batch in enumerate(loader):
        x, y = batch
        x, labels = x
        print(torch.nn.functional.mse_loss(x, y))


if __name__ == '__main__':
    #audio_dataset_label_test()
    #audio_dataset_autoreg_test()
    audio_lit_test(True)
