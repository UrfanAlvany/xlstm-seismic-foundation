import os
import os.path
import torch
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import dataloaders.data_utils.costa_rica_utils as cu

from scipy.signal import decimate
from torch.utils.data import Dataset, DataLoader
from dataloaders.base import SequenceDataset
from evaluation.eval_utils import moving_average

from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11


class CostaRicaQuantized(Dataset):
    def __init__(
            self,
            directory: str = None,
            sample_len: int = 2048,
            downsample: int = 1,
            bits: int = 8,
            normalize_per_slice: bool = False,
            moving_average: bool = False,
            avg_window: int = 16,
            train: str = 'train'
    ):
        super().__init__()

        self.directory = directory
        self.sample_len = sample_len
        self.file_paths = glob.glob(os.path.join(self.directory, "*.pt"))
        self.downsample = downsample
        self.normalize_per_slice = normalize_per_slice
        self.moving_average = moving_average
        self.avg_window = avg_window
        self.bits = int(bits)
        self.train = train

        # number of days needed to extract a sample of the given length and sampling rate
        self.seq_len = (sample_len * downsample) // 8_640_000 + 1

        # extract data range and compute sqrt
        if not self.normalize_per_slice:
            if str(self.directory) == 'dataloaders/data/costa_rica/cove_rifo_15_16_hhz':
                print('Use precomputed data_max')
                self.data_max = 2431.3677220856575
            else:
                d_min, d_max = cu.find_data_min_and_max(self.directory)
                self.data_max = np.sqrt(max(abs(d_min), abs(d_max)))
            self.data_min = -self.data_max

        # extract metadata and sequences
        metadata = [cu.get_metadata(f) for f in self.file_paths]
        # TODO: make this variable
        metadata = cu.filter_channel(metadata, channels='HHZ')
        sorted_metadata = cu.sort_metadata_by_date(metadata)

        # use 95% as training and 5% as test examples
        # training will be further split into train and validation subsets
        self.num_train_examples = int(len(self.file_paths) * 0.95)
        self.num_val_examples = int(len(self.file_paths) * 0.10)
        self.file_paths = [f['path'] for f in sorted_metadata]
        if self.train == 'train':
            self.file_paths = self.file_paths[:self.num_train_examples - self.num_val_examples]
            sorted_metadata = sorted_metadata[:self.num_train_examples - self.num_val_examples]
        elif self.train == 'val':
            self.file_paths = self.file_paths[self.num_train_examples - self.num_val_examples: self.num_train_examples]
            sorted_metadata = sorted_metadata[self.num_train_examples - self.num_val_examples: self.num_train_examples]
        else:
            self.file_paths = self.file_paths[self.num_train_examples:]
            sorted_metadata = sorted_metadata[self.num_train_examples:]

        # extract sequences of consecutive recordings
        self.sequences = cu.extract_sequences_from_metadata_list(sorted_metadata)

        # arrange consecutive recordings in short sequences which can provide a sample of the desired length
        self.tuples = []
        for s in self.sequences:
            if len(s) >= self.seq_len:
                self.tuples += [s[i:i + self.seq_len] for i in range(len(s) - self.seq_len + 1)]

    def __len__(self):
        # length = sum(len(s) - self.seq_len + 1 for s in self.sequences if len(s) >= self.seq_len)
        return len(self.tuples)

    def __getitem__(self, idx):
        # load full recordings
        # start_time = time.time()
        data = []
        for entry in self.tuples[idx]:
            data.append(torch.load(entry['path']))
        data = torch.cat(data)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken load data: {elapsed_time:.6f} seconds")

        # start_time = time.time()
        # random sample
        start_idx = torch.randint(
            low=0,
            high=data.shape[-1] - (self.sample_len + 1) * self.downsample,
            size=(1,)
        ).item()
        stop_idx = start_idx + self.sample_len * self.downsample

        # downsample
        if self.downsample != 1:
            x_plus_one = decimate(data[start_idx:stop_idx + self.downsample], q=self.downsample)
            x_plus_one = torch.from_numpy(x_plus_one.copy()).float()
        else:
            x_plus_one = data[start_idx:stop_idx + 1].float()
        # squash data
        x_plus_one = torch.sqrt(torch.abs(x_plus_one)) * torch.sign(x_plus_one)
        # normalize
        if self.normalize_per_slice:
            x_plus_one = normalize_11_torch(x_plus_one)
        else:
            x_plus_one = normalize_11_torch(x_plus_one, d_min=self.data_min, d_max=self.data_max)

        if self.moving_average:
            x_plus_one = moving_average(x_plus_one, self.avg_window).squeeze()

        # quantize and encode data
        if self.bits > 0:
            encoded = quantize_encode(x_plus_one, self.bits)

            x = encoded[:-1]
            y = encoded[1:]
        else:
            x = x_plus_one[:-1].float()
            y = x_plus_one[1:].float()

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time taken to prepare data: {elapsed_time:.6f} seconds")
        return x.unsqueeze(-1), y.unsqueeze(-1)


class CostaRicaQuantizedLightning(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.d_data = 1

        try:
            normalize_per_slice = self.hparams.normalize_per_slice
        except:
            normalize_per_slice = False

        try:
            moving_average = self.hparams.moving_average
            avg_window = self.hparams.avg_window
        except:
            moving_average = False
            avg_window = 0

        self.dataset_train = CostaRicaQuantized(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            downsample=self.hparams.downsample,
            bits=self.hparams.bits,
            normalize_per_slice=normalize_per_slice,
            moving_average=moving_average,
            avg_window=avg_window,
            train='train'
        )
        self.dataset_val = CostaRicaQuantized(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            downsample=self.hparams.downsample,
            bits=self.hparams.bits,
            normalize_per_slice=normalize_per_slice,
            moving_average=moving_average,
            avg_window=avg_window,
            train='val'
        )
        self.dataset_test = CostaRicaQuantized(
            directory=self.data_dir,
            sample_len=int(8600000 / self.hparams.downsample),
            downsample=self.hparams.downsample,
            bits=self.hparams.bits,
            normalize_per_slice=normalize_per_slice,
            moving_average=moving_average,
            avg_window=avg_window,
            train='test'
        )
        # self.split_train_val(self.hparams.val_split)
        self.num_classes = 2 ** self.dataset_train.bits


class CostaRicaEncDec(CostaRicaQuantized):
    def __init__(self, *args, **kwargs):
        super(CostaRicaEncDec, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        x, y = CostaRicaQuantized.__getitem__(self, idx)
        return x, x


class CostaRicaEncDecLightning(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.d_data = 1

        self.dataset_train = CostaRicaEncDec(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            downsample=self.hparams.downsample,
            bits=self.hparams.bits,
            train='train'
        )
        self.dataset_val = CostaRicaEncDec(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            downsample=self.hparams.downsample,
            bits=self.hparams.bits,
            train='val'
        )
        self.dataset_test = CostaRicaEncDec(
            directory=self.data_dir,
            sample_len=self.hparams.sample_len,
            downsample=self.hparams.downsample,
            bits=self.hparams.bits,
            train='test'
        )
        # self.split_train_val(self.hparams.val_split)
        self.num_classes = 2 ** self.dataset_train.bits


def initialize_dataset_test():
    data_path = 'data/costa_rica/small_subset'
    sample_len = 958464
    downsample = 100
    dataset = CostaRicaQuantized(
        directory=data_path,
        sample_len=sample_len,
        downsample=downsample,
        bits=8,
        train=True
    )
    print(len(dataset))
    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    x, y = next(iter(dataloader))
    assert x.shape == y.shape
    print(f'x.shape: {x.shape}')

    for i in range(batch_size):
        plt.plot(x[i].cpu().detach().numpy())
        plt.title(f'training example {i} vanilla dataset')
        plt.show()

    print(dataset.tuples[0])


def simple_lightning_test():
    data_path = 'data/costa_rica/small_subset'
    sample_len = 400000
    downsample = 100
    bits = 8

    data_config = {
        'sample_len': sample_len,
        'val_split': 0.1,
        'downsample': downsample,
        'bits': bits,
        'moving_average': True,
        'avg_window': 31,
    }
    loader_config = {
        'batch_size': 1,
        'shuffle': False,
    }

    dataset = CostaRicaQuantizedLightning(data_dir=data_path, **data_config)
    train_loader = dataset.train_dataloader(**loader_config)
    val_loader = dataset.val_dataloader(**loader_config)
    test_loader = dataset.test_dataloader(**loader_config)

    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))
    x_test, y_test = next(iter(test_loader))

    print(f'x_train shape: {x_train.shape}')
    print(f'x_val shape: {x_val.shape}')
    print(f'x_test shape: {x_test.shape}')

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(x_train[0].cpu().detach().numpy())
    plt.title("Training example lightning")
    plt.show()

    plt.plot(x_val[0].cpu().detach().numpy())
    plt.title("Validation example lightning")
    plt.show()

    plt.plot(x_test[0].cpu().detach().numpy())
    plt.title("Test example lightning")
    plt.show()

    print(train_loader.dataset.tuples[0])
    for i, (x, y) in enumerate(train_loader):
        assert x.shape[1] == sample_len
        assert y.shape[1] == sample_len
    for i, (x, y) in enumerate(val_loader):
        assert x.shape[1] == sample_len
        assert y.shape[1] == sample_len
    for i, (x, y) in enumerate(test_loader):
        assert x.shape[1] == int(8600000 / downsample)
        assert y.shape[1] == int(8600000 / downsample)


if __name__ == '__main__':
    # initialize_dataset_test()
    simple_lightning_test()
