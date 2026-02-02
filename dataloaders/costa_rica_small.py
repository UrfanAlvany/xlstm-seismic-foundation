import os
import os.path
import torch
import glob
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataloaders.base import SequenceDataset
from dataloaders.data_utils.costa_rica_utils import get_metadata, format_label
from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11


class CostaRicaSmall(Dataset):
    """
    Most basic dataset for Costa Rica waveforms. Will only return one channel. The items returned
    consist of (x, y), both with dimensions [sample_len, 1]. The sequences are shifted by one element.
    The resulting batches have dimensions [batch_size, sample_len, 1].
    """

    def __init__(self, file: str = None,
                 sample_len: int = 2048,
                 normalize_const: float = 20000.0,
                 downsample: int = 1,
                 quantize: bool = False,
                 bits: int = 8,
                 train: bool = True):
        """
        :param file: directory containing waveforms
        :param sample_len: length of returned sequence, should be smaller than approx 8'500'000
        :param normalize_const: data is divided by this constant
        :param downsample: only every n-th element is returned (default = 1, so every sample is returned)
        :param train: if true, the first 95% of files will be used for training and validation
                        else: the remaining 5% of files will be used for testing
        """
        super().__init__()

        self.file = file
        self.sample_len = sample_len
        self.file_paths = glob.glob(os.path.join(self.file, '*.pt'))
        self.normalize_const = 1.0 / float(normalize_const)
        self.downsample = int(downsample)
        self.quantize = quantize
        self.bits = int(bits)
        self.train = train

        self.data_min = -np.sqrt(783285)
        self.data_max = np.sqrt(783285)

        year_day_list = []
        for file in self.file_paths:
            name = file.split('/')[-1]
            meta = get_metadata(name)
            year_day_list.append((meta['year'], meta['day'], file))

        # sort list by year and day
        year_day_list = sorted(year_day_list, key=lambda x: x[0] + x[1])

        self.file_paths = [f for y, d, f in year_day_list]
        self.num_train_examples = int(len(self.file_paths) * 0.95)

        if self.train:
            self.file_paths = self.file_paths[:self.num_train_examples]
            year_day_list = year_day_list[:self.num_train_examples]
        else:
            self.file_paths = self.file_paths[self.num_train_examples:]
            year_day_list = year_day_list[self.num_train_examples:]

        # construct tuples of consecutive recordings
        self.tuple_list = []
        for a, b in zip(year_day_list[:-1], year_day_list[1:]):
            if a[0] == b[0] and int(a[1]) == int(b[1]) - 1:
                self.tuple_list.append((a[-1], b[-1]))
            # else:
            #    self.tuple_list.append((a[-1], None))
            #    self.tuple_list.append((b[-1], None))

        # mapping from label strings to file paths
        self.label_to_path = {}
        for file_path in self.file_paths:
            label = format_label(file_path.split('/')[-1])
            self.label_to_path[label] = file_path

    def __len__(self):
        return len(self.tuple_list)

    def __getitem__(self, idx):
        # file_path = self.file_paths[idx]
        # data = torch.load(file_path)
        f_1 = self.tuple_list[idx][0]
        f_2 = self.tuple_list[idx][1]

        data_1 = torch.load(f_1)
        if f_2 is not None:
            data_2 = torch.load(self.tuple_list[idx][1])
            data = torch.cat((data_1, data_2))
        else:
            data = data_1

        if not self.quantize:
            data = data * self.normalize_const

        start_idx = np.random.randint(0, data.shape[-1] - (self.sample_len + 1) * self.downsample)
        stop_idx = start_idx + self.sample_len * self.downsample

        # expand sequences to accommodate 'channel' (data is 1d)
        if not self.quantize:
            x = data[start_idx:stop_idx:self.downsample].type(torch.FloatTensor)
            y = data[start_idx + self.downsample: stop_idx + self.downsample: self.downsample].type(torch.FloatTensor)
            return x.unsqueeze(-1), y.unsqueeze(-1)
        else:
            # quantize the data
            x_plus_one = decimate(data[start_idx:stop_idx + self.downsample], q=self.downsample)
            x_plus_one = torch.from_numpy(x_plus_one.copy()).float()
            # x_plus_one = data[start_idx:stop_idx + self.downsample:self.downsample].type(torch.FloatTensor)
            x_plus_one = torch.sqrt(torch.abs(x_plus_one)) * torch.sign(x_plus_one)
            x_plus_one = normalize_11_torch(x_plus_one, d_min=self.data_min, d_max=self.data_max)
            encoded = quantize_encode(x_plus_one, self.bits)
            x = encoded[:-1]
            y = encoded[1:]
            return x.unsqueeze(-1), y.unsqueeze(-1)


class CostaRicaSmallLighting(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.d_data = 1

        self.dataset_train = CostaRicaSmall(
            self.data_dir,
            sample_len=self.hparams.sample_len,
            train=True,
            downsample=self.hparams.downsample,
            normalize_const=self.hparams.normalize_const,
            quantize=self.hparams.quantize,
            bits=self.hparams.bits,
        )
        self.dataset_test = CostaRicaSmall(
            self.data_dir,
            sample_len=self.hparams.sample_len,
            train=False,
            downsample=self.hparams.downsample,
            normalize_const=self.hparams.normalize_const,
            quantize=self.hparams.quantize,
            bits=self.hparams.bits,
        )
        self.split_train_val(self.hparams.val_split)

        if self.dataset_train.dataset.quantize:
            self.num_classes = 2 ** self.dataset_train.dataset.bits


def plot_first_examples(dataloader, num_examples=3, plot_len=None):
    fig, ax = plt.subplots(num_examples, 1)
    fig.suptitle('first three examples costa rica small')
    x, y = next(iter(dataloader))
    if plot_len is None:
        # set plot_len to sequence length
        plot_len = x.shape[1]
    for i, a in enumerate(ax):
        a.plot(x[i, :plot_len, 0], label='x')
        a.plot(y[i, :plot_len, 0], label='y')
    plt.legend()
    plt.show()


def run_tests():
    data_path = 'data/costa_rica/small_subset'
    sample_length = 16_000
    batch_size = 32
    expected_num_train_files = 50
    expected_num_test_files = 3
    normalize_by = 20_000
    downsample = 500

    train_dataset = CostaRicaSmall(file=data_path, sample_len=sample_length, normalize_const=normalize_by,
                                   downsample=downsample, train=True)
    test_dataset = CostaRicaSmall(file=data_path, sample_len=sample_length, normalize_const=normalize_by,
                                  downsample=downsample, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('#' * 32)
    print('Basic tests for CostaRicaSmall dataset')
    print('#' * 32, '\n')

    train_file_paths = train_dataset.file_paths
    test_file_paths = test_dataset.file_paths
    print(f'number of files in train dataset: {len(train_file_paths)}, expected: {expected_num_train_files}')
    print(f'number of files in test dataset: {len(test_file_paths)}, expected: {expected_num_test_files}')

    x, y = next(iter(train_loader))

    print('x and y shapes:', x.shape, y.shape)

    print(f'x shape: {x.shape}, y shape: {y.shape}, expected shapes: {batch_size}, {sample_length}, {1}')

    # plot 3 examples, x and y should be shifted by one position
    plot_first_examples(train_loader, plot_len=100)
    plot_first_examples(test_loader, plot_len=sample_length, num_examples=len(test_loader.dataset))

    test_sanity_check = False
    for f in train_file_paths:
        test_sanity_check = test_sanity_check or f in test_file_paths

    print(f'test files found in training files: {test_sanity_check}')

    val_split = 0.1
    data_config = {
        'sample_len': sample_length,
        'val_split': val_split,
        'downsample': downsample,
        'normalize_const': normalize_by,
        'quantize': True,
        'bits': 8
    }
    loader_config = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True}
    dataset_lightning = CostaRicaSmallLighting(data_dir=data_path, **data_config)
    train_loader_pl = dataset_lightning.train_dataloader(**loader_config)
    val_loader_pl = dataset_lightning.val_dataloader(**loader_config)
    test_loader_pl = dataset_lightning.test_dataloader(**loader_config)

    print('\n', '\n', '#' * 32)
    print('Basic tests for CostaRicaSmall Lightning dataset')
    print('#' * 32, '\n')

    print(f'number of training samples: {len(train_loader_pl.dataset)}, '
          f'expected: {int(expected_num_train_files * (1 - val_split))}')
    print(f'number of validation samples: {len(val_loader_pl.dataset)}, '
          f'expected: {int(expected_num_train_files * val_split)}')
    print(f'number of test samples: {len(test_loader_pl.dataset)}, '
          f'expected: {expected_num_test_files}')

    print(f'dataset downsampling: {dataset_lightning.dataset_train.dataset.downsample}, expected: {downsample}')
    print(
        f'dataset normalize_const: {dataset_lightning.dataset_train.dataset.normalize_const}, expected: {1.0 / normalize_by}')

    abs_max = 0
    data_abs_max = None
    for i, data in enumerate(train_loader_pl):
        x, y = data
        temp_max = torch.max(torch.abs(x))
        print(i, temp_max)
        if temp_max > abs_max:
            abs_max = temp_max
            data_abs_max = x
    print(f'abs_max: {abs_max}')
    plt.plot(data_abs_max[0, :, 0])
    plt.suptitle(f'frame with maximum amplitude: {abs_max}')
    plt.show()


if __name__ == '__main__':
    run_tests()
