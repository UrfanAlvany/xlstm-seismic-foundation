import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloaders.base import SequenceDataset
from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11
import omegaconf


class SineWaveformDataset(Dataset):
    def __init__(self, data_dir: str = None,
                 sample_len: int = 2048,
                 min_freq: int = 80,
                 max_freq: int = 2_000,
                 num_frequencies: int = 0,
                 num_random_amplitudes: int = 1,
                 noise_amplitude: float = 0.1,
                 quantize: bool = False,
                 bits: int = 8,
                 overwrite_existing_file: bool = False):
        super().__init__()
        self.file = data_dir
        self.sample_len = sample_len
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_frequencies = num_frequencies
        self.num_random_amplitudes = num_random_amplitudes
        self.noise_amplitude = noise_amplitude
        self.quantize = quantize
        self.bits = bits
        self.overwrite_existing_file = overwrite_existing_file

        self.waveform_len = None
        self.data = None
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.setup()

    def setup(self):
        if os.path.isfile(self.file) and not self.overwrite_existing_file:
            self.data = np.load(self.file)
        else:
            self._generate_dataset()
        self.waveform_len = self.data.shape[1]

    def _generate_dataset(self):
        num_examples_per_freq = 1 + self.num_random_amplitudes
        len_waveform = 4 * self.sample_len
        f_sampling = 16_000

        if self.num_frequencies == 1:
            frequencies = [self.min_freq]
        else:
            frequencies = np.linspace(self.min_freq, self.max_freq, self.num_frequencies)

        t = np.linspace(0, len_waveform / f_sampling, len_waveform)

        data = np.zeros((len(frequencies) * num_examples_per_freq, len_waveform))

        for i in range(data.shape[0]):
            f = frequencies[i // num_examples_per_freq]
            if i % num_examples_per_freq == 0:
                amp = 1.0
            else:
                amp = np.random.random()
            data[i, :] = np.sin(2 * np.pi * f * t) * amp + self.noise_amplitude * np.random.normal(size=len_waveform)

        # data = np.zeros((num_waveforms, len_waveform))
        # for i in range(num_waveforms):
        #    data[i, :] = np.sin(2 * np.pi * 100 * t) + 0.05 * np.random.normal(size=len_waveform)
        if self.quantize:
            data = normalize_11(data)
            data = torch.from_numpy(data).float()
            data = quantize_encode(data, bits=self.bits)
            data = data.numpy()

        print(f'saving waveform data to {self.file}')
        try:
            np.save(self.file, data)
        except:
            print('COULD NOT SAVE WAVEFORM DATA TO FILE')

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        start_idx = np.random.randint(0, self.waveform_len - self.sample_len - 1)

        x = self.data[idx, start_idx:start_idx + self.sample_len]
        y = self.data[idx, start_idx + 1: start_idx + self.sample_len + 1]

        if self.quantize:
            return x[:, np.newaxis], y[:, np.newaxis]
        else:
            return x.astype(np.float32)[:, np.newaxis], y.astype(np.float32)[:, np.newaxis]


class SineWaveLightningDataset(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.d_data = 1
        self.dataset_train = SineWaveformDataset(
            data_dir=self.hparams.data_dir,
            sample_len=self.hparams.sample_len,
            min_freq=self.hparams.min_freq,
            max_freq=self.hparams.max_freq,
            num_frequencies=self.hparams.num_frequencies,
            num_random_amplitudes=self.hparams.num_random_amplitudes,
            noise_amplitude=self.hparams.noise_amplitude,
            quantize=self.hparams.quantize,
            bits=self.hparams.bits,
            overwrite_existing_file=self.hparams.overwrite_existing_file
        )
        self.split_train_val(self.hparams.val_split)

        # if the dataset is quantized, we must specify the number of classes
        if self.dataset_train.dataset.quantize:
            self.num_classes = 2 ** self.dataset_train.dataset.bits


def plot_examples(dataloader, num_examples=3, title=''):
    data_iter = iter(dataloader)
    # plot 3 examples
    num_examples = 3
    fig, ax = plt.subplots(num_examples)
    x, y = next(data_iter)
    for i, a in enumerate(ax):
        a.plot(x[i].numpy())
    plt.suptitle(f'{title}\n{num_examples} random examples')
    plt.show()


def single_freq_test():
    dataset = SineWaveformDataset(
        data_dir='data/basic_waveforms/sine_200hz.npy',
        sample_len=200,
        min_freq=200,
        max_freq=200,
        num_frequencies=1,
        num_random_amplitudes=10,
        noise_amplitude=0.1,
        overwrite_existing_file=True
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    plot_examples(dataloader, title='Single Freq')

    print(f'len(dataloader) = {len(dataloader)}, expected: 11')


def test_multiple_freq():
    dataset = SineWaveformDataset(
        data_dir='data/basic_waveforms/sine_multiple_freq.npy',
        sample_len=200,
        min_freq=80,
        max_freq=2_000,
        num_frequencies=500,
        num_random_amplitudes=0,
        noise_amplitude=0.1,
        overwrite_existing_file=True
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    plot_examples(dataloader, title='Multiple freq')

    print(f'len(dataloader) = {len(dataloader)}, expected: 500')


def sine_waveforms_lightning_test():
    data_config = {
        'data_dir': 'data/basic_waveforms/sine_lightning_test.npy',
        'sample_len': 200,
        'min_freq': 200,
        'max_freq': 200,
        'num_frequencies': 1,
        'num_random_amplitudes': 200,
        'noise_amplitude': 0.1,
        'quantize': False,
        'overwrite_existing_file': True,
        'val_split': 0.1
    }
    dataset = SineWaveLightningDataset(**data_config)
    dataloader = dataset.train_dataloader(**{'batch_size': 32, 'shuffle': False})
    plot_examples(dataloader, title='Lightning Dataloader')
    print(dataset)


if __name__ == '__main__':
    single_freq_test()
    test_multiple_freq()
    sine_waveforms_lightning_test()
