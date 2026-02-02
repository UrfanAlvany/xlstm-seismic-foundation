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
from dataloaders.base import SeisbenchDataLit
from dataloaders.data_utils.seisbench_utils.augmentations import QuantizeAugmentation, FilterZChannel, \
    FillMissingComponents, AutoregressiveShift, TransposeLabels, TransposeSeqChannels, RandomMask, \
    SquashAugmentation, ChunkMask, BertStyleMask, BrainMask, RMSNormAugmentation, CopyXY, PretrainMask
from evaluation.eval_utils import moving_average

from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11
import seisbench
import seisbench.data as sbd
from seisbench.data import MultiWaveformDataset
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from dataloaders.data_utils.signal_encoding import quantize_encode

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


def apply_training_fraction(
        training_fraction,
        train_data,
) -> None:
    """
  Reduces the size of train_data to train_fraction by inplace filtering.
  Filter blockwise for efficient memory savings.

  Args:
    training_fraction: Training fraction between 0 and 1.
    train_data: Training dataset

  Returns:
    None
  """

    if not 0.0 < training_fraction <= 1.0:
        raise ValueError("Training fraction needs to be between 0 and 1.")

    if training_fraction < 1:
        blocks = train_data["trace_name"].apply(lambda x: x.split("$")[0])
        unique_blocks = blocks.unique()
        # Fixed seed for reproducibility (eliminates ±9.3% variance from random sampling)
        rng = np.random.RandomState(42)
        rng.shuffle(unique_blocks)
        target_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
        target_blocks = set(target_blocks)
        mask = blocks.isin(target_blocks)
        train_data.filter(mask, inplace=True)


def remove_unused_augmentations(augmentations):
    return [a for a in augmentations if a is not None]


def get_eval_augmentations(sample_len: int = 4096, d_data: int = 3, bits: int = 0, norm_type: str = 'peak'):
    augmentations = [
        sbg.SteeredWindow(windowlen=sample_len, strategy='pad'),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=norm_type),
        TransposeSeqChannels() if d_data == 3 else None,
    ]
    augmentations = remove_unused_augmentations(augmentations)
    '''
            sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=sample_len // 3, windowlen=sample_len,
                               selection="random",
                               strategy="variable"),
        # sbg.RandomWindow(windowlen=self.sample_len, strategy="pad"),
        FillMissingComponents(),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
        FilterZChannel() if d_data == 1 else None,
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        QuantizeAugmentation(bits=bits),
        TransposeSeqChannels() if d_data == 3 else None,
        TransposeLabels(),
    '''
    return augmentations


class SeisBenchAutoReg(SeisbenchDataLit):
    def __init__(self,
                 sample_len: int = 2048,
                 bits: int = 8,
                 d_data: int = 1,
                 preload: bool = False,
                 normalize_first: bool = False,
                 dataset_name: str = 'ETHZ',
                 norm_type: str = 'peak',
                 alpha: float = 1.0,
                 masking: float = 0.0,
                 bidir_autoreg: bool = False,
                 wav2vec: bool = False,
                 training_fraction: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_len = sample_len
        self.d_data = d_data
        self.bits = bits
        self.preload = preload
        self.normalize_first = normalize_first
        self.norm_type = norm_type
        self.alpha = alpha
        self.masking = masking
        self.bidir_autoreg = bidir_autoreg
        self.wav2vec = wav2vec

        print('dataset_name', dataset_name)

        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]

        multi_waveform_datasets = []
        cache = 'full' if preload else None
        for data_name in dataset_name:
            dataset = sbd.__getattribute__(data_name)(
                sampling_rate=100,
                component_order='ZNE',
                dimension_order='NCW',
                cache=cache
            )
            if "split" not in dataset.metadata.columns:
                print("No split defined, adding auxiliary split.")
                split = np.array(["train"] * len(dataset))
                split[int(0.6 * len(dataset)): int(0.7 * len(dataset))] = "dev"
                split[int(0.7 * len(dataset)):] = "test"

                dataset._metadata["split"] = split  # pylint: disable=protected-access

            multi_waveform_datasets.append(dataset)

        if len(multi_waveform_datasets) == 1:
            data = multi_waveform_datasets[0]
        else:
            # Concatenate multiple datasets
            print(multi_waveform_datasets)
            data = MultiWaveformDataset(multi_waveform_datasets)

        self.train, self.dev, self.test = data.train_dev_test()

        if training_fraction < 1.0:
            apply_training_fraction(training_fraction=training_fraction, train_data=self.train)

        if self.preload:
            self.train.preload_waveforms(pbar=True)
            self.dev.preload_waveforms(pbar=True)

        self.setup()

    def setup(self):
        if not self.wav2vec:
            window_len = self.sample_len + 1 if self.masking == 0 and not self.bidir_autoreg else self.sample_len
        else:
            window_len = self.sample_len

        '''
        sbg.Normalize(
            demean_axis=-1,
            amp_norm_axis=-1,
            amp_norm_type=self.norm_type,
        ) if self.normalize_first else None,
        '''
        '''
                    sbg.WindowAroundSample(list(phase_dict.keys()),
                                           samples_before=self.sample_len,
                                           windowlen=2 * self.sample_len,
                                           selection="random",
                                           strategy="variable"),
                                        '''
        augmentations = [
            sbg.RandomWindow(windowlen=window_len, strategy="pad"),
            FillMissingComponents(),
            # sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
            FilterZChannel() if self.d_data == 1 else None,
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(
                demean_axis=-1,
                amp_norm_axis=-1,
                amp_norm_type=self.norm_type,
            ),
            QuantizeAugmentation(bits=self.bits) if self.bits > 0 else None,
            TransposeSeqChannels() if self.d_data == 3 else None,
            #CopyXY(),
            #AutoregressiveShift() if self.masking == 0 else BrainMask(p1=self.masking, p2=0.5),
        ]
        if self.bidir_autoreg:
            augmentations.append(CopyXY())
        elif self.wav2vec:
            # we dont use y for this pretraining task
            pass
        else:
            augmentations.append(AutoregressiveShift() if self.masking == 0 else PretrainMask(p=self.masking))

        augmentations = remove_unused_augmentations(augmentations)

        self.dataset_train = sbg.GenericGenerator(self.train)
        self.dataset_val = sbg.GenericGenerator(self.dev)
        self.dataset_test = sbg.GenericGenerator(self.test)

        self.dataset_train.add_augmentations(augmentations)
        self.dataset_val.add_augmentations(augmentations)
        self.dataset_test.add_augmentations(augmentations)

        if self.bits > 0:
            self.num_classes = 2 ** self.bits


class SeisBenchPhasePick(SeisbenchDataLit):
    def __init__(
            self,
            sample_len: int = 2048,
            bits: int = 8,
            d_data: int = 1,
            preload: bool = False,
            sample_boundaries=(None, None),
            sigma=20,
            dataset_name: str = 'ETHZ',
            norm_type: str = 'sqrt',
            training_fraction: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_len = sample_len
        self.bits = bits
        self.d_data = d_data
        self.preload = preload
        self.sample_boundaries = sample_boundaries
        self.sigma = sigma
        self.norm_type = norm_type
        self.training_fraction = training_fraction

        dataset_kwargs = {
            'sampling_rate': 100,
            'component_order': 'ZNE',
            'dimension_order': 'NCW'
        }
        if self.preload:
            dataset_kwargs['cache'] = 'full'

        if dataset_name == 'ETHZ':
            data = sbd.ETHZ(**dataset_kwargs)
        elif dataset_name == 'GEOFON':
            data = sbd.GEOFON(**dataset_kwargs)
        elif dataset_name == 'STEAD':
            data = sbd.STEAD(**dataset_kwargs)
        elif dataset_name == 'INSTANCE':
            data = sbd.InstanceCountsCombined(**dataset_kwargs)
        elif dataset_name == 'VCSEIS':
            data = sbd.VCSEIS(**dataset_kwargs)
        elif dataset_name == 'CEED':
            data = sbd.CEED(**dataset_kwargs)
        else:
            print(f'Unknown dataset: {dataset_name}')
        self.train, self.dev, self.test = data.train_dev_test()

        if self.training_fraction < 1.0:
            apply_training_fraction(training_fraction=training_fraction, train_data=self.train)

        if self.preload:
            self.train.preload_waveforms(pbar=True)
            self.dev.preload_waveforms(pbar=True)

        self.setup()

    def setup(self):

        augmentations = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=self.sample_len,
                        windowlen=2 * self.sample_len,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=self.sample_len,
                strategy="pad",
            ),
            FillMissingComponents(),
            FilterZChannel() if self.d_data == 1 else None,
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(
                demean_axis=-1,
                amp_norm_axis=-1,
                amp_norm_type=self.norm_type,
            ),
            QuantizeAugmentation(bits=self.bits) if self.bits > 0 else None,
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
            TransposeSeqChannels() if self.d_data == 3 else None,
            TransposeLabels(),
        ]
        augmentations = remove_unused_augmentations(augmentations)

        self.dataset_train = sbg.GenericGenerator(self.train)
        self.dataset_val = sbg.GenericGenerator(self.dev)
        self.dataset_test = sbg.GenericGenerator(self.test)

        self.dataset_train.add_augmentations(augmentations)
        self.dataset_val.add_augmentations(augmentations)
        self.dataset_test.add_augmentations(augmentations)

        if self.bits > 0:
            self.num_classes = 2 ** self.bits


def phase_pick_test():
    data_config = {
        'sample_len': 4096,
        'bits': 0,
        'd_data': 3,
        'dataset_name': ['ETHZ', 'GEOFON'],
        'norm_type': 'std',
    }
    loader_config = {
        'batch_size': 64,
        'num_workers': 0,
        'shuffle': True,
    }
    dataset = SeisBenchAutoReg(**data_config)
    loader = DataLoader(dataset.dataset_val, **loader_config)

    batch = next(iter(loader))
    print(len(loader.dataset))

    x, mask = batch['X']
    #mask = batch['mask']
    s = 0
    l = -1
    for i in range(16):
        plt.plot(x[i, s:l])
        plt.show()
        #plt.plot(mask[i, s:l])
        #plt.show()

    total_avg = 0
    for i, batch in enumerate(loader):
        x, mask = batch['X']
        y = batch['y']
        autoreg_mse = torch.nn.functional.mse_loss(x[mask], y[mask])
        total_avg += autoreg_mse
        print(f'autoreg_mse: {autoreg_mse :.4f}, log_mse: {torch.log(autoreg_mse):.4f}')
    print('total_avg: ', total_avg / len(loader))

def bidir_autoreg_test():
    data_config = {
        'sample_len': 4096,
        'bits': 0,
        'd_data': 3,
        'normalize_first': True,
        'dataset_name': ['ETHZ'],
        'norm_type': 'std',
        'bidir_autoreg': True,
    }
    loader_config = {
        'batch_size': 64,
        'num_workers': 0,
        'shuffle': True,
    }
    dataset = SeisBenchAutoReg(**data_config)
    loader = DataLoader(dataset.dataset_val, **loader_config)

    batch = next(iter(loader))
    print(len(loader.dataset))
    total_avg = 0
    for i, batch in enumerate(loader):
        x, y = batch['X'], batch['y']
        l1 = torch.nn.functional.mse_loss(x[:, 1:, :], y[:, :-1, :])
        l2 = torch.nn.functional.mse_loss(x[:, :-1, :], y[:, 1:, :])
        print(f'bidir loss: {l1 + l2 :.4f}')
        total_avg += l1 + l2
    print('total_avg: ', total_avg / len(loader))


def fraction_test():
    training_fractions = [0.05, 0.1, 0.2, 0.5, 1]
    dataset_name = 'ETHZ'
    batch_sizes = [32, 64, 64, 128, 256]

    for fraction, batch_size in zip(training_fractions, batch_sizes):
        data_config = {
            'sample_len': 4096,
            'bits': 0,
            'd_data': 3,
            'dataset_name': dataset_name,
            'norm_type': 'std',
            'training_fraction': fraction
        }
        loader_config = {
            'batch_size': batch_size,
            'num_workers': 0,
            'shuffle': True,
            'drop_last': True,
        }
        dataset = SeisBenchAutoReg(**data_config)
        loader = DataLoader(dataset.dataset_train, **loader_config)

        batch = next(iter(loader))
        print(f'fraction: {fraction}, len(dataset): {len(loader.dataset)}')
        print(f'batch_size: {batch_size}, num_batches: {len(loader)}')


def seed_test():
    torch.manual_seed(0)
    np.random.seed(0)
    batch_size = 16

    data_config = {
        'sample_len': 4096,
        'bits': 0,
        'd_data': 3,
        'dataset_name': 'ETHZ',
        'norm_type': 'std',
        'training_fraction': 0.05
    }
    loader_config = {
        'batch_size': batch_size,
        'num_workers': 0,
        'shuffle': True,
        'drop_last': True,
    }
    dataset = SeisBenchPhasePick(**data_config)
    loader = DataLoader(dataset.dataset_train, **loader_config)

    batch = next(iter(loader))
    print(f'fraction: {0.05}, len(dataset): {len(loader.dataset)}')
    print(f'batch_size: {batch_size}, num_batches: {len(loader)}')

    plt.plot(batch['X'][0])
    plt.show()


if __name__ == "__main__":
    #phase_pick_test()
    #bidir_autoreg_test()
    fraction_test()
    #seed_test()
