import os
import pickle
import bisect
from pathlib import Path
import obspy
from obspy import Stream
from obspy.core.inventory.inventory import Inventory
from obspy import UTCDateTime
from obspy import read
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from dataloaders.insight_utils.insight_plotting_utils import plot_example
from dataloaders.insight_utils.preprocessing import full_preprocessing

from dataloaders.base import SeisbenchDataLit

from dataloaders.insight_utils.insight_data_utils import load_raw_waveform_data
from dataloaders.insight_utils.insight_data_utils import phase_list_P, phase_list_S
import dataloaders.insight_utils.insight_data_utils as insight_data_utils


class InsightDataset2(Dataset):
    def __init__(self,
                 data_dir: str | Path,
                 data_file: str = 'dataloaders/data/insight/dataset_prep_faster_32768.pkl',
                 window_size_samples: int = 32768,
                 norm_type: str = 'std',
                 components: str = 'ZNE',
                 fs: int = 20,
                 split: str = 'train',
                 default_uncertainty: float = 20.0,
                 overwrite_uncertainties: bool = False,
                 noise_fraction: float = 0.5,
                 return_event_name: bool = False,
                 num_validation_repeats: int = 10,
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_file = Path(data_file)
        self.window_size_samples = window_size_samples
        self.window_size_seconds = float(window_size_samples) / fs
        self.norm_type = norm_type
        self.components = components
        self.fs = fs
        self.split = split
        self.default_uncertainty = default_uncertainty
        self.overwrite_uncertainties = overwrite_uncertainties
        self.noise_fraction = noise_fraction
        self.return_event_name = return_event_name
        self.num_validation_repeats = num_validation_repeats

        root_dir = Path(__file__).resolve().parent.parent
        self.catalogue_path = root_dir / 'dataloaders/data/insight/catalogs/events_InSIght_v14.pkl'


        if not self.data_file.exists():
            print(f"Data file {self.data_file} does not exist. Please run the preprocessing script first.")
        

        # load catalogue and preprocessed data information
        print(f'Loading catalogue from {self.catalogue_path}')
        self.catalogue = insight_data_utils.load_InSight_catalogue(self.catalogue_path)

        print(f'Loading preprocessed metadata data from {self.data_file}')
        with open(self.data_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
            self.events_with_picks = preprocessed_data['events_with_picks']
            self.events_with_noise_traces = preprocessed_data['events_with_noise_traces']
            self.picks_with_overlap = preprocessed_data['picks_with_overlap']
            self.noise_traces_with_waveform_data = preprocessed_data['noise_traces_with_waveform_data']

            assert self.window_size_samples == preprocessed_data['WINDOW_SIZE_SAMPLES'], \
                f"Window size samples mismatch: {self.window_size_samples} != {preprocessed_data['WINDOW_SIZE_SAMPLES']}"
            assert self.fs == preprocessed_data['FS'], \
                f"Sampling frequency mismatch: {self.fs} != {preprocessed_data['FS']}"

        # apply training/dev/test split with 80/10/10 ratio
        if self.split == 'train':
            self.events_with_picks = dict(list(self.events_with_picks.items())[:int(len(self.events_with_picks) * 0.8 + 1)])  # use only 80% of the events for training
        elif self.split == 'dev':
            self.events_with_picks = dict(list(self.events_with_picks.items())[int(len(self.events_with_picks) * 0.8 + 1):int(len(self.events_with_picks) * 0.9 + 1)])
        elif self.split == 'test':
            self.events_with_picks = dict(list(self.events_with_picks.items())[int(len(self.events_with_picks) * 0.9 + 1):])

        # only keep noise traces for the events that have picks
        self.events_with_noise_traces = [event for event in self.events_with_noise_traces if event in self.events_with_picks]
        # only keep noise traces with waveform data for the events that have picks
        self.noise_trace_dict = {i: {'start': s, 'end': e, 'event': event, 'location': l} for (i, (s, e, event, l)) in enumerate(self.noise_traces_with_waveform_data)
                                 if event in self.events_with_picks}

        # fill missing pick uncertainties with a default value
        # Optionally overwrite uncertainties with a specified value
        for event, picks in self.events_with_picks.items():
            for pick in picks:
                if self.overwrite_uncertainties or 'lower_uncertainty' not in pick or pick['lower_uncertainty'] is None:
                    pick['lower_uncertainty'] = self.default_uncertainty
                if self.overwrite_uncertainties or 'upper_uncertainty' not in pick or pick['upper_uncertainty'] is None:
                    pick['upper_uncertainty'] = self.default_uncertainty
        # only keep first P and S picks for each event
        self.events_with_picks = insight_data_utils.find_most_certain_picks(self.events_with_picks)
        
        print("Loading waveform data for events only, skipping full data loading.")
        self.event_waveform_data = insight_data_utils.load_waveforms_for_events(
            event_dict=self.events_with_picks,
            data_folder=self.data_dir,
            format=self.components,
            fs=self.fs,
            window_size_in_seconds=self.window_size_seconds,
            num_windows_around_event=5,
        )

        # compile list of all event names that have phase picks
        self.all_event_names = list(self.events_with_picks.keys())

        print(f'Dataset {self.split.upper()}')
        print(f'Number of events with picks: {len(self.events_with_picks)}')
        print(f'Number of noise traces: {len(self.noise_trace_dict.keys())}')
        print(f'Number of events with noise traces: {len(self.events_with_noise_traces)}')
        num_picks = sum(len(picks) for picks in self.events_with_picks.values())
        print(f'Number of picks: {num_picks}')


    def random_window_around_pick(
            self,
            pick: dict,
        ) -> Stream:
        """
        Crop a random window around a given pick in the stream.
        Uses .slice() and then .copy() to ensure the data is not modified in place.
        The cropped trace will contain the pick, but not necessarily the full Gaussian mask.

        :param pick: Dictionary containing the pick information, must contain 'time' key with UTCDateTime.
        :return: Stream object containing the cropped waveform data.
        """
        event_time = pick['time']
        fs = float(self.fs)

        start_offset_in_seconds = np.random.randint(self.window_size_samples) / fs
        #print(f"Start offset in seconds: {start_offset_in_seconds}")

        cropped = self.event_waveform_data[pick['event_name']].slice(
            starttime=event_time - start_offset_in_seconds,
            endtime=event_time + self.window_size_seconds - start_offset_in_seconds - 1. / fs,
        )

        cropped.copy()
        cropped.merge(method=1, fill_value=0)

        # pad start and end of traces if necessary
        cropped.trim(
            starttime=event_time - start_offset_in_seconds,
            endtime=event_time + self.window_size_seconds - start_offset_in_seconds - 1. / fs, # TODO fix this!
            pad=True,
            fill_value=0,
        )
        
        # the length should be +- 1 sample of the window size before ensure_crop_length
        cropped = insight_data_utils.ensure_crop_length(cropped, self.window_size_samples)

        assert len(cropped[0].data) == self.window_size_samples, f"Cropped data length {len(cropped[0].data)} does not match window size {self.window_size_samples}."
        assert np.sum(cropped[0].data) != 0, "Cropped data contains only zeros, check the pick time and window size."
        
        return cropped
    
    def get_label_sequences(self, cropped_st: Stream, picks_list: list,) -> torch.Tensor:
        """Generate label sequences for the cropped stream based on the picks.
        Each label sequence is a 3D array with shape (n_samples, 3), where:
        - n_samples is the window size and the number of samples in the cropped stream.
        - The first channel corresponds to P picks, the second to S picks, and the third to noise.
        Each pick is represented by a Gaussian mask centered at the pick time with a width based on the uncertainty.

        :param cropped_st: Stream object containing the cropped waveform data.
        :param picks_list: List of dictionaries containing pick information.
        :return: Numpy array of shape (n_samples, 3) containing the label sequences.
        """
        n = len(cropped_st[0].data)
        fs = cropped_st[0].stats.sampling_rate  # Sampling frequency
        x = torch.arange(n)

        p, s, noise = torch.zeros(n), torch.zeros(n), torch.ones(n)

        for pick in picks_list:
            mu = insight_data_utils.get_sample_offset(pick['time'], cropped_st[0])
            sigma = (pick['upper_uncertainty'] + pick['lower_uncertainty']) / 2.0 * fs

            gaussian_mask = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
            if pick['pick_type'] == 'P':
                p = torch.max(p, gaussian_mask)
            elif pick['pick_type'] == 'S':
                s = torch.max(s, gaussian_mask)
        
        noise = noise - (p + s)  # 1 - P - S
        
        noise = torch.clip(noise, 0, 1)  # Ensure noise is non-negative
        label_sequences = torch.stack((p, s, noise), axis=-1)
        return label_sequences.type(torch.float32)
    
    def get_noise_trace(self) -> Stream:

        cropped = Stream()
        while len(cropped) < 3 or any(np.sum(tr.data) == 0 for tr in cropped):
            rand_idx = np.random.randint(len(self.noise_trace_dict))
            event_name = self.noise_trace_dict[rand_idx]['event']
            start = self.noise_trace_dict[rand_idx]['start']
            end = self.noise_trace_dict[rand_idx]['end']

            random_start_time = UTCDateTime(start) + np.random.uniform(0, end - self.window_size_seconds - start)

            cropped = self.event_waveform_data[event_name].slice(
                starttime=random_start_time,
                endtime=random_start_time + self.window_size_seconds,
            )

            cropped.copy()
            cropped.merge(method=1, fill_value=0)

        # pad start and end of traces if necessary
        cropped.trim(
            starttime=random_start_time,
            endtime=random_start_time + self.window_size_seconds,
            pad=True,
            fill_value=0,
        )
        
        # the length should be +- 1 sample of the window size before ensure_crop_length
        cropped = insight_data_utils.ensure_crop_length(cropped, self.window_size_samples)
        
        return cropped, event_name


    def __len__(self):
        """
        Returns the number of events in the dataset.
        """
        if self.split == 'train':
            return int(len(self.events_with_picks) * (1 + self.noise_fraction))
        elif self.split == 'dev':
            return self.num_validation_repeats * len(self.events_with_picks)
        elif self.split == 'test':
            return len(self.events_with_picks)

    def __getitem__(self, idx):
        if self.split == 'train' and idx >= len(self.all_event_names):
            #
            # Randomly selected noise trace
            #
            p_and_s = torch.zeros((self.window_size_samples, 2), dtype=torch.float32)
            noise = torch.ones((self.window_size_samples, 1), dtype=torch.float32)
            label_sequences = torch.cat((p_and_s, noise), dim=1)  # [window_size_samples, 3]
            st, name = self.get_noise_trace()
            event_name = 'noise'
            
            picks = self.events_with_picks[name].copy()  # make a copy to avoid modifying the original data
            pick = np.random.choice(picks)
            if event_name in self.picks_with_overlap:
                overlapping_picks = self.picks_with_overlap[event_name]
                for op in overlapping_picks:
                    picks += self.events_with_picks[op].copy()
            label_sequences = self.get_label_sequences(st, picks) # [window_size, 3]

            assert torch.max(label_sequences[:, 0]) < 1e-4, f'Label sequences for P trace must be all zeros. {torch.max(label_sequences[:, 0])}'
            assert torch.max(label_sequences[:, 1]) < 1e-4, f'Label sequences for S trace must be all zeros. {torch.max(label_sequences[:, 1])}'
            assert torch.min(label_sequences[:, 2]) > 0.99, f'Label sequences for noise trace must be all ones. {torch.min(label_sequences[:, 2])}'

        else:
            if self.split == 'dev':
                idx = idx % len(self.events_with_picks)  # go through dev set multiple times to eliminate randomness
            # randomly select a pick
            event_name = self.all_event_names[idx]
            picks = self.events_with_picks[event_name].copy()  # make a copy to avoid modifying the original data
            pick = np.random.choice(picks)

            # append overlapping picks if they exist
            
            if event_name in self.picks_with_overlap:
                overlapping_picks = self.picks_with_overlap[event_name]
                for op in overlapping_picks:
                    picks += self.events_with_picks[op].copy()
            
            # get the waveform data for the event
            st = self.random_window_around_pick(pick=pick)
            
            # get the label sequences for the cropped stream
            label_sequences = self.get_label_sequences(st, picks) # [window_size, 3]
            #label_sequences = torch.tensor(label_sequences, dtype=torch.float32) # [window_size, 3]

        # ============================================================
        # Preprocessing
        # ============================================================
        assert all(len(tr.data) == self.window_size_samples for tr in st), \
            f"All traces must have {self.window_size_samples} samples, but got {[len(tr.data) for tr in st]}."
        
        st = insight_data_utils.normalize_trace(st, method=self.norm_type)
        st.taper(max_percentage=0.01)
        st.filter('highpass', freq=0.05)

        # convert to torch tensors
        # order traces by component, either ZNE or UVW.
        if self.components == 'ZNE':
            compoonents_list = ['Z', 'N', 'E']
        if self.components == 'UVW':
            compoonents_list = ['U', 'V', 'W']
        traces = [torch.tensor(st.select(component=comp)[0].data, dtype=torch.float32) for comp in compoonents_list]
        assert all(len(traces[i]) == self.window_size_samples for i in range(len(traces))), \
            f"All traces must have {self.window_size_samples} samples, but got {[len(traces[i]) for i in range(len(traces))]}."
        try:
            traces = torch.stack(traces, dim=1) # [window_size, 3]
        except:
            print(f"Error stacking traces:")

        if self.return_event_name:
            return traces, label_sequences, event_name
        else:
            return traces, label_sequences
        

class InsightDataLit2(SeisbenchDataLit):
    def __init__(self,
                 data_dir: str | Path,
                 data_file: str = 'dataloaders/data/insight/dataset_prep_faster_32768.pkl',
                 window_size_samples: int = 32768,
                 norm_type: str = 'std',
                 components: str = 'ZNE',
                 fs: int = 20,
                 default_uncertainty: float = 20.0,
                 overwrite_uncertainties: bool = False,
                 noise_fraction: float = 0.5,
                 num_validation_repeats: int = 10,
                 **kwargs):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.dataset_kwargs = {
            'data_file': data_file,
            'window_size_samples': window_size_samples,
            'norm_type': norm_type,
            'components': components,
            'fs': fs,
            'default_uncertainty': default_uncertainty,
            'overwrite_uncertainties': overwrite_uncertainties,
            'noise_fraction': noise_fraction,
            'num_validation_repeats': num_validation_repeats,
        }

        self.setup()

    def setup(self):
        """
        Setup the dataset for training, validation, and testing.
        """
        self.dataset_train = InsightDataset2(
            data_dir=self.data_dir,
            split='train',
            **self.dataset_kwargs
        )
        self.dataset_val = InsightDataset2(
            data_dir=self.data_dir,
            split='dev',
            **self.dataset_kwargs
        )
        self.dataset_test = InsightDataset2(
            data_dir=self.data_dir,
            split='test',
            **self.dataset_kwargs
        )
        self.d_data = 3


if __name__ == "__main__":
    # Example usage
    dataset = InsightDataLit2(data_dir='dataloaders/data/insight',
                            data_file='dataloaders/data/insight/dataset_prep_faster_8192.pkl',
                            window_size_samples=8192,
                            norm_type='std',
                            components='ZNE',
                            fs=20,
                            default_uncertainty=20.0,
                            noise_fraction=0.5,
                            return_event_name=True)
    loader_config = {
        'batch_size': 16,
        'num_workers': 0,
        'pin_memory': True,
    }

    loaders = {
        'train': DataLoader(dataset.dataset_train, **loader_config),
        'dev': DataLoader(dataset.dataset_val, **loader_config),
        'test': DataLoader(dataset.dataset_test, **loader_config),
    }
    # print loaders:
    for split, loader in loaders.items():
        print(f"Loader for {split}: {len(loader)} batches")

    for split in ['train', 'dev', 'test']:
        
        print(f'LOADING DATASET FOR SPLIT: {split.upper()}')

        loader = loaders[split]

        for ep in range(20):
            #print(f"\n\n\nEpoch {ep + 1} - Split: {split.upper()}")
            for i, (traces, labels) in enumerate(tqdm(loader, desc=f"Epoch {ep + 1} - Split: {split.upper()}")):
                #print(event_name)
                assert traces.shape[-1] == 3, f"Expected 3 components, got {traces.shape[1]}."
