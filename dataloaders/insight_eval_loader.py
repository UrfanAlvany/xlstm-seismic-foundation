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
from pandas import DataFrame
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd

from dataloaders.insight_utils.insight_plotting_utils import plot_example
from dataloaders.insight_utils.preprocessing import full_preprocessing

from dataloaders.base import SeisbenchDataLit

from dataloaders.insight_utils.insight_data_utils import load_raw_waveform_data
from dataloaders.insight_utils.insight_data_utils import phase_list_P, phase_list_S
import dataloaders.insight_utils.insight_data_utils as insight_data_utils
from dataloaders.insight_utils.insight_plotting_utils import plot_example


class InsightEvalDataset(Dataset):
    def __init__(self,
                 data_dir: str | Path,
                 targets: DataFrame,
                 task: str,
                 fs: int = 20,
                 window_size_in_samples: int = 32768,
                 components: str = 'ZNE',
                 norm_type: str = 'std',
                 ):
        """
        Initialize the InsightEvalDataset. 

        @param data_dir: Path to the directory containing the waveform data.
        @param targets: DataFrame containing the targets for the dataset. Needs to be a Dataframe with columns 'start_time' and 'end_time'.
        The 'start_time' and 'end_time' columns should be in UTCDateTime format or convertible to it.
        @param task: The task to perform, either 'task1', 'task23' or 'task4'.
        task1: event detection. task23: phase discrimination and onset determination. task4: MQNet deep catalogue.
        @param fs: Sampling frequency of the waveform data in Hz. Default is 20 Hz.
        @param window_size_in_samples: Size of the sliding window in samples. Default is 32768 samples (1638.4 seconds at 20 Hz).
        @param components: The components to use, either 'ZNE' or 'UVW'. Default is 'ZNE'.
        @param norm_type: Normalization type, either 'std' for standard deviation or 'peak' for maximum value. Default is 'std'.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.targets = targets
        self.task = task
        self.fs = float(fs)
        self.window_size_in_samples = window_size_in_samples
        self.window_size_in_seconds = self.window_size_in_samples / self.fs
        self.components = components
        self.norm_type = norm_type

        assert task in ['task1', 'task23', 'task4'], f"Task {task} not supported. Choose 'task1', 'task23' or 'task4'."
        assert components in ['ZNE', 'UVW'], f"Components {components} not supported. Choose 'ZNE' or 'UVW'."
        assert norm_type in ['std', 'peak'], f"Normalization type {norm_type} not supported. Choose 'std' or 'peak'."

        self.daily_data = insight_data_utils.load_waveform_data_by_day(
            data_folder=self.data_dir,
            format='ZNE',
            fs=self.fs,
        )

        try:
            self.deep_catalogue = pd.read_csv(self.data_dir / 'MQNet_DeepCatalogue.csv')
            self.deep_catalogue.rename(columns={'utc_start': 'start_time'}, inplace=True)
            self.deep_catalogue.rename(columns={'utc_end': 'end_time'}, inplace=True)
            print(f"Deep catalogue loaded with {len(self.deep_catalogue)} entries.")
        except:
            print("Deep catalogue not found.")
            self.deep_catalogue = None


    def set_new_task(self, task: str, targets: DataFrame):
        """
        Set a new task and update the targets DataFrame.

        @param task: The new task to set, either 'task1' or 'task23'.
        @param targets: The DataFrame containing the targets for the new task.
        """
        assert task in ['task1', 'task23', 'task4'], f"Task {task} not supported. Choose 'task1', 'task23' or 'task4'."
        assert isinstance(targets, DataFrame), "Targets must be a pandas DataFrame."
        assert 'start_time' in targets.columns and 'end_time' in targets.columns, \
            "Targets DataFrame must contain 'start_time' and 'end_time' columns."
        
        # setup the new task and targets
        self.task = task
        self.targets = targets
        print(f"Task set to {self.task} with {len(self.targets)} targets.")



    def get_traces_containing(self, start_time: UTCDateTime, end_time: UTCDateTime) -> tuple[torch.Tensor, UTCDateTime, torch.Tensor]:
        """
        Get three-component traces of length self.window_size_in_samples that contain the given start and end times.
        The position of the window defined by start_time and end_time is random uniform within the trace.

        @param start_time: UTCDateTime representing the start time of the window.
        @param end_time: UTCDateTime representing the end time of the window.
        @return: A tuple containing:
            - traces: A tensor of shape [window_size, 3] containing the three components (Z, N, E or U, V, W).
            - trace_start_time: The start time of the trace.
            - window_border_indices: A tensor of shape [2] containing the start and end indices of the window within the trace.
        """
        assert end_time > start_time, "End time must be after start time."


        start_day = (start_time - self.window_size_in_seconds).strftime('%Y-%m-%d')
        end_day = (end_time + self.window_size_in_seconds).strftime('%Y-%m-%d')
        st = Stream()
        if start_day == end_day:
            # If the event is on the same day, load that day's data
            st = self.daily_data[start_day]
        else:
            # If the event spans multiple days, load both days' data
            st += self.daily_data[start_day]
            st += self.daily_data[end_day]
        
        # select a random start time such that random_start + self.window_size_in_seconds <= end_time
        random_start = start_time + np.random.uniform(0, (end_time - start_time - self.window_size_in_seconds))

        st = st.slice(
            starttime=random_start,
            endtime=random_start + self.window_size_in_seconds - 1.0 / self.fs,
        ).copy().merge(method=1, fill_value=0)

        st.trim(
            starttime=random_start,
            endtime=random_start + self.window_size_in_seconds - 1.0 / self.fs,
            pad=True,
            fill_value=0,
        )
        st = insight_data_utils.ensure_crop_length(st, self.window_size_in_samples)

        st = insight_data_utils.normalize_trace(st, method=self.norm_type)
        st.taper(max_percentage=0.01)
        st.filter('highpass', freq=0.05)

        if self.components == 'ZNE':
            compoonents_list = ['Z', 'N', 'E']
        if self.components == 'UVW':
            compoonents_list = ['U', 'V', 'W']
        traces = [torch.tensor(st.select(component=comp)[0].data, dtype=torch.float32) for comp in compoonents_list]
        traces = torch.stack(traces, dim=1)  # [window_size, 3]
        

        trace_start_time = st[0].stats.starttime
        window_start_index = int((start_time - trace_start_time) * self.fs)
        window_end_index = int((end_time - trace_start_time) * self.fs)

        assert window_start_index >= 0, f"Window start index {window_start_index} is negative."
        assert window_end_index <= len(st[0].data), f"Window end index {window_end_index} exceeds trace length {len(st[0].data)}."

        return traces, trace_start_time, torch.tensor([window_start_index, window_end_index], dtype=torch.int32)


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        @param idx: Index of the item to retrieve.
        @return: A tuple containing:
            - traces: A tensor of shape [window_size, 3] containing the three components (Z, N, E or U, V, W).
            - trace_start_time: The start time of the trace as a string (UTCDateTime format). Start time of the trace, not the start of the window.
            - window_border_indices: A tensor of shape [2] containing the start and end indices of the window within the trace.
        """
        entry = self.targets.iloc[idx]
        window_start_time = UTCDateTime(entry['start_time'])
        window_end_time = UTCDateTime(entry['end_time'])
        traces, trace_start_time, window_border_indices = self.get_traces_containing(window_start_time, window_end_time)
        return traces, str(trace_start_time), window_border_indices


def deep_catalogue_test(show_plots: bool = False):
    # initialize the dataset with the path to the data directory and the task
    dataset = InsightEvalDataset(
        data_dir='dataloaders/data/insight',
        targets=pd.DataFrame(),  # empty DataFrame, we will set the task later
        task='task1',   
    )
    # load the task4 target csv
    task4_targets = pd.read_csv('dataloaders/data/insight/task4.csv')

    # set the task to 'task4' to use the deep catalogue
    dataset.set_new_task(task='task4', targets=task4_targets)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # iterate over the loader with tqdm and check that all sample lengths are equal to dataset.window_size_in_samples
    for i, batch in enumerate(tqdm(loader, desc="Loading task4 samples")):
        traces, trace_start_times, window_border_indices = batch
        if show_plots:
            plot_example(
                cropped_st=traces[0],
                label_sequences=torch.zeros_like(traces[0]),
                title=f'First example from deep catalogue (batch {i})',
            )
        assert traces.shape[-1] == 3, f"Expected 3 components, got {traces.shape[1]}"
        assert traces.shape[1] == dataset.window_size_in_samples, f"Expected {dataset.window_size_in_samples} samples, got {traces.shape[0]}"
        
if __name__ == '__main__':
    deep_catalogue_test(show_plots=False)
    print("Deep catalogue test passed.")
