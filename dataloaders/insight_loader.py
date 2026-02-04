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


class InsightPretrainingDataset(Dataset):
    def __init__(self,
                 data_dir: str | Path,
                 window_size_samples: int = 32768, # 27 minutes at 20Hz
                 norm_type: str = 'std',
                 components: str = 'ZNE',
                 fs: int = 20,
                 train: str = 'train',
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.window_size_samples = window_size_samples
        self.window_size_in_seconds = window_size_samples / float(fs)  # in seconds
        self.norm_type = norm_type
        self.components = components
        self.fs = float(fs)
        self.train = train

        root_dir = Path(__file__).resolve().parent.parent

        self.dev_days = [
            UTCDateTime(2020, 3, 3),
            UTCDateTime(2021, 11, 21),
        ]

        self.test_days = [
            UTCDateTime(2020, 1, 20),
            UTCDateTime(2021, 12, 5),
        ]

        def load_limited_daily_data(days: list[UTCDateTime]) -> dict:
            """
            Load only the daily data for the specified days.
            """
            all_files = insight_data_utils.get_all_mseed_files(self.data_dir, self.components)
            daily_data = {}
            # add consecutive day to the list of days to load
            next_days = [day + 86500 for day in days]  # add one day to each day
            days += next_days

            for day in days:
                file_name = insight_data_utils.utcdatetime_to_filename(day, self.components)
                file_name = Path(self.data_dir) / file_name
                if str(file_name) in all_files:
                    st = read(file_name)

                    for trace in st:
                        if trace.stats.sampling_rate != fs:
                            print('Resampling!')
                            trace.resample(fs)
                    st.merge(method=1, fill_value=0)
                    daily_data[day.strftime('%Y-%m-%d')] = st
                else:
                    print(f"Warning: File {file_name} not found in {self.data_dir}. Skipping this day.")

            print(f"Loaded {len(daily_data)} days of data from {self.data_dir}.")
            return daily_data

        # load task4 file. This file contains overlapping windows of size 27min for the full InSight mission.
        # should only contain windows that have waveform data.
        print(f"Loading task4 file from {root_dir / 'dataloaders/data/insight/task4.csv'}.")
        task4_file = pd.read_csv(root_dir / 'dataloaders/data/insight/task4.csv')
        self.windows = task4_file[['start_time', 'end_time']].copy()
        
        # convert start_time and end_time to UTCDateTime objects
        self.windows['start_time'].apply(UTCDateTime)
        self.windows['end_time'].apply(UTCDateTime)

        # select two days of data for the dev and test sets
        windows_dev = pd.DataFrame()
        for day in self.dev_days:
            windows_dev = pd.concat([
                windows_dev,
                self.windows[
                    (self.windows['start_time'] >= day) & (self.windows['end_time'] <= day + 86400)
                ]
            ], ignore_index=False) # keep indeces to drop them from training data

        windows_test = pd.DataFrame()
        for day in self.test_days:
            windows_test = pd.concat([
                windows_test,
                self.windows[
                    (self.windows['start_time'] >= day) & (self.windows['end_time'] <= day + 86400)
                ]
            ], ignore_index=False)
        

        # load waveform data and drop the windows that are not in the split
        if self.train == 'train':
            # drop the windows that are in the dev and test sets
            train_windows = self.windows.drop(windows_dev.index).drop(windows_test.index)
            self.windows = train_windows.reset_index(drop=True)
            print(f"Using {len(self.windows)} windows for the training set.")
            
            # load the full waveform data
            print(f"Loading raw waveform data from {self.data_dir} with components {self.components} at {self.fs}Hz.")
            self.daily_data = insight_data_utils.load_waveform_data_by_day(
                data_folder=self.data_dir,
                format=self.components,
                fs=self.fs,
            )
        elif self.train == 'dev':
            # load only the waveform data for the dev set
            self.daily_data = load_limited_daily_data(self.dev_days)
            self.windows = windows_dev.reset_index(drop=True)
            print(f"Using {len(self.windows)} windows for the dev set.")
        elif self.train == 'test':
            # load only the waveform data for the test set
            self.daily_data = load_limited_daily_data(self.test_days)
            self.windows = windows_test.reset_index(drop=True)
            print(f"Using {len(self.windows)} windows for the test set.")


    def __len__(self):
        """
        Return the number of windows in the dataset.
        """
        return len(self.windows)
    

    def __getitem__(self, idx):
        start_time = UTCDateTime(self.windows.iloc[idx]['start_time'])
        end_time = UTCDateTime(self.windows.iloc[idx]['end_time'])

        start_day = start_time.strftime('%Y-%m-%d')
        end_day = end_time.strftime('%Y-%m-%d')

        st = Stream()
        if start_day == end_day:
            st += self.daily_data[start_day]
        else:
            st += self.daily_data[start_day]
            st += self.daily_data[end_day]
        
        
        random_shift = 0# np.random.uniform(-self.window_size_in_seconds / 4.0, self.window_size_in_seconds / 4.0)
        st = st.slice(
            starttime=start_time + random_shift,
            endtime=end_time + random_shift + 1/self.fs,
        ).copy().merge(method=1, fill_value=0)

        st.trim(
            starttime=start_time + random_shift,
            endtime=end_time + random_shift + 1/self.fs,
            pad=True,
            fill_value=0,
        )
        

        # ensure the lenght of the stream and fill missing channels
        st = insight_data_utils.fill_missing_channels(st=st, 
                                                      components=self.components,
                                                      strategy='copy',
                                                      window_size=self.window_size_samples,
                                                      )
        st = insight_data_utils.ensure_crop_length(st, self.window_size_samples)
        
        # preprocessing
        st = insight_data_utils.normalize_trace(st, method=self.norm_type)
        st.taper(max_percentage=0.01)
        st.filter('highpass', freq=0.05)

        components_list = [c for c in self.components] # e.g. ['Z', 'N', 'E'] or ['U', 'V', 'W']
        traces = [torch.tensor(st.select(component=comp)[0].data, dtype=torch.float32) for comp in components_list]
        traces = torch.stack(traces, dim=1)  # [window_size, 3]
        
        # return traces and placeholder for labels
        # labels are not used in pretraining, but the dataloader expects them
        return traces, 0

        

class InsightDataset(Dataset):
    def __init__(self, 
                 data_dir: str | Path,
                 window_size_samples: int = 32768, # 27 minutes at 20Hz -> see MarsQuakeNet
                 norm_type: str = 'std',
                 components: str = 'ZNE',
                 fs: int = 20,
                 train: str = 'train',
                 default_uncertainty: float = 20.0,
                 overwrite_uncertainties: bool = False,
                 load_inventory: bool = True,
                 noise_fraction: float = 0.5,
                 pre_slice: bool = True,
                 merge_when_loading: bool = False,
                 preload_waveform_data: bool = True,
                 load_only_event_data: bool = False,
                 return_event_name: bool = False,
                 use_daily_datastructure: bool = False,
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.window_size_samples = window_size_samples
        self.window_size_in_seconds = window_size_samples / float(fs)  # in seconds
        self.norm_type = norm_type
        self.components = components
        self.fs = fs
        self.train = train
        self.default_uncertainty = default_uncertainty
        self.overwrite_uncertainties = overwrite_uncertainties
        self.noise_fraction = noise_fraction
        self.pre_slice = pre_slice
        self.return_event_name = return_event_name
        self.load_only_event_data = load_only_event_data
        self.use_daily_datastructure = use_daily_datastructure

        root_dir = Path(__file__).resolve().parent.parent

        self.metadata_path = root_dir / 'dataloaders/data/insight/dataless.XB.ELYSE.seed'
        self.catalogue_path = root_dir / 'dataloaders/data/insight/catalogs/events_InSIght_v14.pkl'


        # Load the raw waveform data
        if preload_waveform_data and not load_only_event_data and not use_daily_datastructure:
            print(f"Loading raw waveform data from {self.data_dir} with components {self.components} at {self.fs}Hz.")
            self.full_data = load_raw_waveform_data(self.data_dir, format=self.components, fs=self.fs, merge=merge_when_loading)
        elif use_daily_datastructure:
            print(f"Loading daily waveform data from {self.data_dir} with components {self.components} at {self.fs}Hz.")
            self.daily_data = insight_data_utils.load_waveform_data_by_day(
                data_folder=self.data_dir,
                format=self.components,
                fs=self.fs,
            )
        elif not preload_waveform_data and not load_only_event_data and not use_daily_datastructure:
            print('WARNING: not loading raw waveform data')
        
        #self.full_data.detrend('constant')  # Detrend the data to remove any constant offsets

        if load_inventory:
            # Load the metadata inventory
            print(f"Loading metadata inventory from {self.metadata_path}.")
            self.inventory = insight_data_utils.get_metadata_inventory(self.metadata_path)
        else:
            print("Skipping loading of inventory.")
            self.inventory = None

        # Load the InSight catalogue
        print(f"Loading InSight catalogue from {self.catalogue_path}.")
        self.catalogue = insight_data_utils.load_InSight_catalogue(self.catalogue_path)

        # get all phase picks from the catalogue
        self.all_phase_picks = insight_data_utils.get_event_picks(self.catalogue)

        # make sure all phase picks have corresponding waveform data
        if preload_waveform_data:
            self.start_time = min(tr.stats.starttime for tr in self.full_data)
            self.end_time = max(tr.stats.endtime for tr in self.full_data)
        else:
            self.start_time = UTCDateTime(2019, 6, 1)
            self.end_time = UTCDateTime(2022, 5, 26)
            print(f"Using default start time {self.start_time} and end time {self.end_time} for filtering picks.")
        
        self._filter_picks_by_date_range(
            start_date=self.start_time,
            end_date=self.end_time,
        )

        # fill missing pick uncertainties with a default value
        # Optionally overwrite uncertainties with a specified value
        insight_data_utils.fill_missing_uncertainities(
            self.all_phase_picks, 
            default_uncertainty=self.default_uncertainty,
            overwrite=self.overwrite_uncertainties,
        )

        # compile list of all picks for each event
        self.events_with_picks = insight_data_utils.get_dict_by_event_name(self.all_phase_picks)

        # TODO: remove this hack
        # event S0672a seems to be missing the raw waveform data.
        del self.events_with_picks['S0672a']
        #print(f"Number of events with picks: {len(self.events_with_picks)}")


        # TODO: double check this split. For now this is a placeholder.
        if self.train == 'train':
            self.events_with_picks = dict(list(self.events_with_picks.items())[:int(len(self.events_with_picks) * 0.8)])  # use only 80% of the events for training
        elif self.train == 'dev':
            self.events_with_picks = dict(list(self.events_with_picks.items())[int(len(self.events_with_picks) * 0.8):int(len(self.events_with_picks) * 0.9)])
        elif self.train == 'test':
            self.events_with_picks = dict(list(self.events_with_picks.items())[int(len(self.events_with_picks) * 0.9):])


        # only keep first P and S picks for each event
        #self.events_with_picks = insight_data_utils.find_first_picks(self.events_with_picks)
        self.events_with_picks = insight_data_utils.find_most_certain_picks(self.events_with_picks)

        if load_only_event_data:
            print("Loading waveform data for events only, skipping full data loading.")
            self.event_waveform_data = insight_data_utils.load_waveforms_for_events(
                event_dict=self.events_with_picks,
                data_folder=self.data_dir,
                format=self.components,
                fs=self.fs,
                window_size_in_seconds= self.window_size_in_seconds,
            )
        
        # compile list of all event names that have phase picks
        self.all_event_names = list(self.events_with_picks.keys())

        if self.pre_slice:
            self.slices = self.prepare_slices()

        assert all(len(picks) > 0 and len(picks) <= 2 for picks in self.events_with_picks.values()), \
            "Each event must have at least one pick and at most two picks (P and S)."


    def prepare_slices(self):
        """
        Prepare slices (references to the original data) for each event.
        """
        print("\nPreparing slices for each event...")
        slices = {}
        
        for event_name, picks in tqdm(self.events_with_picks.items(), desc="Slicing events"):
            slice = self.full_data.slice(
                starttime=min(pick['time'] for pick in picks) - 1.1 * self.window_size_in_seconds,
                endtime=max(pick['time'] for pick in picks) + 1.1 * self.window_size_in_seconds,
            )
            slices[event_name] = slice
            if len(slice) == 0:
                print(f"Warning: No raw data found for event {event_name}.")
    
        return slices
    

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
        window_size_in_seconds = self.window_size_samples / fs

        if self.pre_slice:
            cropped = self.slices[pick['event_name']].slice(
                starttime=event_time - start_offset_in_seconds,
                endtime=event_time + window_size_in_seconds - start_offset_in_seconds - 1. / fs, # TODO fix this!
            )
        elif self.load_only_event_data:
            cropped = self.event_waveform_data[pick['event_name']].slice(
                starttime=event_time - start_offset_in_seconds,
                endtime=event_time + window_size_in_seconds - start_offset_in_seconds - 1. / fs,
            )
        elif self.use_daily_datastructure:
            start_day = (event_time - start_offset_in_seconds).strftime('%Y-%m-%d')
            end_day = (event_time + window_size_in_seconds - start_offset_in_seconds).strftime('%Y-%m-%d')
            st = self.daily_data[start_day]
            if end_day != start_day:
                st += self.daily_data[end_day]
            cropped = st.slice(
                starttime=event_time - start_offset_in_seconds,
                endtime=event_time + window_size_in_seconds - start_offset_in_seconds - 1. / fs,
            )
        else:
            cropped = self.full_data.slice(
                starttime=event_time - start_offset_in_seconds,
                endtime=event_time + window_size_in_seconds - start_offset_in_seconds - 1. / fs,
            )

        cropped.copy()
        cropped.merge(method=1, fill_value=0)
        '''
        if not all(len(tr.data) == self.window_size_samples for tr in cropped):
            print(f"Warning: Cropped data length does not match window size {self.window_size_samples}.")
            print(f"Cropped data lengths: {[len(tr.data) for tr in cropped]}")
            print(f"Event time: {event_time}, Start offset: {start_offset_in_seconds}, Window size: {window_size_in_seconds}")
        '''
        # pad start and end of traces if necessary
        cropped.trim(
            starttime=event_time - start_offset_in_seconds,
            endtime=event_time + window_size_in_seconds - start_offset_in_seconds - 1. / fs, # TODO fix this!
            pad=True,
            fill_value=0,
        )
        
        # the length should be +- 1 sample of the window size before ensure_crop_length
        cropped = insight_data_utils.ensure_crop_length(cropped, self.window_size_samples)

        assert len(cropped[0].data) == self.window_size_samples, f"Cropped data length {len(cropped[0].data)} does not match window size {self.window_size_samples}."
        assert np.sum(cropped[0].data) != 0, "Cropped data contains only zeros, check the pick time and window size."
        
        return cropped
    

    def normalize_trace(self, st: Stream, method: str ='std') -> Stream:
        """
        Remove the mean and normalize ('peak') or standardize ('std') the traces in a Stream object.
        Normalization is done per trace resp. per component.
        The data is modified in place.
        
        :param st: Stream object containing the traces to normalize.
        :param method: Normalization method, either 'std' for standard deviation or 'peak' for maximum value.
        :return: Normalized Stream object.
        """
        if method not in ['std', 'peak']:
            raise ValueError("Normalization method must be either 'std' or 'peak'")
        
        # Create a copy to avoid modifying the original Stream
        # Necessary because .slice() produces references to the original data
        #st = st.copy()  

        for trace in st:
            if method == 'std':
                #print(trace.stats)
                #print(f'Std: {np.std(trace.data)}, Mean: {np.mean(trace.data)}')
                trace.data = (trace.data - np.mean(trace.data)) / np.std(trace.data)
            elif method == 'max':
                trace.data = (trace.data - np.mean(trace.data)) / np.max(np.abs(trace.data))
        
        return st
    

    def _filter_picks_by_date_range(self, start_date: UTCDateTime, end_date: UTCDateTime) -> None:
        """
        Filter the phase picks by a specified date range.
        Modifies self.all_phase_picks
        :param start_date: UTCDateTime object representing the start of the date range.
        :param end_date: UTCDateTime object representing the end of the date range.
        """
        num_picks_before = len(self.all_phase_picks)
        filtered_picks = {}
        
        for pick_id, pick in self.all_phase_picks.items():
            if start_date <= pick['time'] <= end_date:
                filtered_picks[pick_id] = pick
        self.all_phase_picks = filtered_picks

        num_picks_after = len(self.all_phase_picks)
        print(f'Removed {num_picks_before - num_picks_after} picks outside the date range {start_date} to {end_date}.')


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
                p += gaussian_mask    
            elif pick['pick_type'] == 'S':
                s += gaussian_mask
            noise -= gaussian_mask
        
        noise = torch.clip(noise, 0, 1)  # Ensure noise is non-negative
        label_sequences = torch.stack((p, s, noise), axis=-1)
        return label_sequences.type(torch.float32)

    '''
    def contains_pick(self, start_time: UTCDateTime, end_time: UTCDateTime) -> bool:
        """
        Check if there is any pick within the specified time range.
        """
        for pick in self.all_phase_picks.values():
            if start_time <= pick['time'] <= end_time:
                print(f"Found pick {pick['event_name']} at {pick['time']} within the range {start_time} to {end_time}.")
                return True'''
    
    def build_pick_time_index(self):
        print("Building pick time index for fast lookup...")
        self._pick_times = sorted((pick['time'], pick['event_name']) for pick in self.all_phase_picks.values())
        self._times = [t for t, _ in self._pick_times]

    def contains_pick(self, start_time: UTCDateTime, end_time: UTCDateTime) -> bool:
        """
        Fast check if there is any pick within the specified time range using binary search.
        """
        if not hasattr(self, '_pick_times'):
            self.build_pick_time_index()
        
        i = bisect.bisect_left(self._times, start_time)
        if i < len(self._times) and self._times[i] <= end_time:
            # Optionally print info:
            #print(f"Found pick {self._pick_times[i][1]} at {self._pick_times[i][0]} within the range {start_time} to {end_time}.")
            return True
        return False

    def get_noise_trace(self):
        """
        Get a random trace from the full data.
        """
        cropped = Stream()
        # there are gaps in the data, so we need to keep trying until we find a valid trace
        while len(cropped) < 3 or any(np.sum(tr.data) == 0 for tr in cropped):
            # randomly select a start time within the data range
            if self.use_daily_datastructure:
                # select two consecutive random days
                start_day = np.random.choice(list(self.daily_data.keys()))
                next_day = (UTCDateTime(start_day) + 86400).strftime('%Y-%m-%d')  # add one day
                if next_day in self.daily_data:
                    # next day waveform data exists
                    # we can choose a random window [start_day 0h00 to next_day 23h59]
                    range_end = UTCDateTime(next_day) + 86400 - self.window_size_in_seconds - 1
                else:
                    # next day waveform data does not exist
                    # we can choose a random window [start_day 0h00 to start_day 23h59]
                    range_end = UTCDateTime(start_day) + 86400 - self.window_size_in_seconds - 1

                random_ts = np.random.uniform(UTCDateTime(start_day).timestamp, range_end.timestamp)
                random_time = UTCDateTime(random_ts)
                end_time = random_time + self.window_size_in_seconds - 1. / self.fs
            
            else:
                # using fully preloaded data organized in a single Stream object
                random_ts = np.random.uniform(self.start_time.timestamp, (self.end_time - self.window_size_in_seconds).timestamp)
                random_time = UTCDateTime(random_ts)
                end_time = random_time + self.window_size_in_seconds - 1. / self.fs

            # check if the random time contains a pick
            # start the check before the random time. The event might start before but still end within the window.
            if not self.contains_pick(random_time - self.window_size_in_seconds, end_time):
                if self.use_daily_datastructure:
                    # now we can load the data for the selected window
                    # the data is for the full day, so we need to slice it
                    # get the days, where the random window starts and ends
                    start_day = random_time.strftime('%Y-%m-%d')
                    end_day = (random_time + self.window_size_in_seconds).strftime('%Y-%m-%d')
                    st = self.daily_data[start_day]
                    if end_day != start_day:
                        st += self.daily_data[end_day]
                    cropped = st.slice(
                        starttime=random_time, 
                        endtime=end_time,
                    ).copy().merge(method=1, fill_value=0)

                else:
                    cropped = self.full_data.slice(
                        starttime=random_time, 
                        endtime=end_time,
                    ).copy().merge(method=1, fill_value=0)
        
        if not all(len(tr.data) == self.window_size_samples for tr in cropped):
            print(f"Warning: Cropped data (NOISE) length does not match window size {self.window_size_samples}.")
            print(f"Cropped data lengths: {[len(tr.data) for tr in cropped]}")

        # pad start and end of traces
        cropped.trim(
            starttime=random_time,
            endtime=end_time,
            pad=True,
            fill_value=0,
        )

        cropped = insight_data_utils.ensure_crop_length(cropped, self.window_size_samples)

        assert len(cropped[0].data) == self.window_size_samples, \
        f"Cropped data length {len(cropped[0].data)} does not match window size {self.window_size_samples}.\
            Start time: {random_time}, End time: {random_time + self.window_size_in_seconds - 1. / self.fs}"
            
        return cropped


    def get_number_of_noise_traces(self):
        return self.__len__() - len(self.events_with_picks)


    def __len__(self):
        if self.train == 'train':
            # Training set might include noise traces
            return int((1 + self.noise_fraction) * len(self.events_with_picks))
        else:
            # Validation and test sets do not include noise traces
            return len(self.events_with_picks)
    
    def __getitem__(self, idx):
        # ============================================================
        # Select event trace and generate label sequences
        # ============================================================
        if self.train == 'train' and idx >= len(self.all_event_names):
            #
            # Randomly selected noise trace
            #

            p_and_s = torch.zeros((self.window_size_samples, 2), dtype=torch.float32)
            noise = torch.ones((self.window_size_samples, 1), dtype=torch.float32)
            label_sequences = torch.cat((p_and_s, noise), dim=1)  # [window_size_samples, 3]
            st = self.get_noise_trace()
            event_name = 'noise'
        else:
            #
            # Randomly selected event trace, that contains P and/or S picks
            #
            
            # randomly select a pick
            event_name = self.all_event_names[idx]
            picks = self.events_with_picks[event_name]
            pick = np.random.choice(picks)

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
        # TODO: do we filter?? and do we do it here??
        assert all(len(tr.data) == self.window_size_samples for tr in st), \
            f"All traces must have {self.window_size_samples} samples, but got {[len(tr.data) for tr in st]}."
        st.taper(max_percentage=0.01)
        assert all(len(tr.data) == self.window_size_samples for tr in st), \
            f"All traces must have {self.window_size_samples} samples, but got {[len(tr.data) for tr in st]}."
        st.filter('highpass', freq=0.05)
        assert all(len(tr.data) == self.window_size_samples for tr in st), \
            f"All traces must have {self.window_size_samples} samples, but got {[len(tr.data) for tr in st]}."


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

    def get_trace_around_time(
        self,
        time: UTCDateTime,
        window_size_in_seconds: float | None = None,
    ) -> torch.Tensor:
        """
        Get a trace around a given time. The target time will be in the center of the trace.
        The trace will be cropped to the specified window size or the default window size if not provided.
        The trace will be preprocessed (normalized, tapered, filtered) as specified in the class parameters.

        :param time: UTCDateTime object representing the time to get the trace around.
        :param window_size_in_seconds: float, optional. If provided, the trace will be cropped to this size.
        If None, the default window size will be used (self.window_size_in_seconds).
        :return: Torch tensor containing the trace data.
        """
        if window_size_in_seconds is None:
            window_size_in_seconds = self.window_size_in_seconds

        if self.use_daily_datastructure:
            # get the day of the time
            day = time.strftime('%Y-%m-%d')
            st = self.daily_data[day]
            cropped = st.slice(
                starttime=time - window_size_in_seconds / 2,
                endtime=time + window_size_in_seconds / 2,
            )
        else:
            cropped = self.full_data.slice(
                starttime=time - window_size_in_seconds / 2,
                endtime=time + window_size_in_seconds / 2,
            )

        st = cropped.copy().merge(method=1, fill_value=0)

        st = insight_data_utils.ensure_crop_length(st, self.window_size_samples)
        st = insight_data_utils.normalize_trace(st, method=self.norm_type)
        st.taper(max_percentage=0.01)
        st.filter('highpass', freq=0.05)

        if self.components == 'ZNE':
            compoonents_list = ['Z', 'N', 'E']
        if self.components == 'UVW':
            compoonents_list = ['U', 'V', 'W']
        traces = [torch.tensor(st.select(component=comp)[0].data, dtype=torch.float32) for comp in compoonents_list]
        traces = torch.stack(traces, dim=1)  # [window_size, 3]
        
        return traces

class InsightDataLit(SeisbenchDataLit):
    def __init__(
            self,
            data_dir: str | Path,
            window_size_samples: int = 32768, # 27 minutes at 20Hz -> see MarsQuakeNet
            norm_type: str = 'std',
            components: str = 'ZNE',
            fs: int = 20,
            default_uncertainty: float = 20.0,
            overwrite_uncertainties: bool = False,
            load_inventory: bool = True,
            noise_fraction: float = 0.5,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.dataset_kwargs = {
            'window_size_samples': window_size_samples,
            'norm_type': norm_type,
            'components': components,
            'fs': fs,
            'default_uncertainty': default_uncertainty,
            'overwrite_uncertainties': overwrite_uncertainties,
            'load_inventory': load_inventory,
            'noise_fraction': noise_fraction,
            'pre_slice': False,
            'merge_when_loading': False, 
            'preload_waveform_data': False,
        }

        self.setup()
    
    def setup(self):
        self.dataset_train = InsightDataset(
            data_dir=self.data_dir, 
            train='train', 
            use_daily_datastructure=True,
            **self.dataset_kwargs)
        
        self.dataset_val = InsightDataset(
            data_dir=self.data_dir, 
            train='dev', 
            use_daily_datastructure=False,
            load_only_event_data=True,
            **self.dataset_kwargs
            )
        
        self.dataset_test = InsightDataset(
            data_dir=self.data_dir, 
            train='test',
            use_daily_datastructure=False,
            load_only_event_data=True,
            **self.dataset_kwargs,
            )

        self.d_data = 3  # Number of components (Z, N, E or U, V, W)

class InsightPretrainDataLit(SeisbenchDataLit):
    def __init__(
            self,
            data_dir: str | Path,
            window_size_samples: int = 32768,  # 27 minutes at 20Hz
            norm_type: str = 'std',
            components: str = 'ZNE',
            fs: int = 20,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.data_dir = data_dir
        self.dataset_kwargs = {
            'window_size_samples': window_size_samples,
            'norm_type': norm_type,
            'components': components,
            'fs': fs,
        }

        self.setup()

    def setup(self):
        self.dataset_train = InsightPretrainingDataset(
            data_dir=self.data_dir,
            train='train',
            **self.dataset_kwargs)

        self.dataset_val = InsightPretrainingDataset(
            data_dir=self.data_dir,
            train='dev',
            **self.dataset_kwargs)

        self.dataset_test = InsightPretrainingDataset(
            data_dir=self.data_dir,
            train='test',
            **self.dataset_kwargs)

        self.d_data = 3  # Number of components (Z, N, E or U, V, W)

# ============================================================
# Tests
# ============================================================

def pretrained_dataset_test():
    loaders = {}
    splits = ['train', 'dev', 'test']
    for split in splits:
        dataset = InsightPretrainingDataset(
            data_dir='dataloaders/data/insight',
            window_size_samples=32768,  # 27 minutes at 20Hz
            norm_type='std',
            components='ZNE',
            fs=20,
            train=split,
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        loaders[split] = loader

    print('Succesfully prepared Insight pretraining dataset loaders.\n')

    total_examples = sum(len(loader.dataset) for loader in loaders.values())
    print(f"Total number of examples across all splits: {total_examples}")

    for split, loader in loaders.items():
        print(f"Number of {split} examples: {len(loader.dataset)}")
        print(f"Number of {split} batches: {len(loader)}")

    print('\n')
    for split, loader in loaders.items():
        for i, (traces, labels) in enumerate(tqdm(loader, desc=f"Checking {split} loader")):
            assert traces.shape[-1] == 3, f"Expected 3 components, got {traces.shape[1]} in {split} loader."
            assert traces.shape[1] == 32768, f"Expected window size of 32768 samples, got {traces.shape[1]} in {split} loader."
            # check if traces contain NaNs or Infs
            assert not torch.isnan(traces).any(), f"NaN values found in traces of {split} loader."
            assert not torch.isinf(traces).any(), f"Inf values found in traces of {split} loader."
            # check if labels are all zeros

    print("All checks passed for the Insight pretraining dataset loaders.\n\n\n\n")


def lightning_dataset_test():
    dataset_config = {
        'data_dir': 'dataloaders/data/insight',
        'window_size_samples': 32768,  # 27 minutes at 20Hz
        'norm_type': 'std',
        'components': 'ZNE',
        'fs': 20,
        'default_uncertainty': 20.0,
        'overwrite_uncertainties': False,
        'load_inventory': False,
        'noise_fraction': 0.5,  # +50% noise traces, so 1/3 of the dataset is noise
        'pre_slice': False,
        'merge_when_loading': False,  # merge traces when loading
        'preload_waveform_data': True,
    }

    loader_config = {
        'batch_size': 32,
        'num_workers': 0,
        'pin_memory': True,
    }

    dataset = InsightDataLit(**dataset_config)
    train_loader = DataLoader(dataset.dataset_train, **loader_config)
    val_loader = DataLoader(dataset.dataset_val, **loader_config)
    test_loader = DataLoader(dataset.dataset_test, **loader_config)
    print(f"Number of training examples: {len(dataset.dataset_train)}")
    print(f"Number of validation examples: {len(dataset.dataset_val)}")
    print(f"Number of test examples: {len(dataset.dataset_test)}")

    # print number of batches in each loader
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")


def val_split_test():

    val_event_names = []
    for i in range(20):
        dataset = InsightDataset(
            data_dir='dataloaders/data/insight',
            window_size_samples=32768,  # 27 minutes at 20Hz
            norm_type='std',
            components='ZNE',
            fs=20,
            train='dev',
            default_uncertainty=20.0,
            overwrite_uncertainties=False,
            load_inventory=False,
            noise_fraction=0.5,  # +50% noise traces, so 1/3 of the dataset is noise
            pre_slice=False,
            merge_when_loading=False,  # merge traces when loading
            preload_waveform_data=False,
        )
        val_event_names.append(dataset.all_event_names)
    
    # check that the validation event names are the same for all datasets
    for i in range(1, len(val_event_names)):
        assert val_event_names[i] == val_event_names[0], \
            f"Validation event names are different: {val_event_names[i]} != {val_event_names[0]}"
    print("Validation event names are the same for all datasets.\n\n\n\n")

def get_event_type_summary(partition: str = 'train', print_events: bool = True):
    dataset = InsightDataset(
        data_dir='dataloaders/data/insight',
        window_size_samples=32768,  # 27 minutes at 20Hz
        norm_type='std',
        components='ZNE',
        fs=20,
        train=partition,
        default_uncertainty=20.0,
        overwrite_uncertainties=False,
        load_inventory=False,
        noise_fraction=0.5,  # +50% noise traces, so 1/3 of the dataset is noise
        pre_slice=False,
        merge_when_loading=False,  # merge traces when loading
        preload_waveform_data=True,
    )

    catalogue = dataset.catalogue
    dev_event_names = dataset.all_event_names
    print(f"Number of events in the {partition} set: {len(dev_event_names)}")
    print('\n\n')

    counts = {}

    for i, event_name in enumerate(dev_event_names):
        event_entry = catalogue[catalogue['name'] == event_name]
        key = f'{event_entry["quality"].iloc[0]}_{event_entry["type"].iloc[0]}'
        if key not in counts:
            counts[key] = 0
        counts[key] += 1

        if print_events:
            print(f"Event: {event_name}, quality: {event_entry['quality'].iloc[0]}, type: {event_entry['type'].iloc[0]}")


    print('\n\n')
    print(f"Counts of event types and qualities in the {partition} set:")
    for key, count in sorted(counts.items(), key=lambda x: x[0]):
        print(f"{key}: {count}")
    print('\n\n')


def print_all_summaries():
    for partition in ['dev', 'train', 'test']:
        print(f"Summary for {partition} partition:")
        get_event_type_summary(partition=partition, print_events=False)
        print('\n\n')


def train_loader_iterator_test(train='train', epochs=100):
    dataset = InsightDataset(
        data_dir='dataloaders/data/insight',
        window_size_samples=32768,  # 27 minutes at 20Hz
        norm_type='std',
        components='ZNE',
        fs=20,
        train=train,
        default_uncertainty=20.0,
        overwrite_uncertainties=False,
        load_inventory=False,
        noise_fraction=0.5,  # 0.5 means +50% noise traces, so 1/3 of the dataset is noise
        pre_slice=False,  # pre-slice the training set
        merge_when_loading=False,  # merge traces when loading
        return_event_name=True,
        preload_waveform_data=False,
        load_only_event_data=train != 'train',  # Load only event data for validation and test
        use_daily_datastructure=train == 'train',
    )

    st_short = Stream()
    for c in ['Z', 'N', 'E']:
        # add random np array of length 32767 as trace to the stream
        data = np.random.randn(32768).astype(np.float32)
        st_short.append(obspy.Trace(data=data, header={
            'network': 'XB',
            'station': 'ELYSE',
            'location': '00',
            'channel': c,
            'sampling_rate': 20.0,
            'starttime': UTCDateTime(0),
        }))
        


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,)
    
    t, l, names = next(iter(dataloader))
    print('\n\nTest output')
    print(f'Traces shape: {t.shape}') 
    print(f'Label sequences shape: {l.shape}')  
    print(f'Number of examples in dataset: {len(dataset)}')
    print(f'Number of unique events: {len(dataset.all_event_names)}')
    print(f'Number of noise traces: {dataset.get_number_of_noise_traces()}')
    '''
    
    for i in range(t.shape[0]):
        plot_example(
            cropped_st=t[i],
            label_sequences=l[i],
            title=f'Event: {names[i]}',
            #clip=1.0
        )
    '''

    # iterate through the whole dataset 5 times and check the output shapes
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch + 1}')):
            traces, labels, event_names = batch
            assert traces.shape[-1] == 3, f"Expected traces to have 3 components, got {traces.shape[-1]} components."
            assert traces.shape[1] == dataset.window_size_samples, \
                f"Expected traces to have {dataset.window_size_samples} samples, got {traces.shape[1]} samples."
            #print(f'Batch {batch_idx + 1}:')
            #print(f'Traces shape: {traces.shape}, Labels shape: {labels.shape}, Event names: {event_names}')
            '''if i == 0:
                plot_example(
                    cropped_st=traces[0],
                    label_sequences=labels[0],
                    title=f'Event: {event_names[0]}',
                )'''

if __name__ == "__main__":
    #print_all_summaries()
    #lightning_dataset_test()
    '''
    for train in ['train', 'dev', 'test']:
        print(f"Testing {train} dataset:".upper())
        train_loader_iterator_test(train=train)
        print('\n\n')

    '''
    pretrained_dataset_test()
        
