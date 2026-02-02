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


DATA_PATH = 'dataloaders/data/insight/'


# load catalogues
# load v14 catalogue
catalogue_v14 = insight_data_utils.load_InSight_catalogue('dataloaders/data/insight/catalogs/events_InSIght_v14.pkl')
# load MQNet catalogue
catalogue_MQ = pd.read_csv('dataloaders/data/insight/MQNet_DeepCatalogue.csv')
# load MQNet alldetections
alldetections_MQ = pd.read_csv('dataloaders/data/insight/MQNet_alldetections.csv')


# get events with phase picks from v14 catalogue
all_phase_picks = insight_data_utils.get_event_picks(catalogue_v14)
events_with_picks = insight_data_utils.get_dict_by_event_name(all_phase_picks)
print(f"Number of events with phase picks: {len(events_with_picks)}")
print(f'Total number of phase picks before processing: {sum([len(picks) for picks in events_with_picks.values()])}')

WINDOW_SIZE_SAMPLES = 8192*4
FS = 20.0
WINDOW_SIZE_SECONDS = WINDOW_SIZE_SAMPLES / FS

print(f"\nWindow size in samples: {WINDOW_SIZE_SAMPLES}")
print(f"Window size in Hz: {FS} Hz")
print(f"Window size in seconds: {WINDOW_SIZE_SECONDS:.2f} s")

# Build sorted lists for start and end times
start_times = catalogue_v14['start_time'].apply(UTCDateTime).tolist()
end_times = catalogue_v14['end_time'].apply(UTCDateTime).tolist()
names = catalogue_v14['name'].tolist()

# Sort by start_time
sorted_events = sorted(zip(start_times, end_times, names), key=lambda x: x[0])
sorted_start_times = [e[0] for e in sorted_events]

filtered_catalogue = catalogue_v14[~catalogue_v14['name'].isin(events_with_picks.keys())]
start_times_filtered = filtered_catalogue['start_time'].apply(UTCDateTime).tolist()
end_times_filtered = filtered_catalogue['end_time'].apply(UTCDateTime).tolist()
names_filtered = filtered_catalogue['name'].tolist()

# Sort by start_time for filtered catalogue
sorted_events_filtered = sorted(zip(start_times_filtered, end_times_filtered, names_filtered), key=lambda x: x[0])
sorted_start_times_filtered = [e[0] for e in sorted_events_filtered]

# MQNet_alldetections (similar logic)
start_times_MQ = alldetections_MQ['utc_start'].apply(UTCDateTime).tolist()
end_times_MQ = alldetections_MQ['utc_end'].apply(UTCDateTime).tolist()
names_MQ = alldetections_MQ['event_name'].tolist()
sorted_events_MQ = sorted(zip(start_times_MQ, end_times_MQ, names_MQ), key=lambda x: x[0])
sorted_start_times_MQ = [e[0] for e in sorted_events_MQ]

def check_contains_event(
    start_time: UTCDateTime, 
    end_time: UTCDateTime, 
    print_output: bool = False, 
    event_name: str = None, 
    ignore_events_with_picks: bool = False,
    ignore_events_T: bool = False,
    ) -> bool:
    """
    Fast check if the window specified by start_time and end_time overlaps with any event in the v14 catalogue and MQNet_alldetections.
    Uses sorted lists and bisect for efficient lookup.
    """

    # Find possible overlaps using bisect
    if ignore_events_with_picks:
        idx = bisect.bisect_left(sorted_start_times_filtered, end_time)
    else:
        idx = bisect.bisect_left(sorted_start_times, end_time)
    for i in range(max(0, idx - 10), min(len(sorted_events), idx + 10)):
        if ignore_events_with_picks:
            event_start, event_end, found_name = sorted_events_filtered[i]
        else:
            event_start, event_end, found_name = sorted_events[i]
        if ignore_events_T and found_name.startswith('T'):
            continue
        if event_start < end_time and event_end > start_time:
            if event_name is not None and found_name != event_name:
                if print_output:
                    print(f'Found in v14 catalogue')
                    print(f"Event {found_name} overlaps with current event {event_name}.")
                return True
            elif event_name is None:
                if print_output:
                    print(f'Found in v14 catalogue')
                    print(f"Event {found_name} overlaps current event.")
                return True

    idx_MQ = bisect.bisect_left(sorted_start_times_MQ, end_time)
    for i in range(max(0, idx_MQ - 10), min(len(sorted_events_MQ), idx_MQ + 10)):
        event_start, event_end, found_name = sorted_events_MQ[i]
        if ignore_events_T and found_name.startswith('T'):
            continue
        if event_start < end_time and event_end > start_time:
            if event_name is not None and found_name != event_name:
                if print_output:
                    print(f'Found in MQNet_alldetections')
                    print(f"Event {found_name} overlaps with current event {event_name}.")
                return True
            elif event_name is None:
                if print_output:
                    print(f'Found in MQNet_alldetections')
                    print(f"Event {found_name} overlaps with current event.")
                return True

    return False



# find negative examples around the events with picks
NUM_WINDOWS = 3.0
DECREMENT = 0.2
BUFFER_IN_SECONDS = 5 * 60  # 5 minutes buffer

negative_examples = []
events_with_noise_traces = []


print(f'\nStep 1: Find noise traces around events with phase picks')
for event_name, picks in tqdm(events_with_picks.items(), desc="Finding negative examples"):
    start_time = catalogue_v14[catalogue_v14['name'] == event_name]['start_time'].values[0]
    end_time = catalogue_v14[catalogue_v14['name'] == event_name]['end_time'].values[0]
    start_time = UTCDateTime(start_time) - BUFFER_IN_SECONDS  # 5 min buffer
    end_time = UTCDateTime(end_time) + BUFFER_IN_SECONDS  # 5 min buffer

    # check if the windows before contain any events
    before_start = start_time - NUM_WINDOWS * WINDOW_SIZE_SECONDS
    before_end = start_time

    if not check_contains_event(before_start, before_end, event_name=event_name, print_output=False):
        negative_examples.append((before_start, before_end, event_name, 'before'))
        events_with_noise_traces.append(event_name)
    else:
        # the window before the event contains an event. try to find a smaller window
        i = NUM_WINDOWS
        while i >= 1 and check_contains_event(before_start, before_end, print_output=False):
            i -= DECREMENT
            before_start = start_time - i * WINDOW_SIZE_SECONDS
            before_end = start_time
        if i >= 1 and not check_contains_event(before_start, before_end, print_output=False):
                negative_examples.append((before_start, before_end, event_name, 'before'))
                events_with_noise_traces.append(event_name)


    # check if the windows after contain any events
    after_start = end_time
    after_end = end_time + NUM_WINDOWS * WINDOW_SIZE_SECONDS
    
    if not check_contains_event(after_start, after_end, event_name=event_name, print_output=False):
        negative_examples.append((after_start, after_end, event_name, 'after'))
        events_with_noise_traces.append(event_name)
    else:
        # the window after the event contains an event. try to find a smaller window
        i = NUM_WINDOWS
        while i >= 1 and check_contains_event(after_start, after_end, print_output=False):
            i -= DECREMENT
            after_start = end_time
            after_end = end_time + i * WINDOW_SIZE_SECONDS
        if i >= 1 and not check_contains_event(after_start, after_end, print_output=False):
                negative_examples.append((after_start, after_end, event_name, 'after'))
                events_with_noise_traces.append(event_name)

events_with_noise_traces = list(set(events_with_noise_traces))  # remove duplicates
print(f"Number of negative examples found: {len(negative_examples)}")
print(f"Number of events with noise traces: {len(events_with_noise_traces)}")
print(f'Percentage of pick events with noise traces: {len(events_with_noise_traces) / len(events_with_picks) * 100:.2f}%')


'''
Now we have a list of negative examples (negative_examples), that do not contain any other events in the v14 catalogue or MQNet_alldetections.
They should not contain S, D or T events (S -> v14, D -> MQNet, T -> super high frequency events).

The list events_with_noise_traces contains the names of the events that have noise traces, either before or after the event with picks.
'''

# for each pick, check if it overlaps with another pick
picks_with_overlap = []

print(f'\nStep 2: Check overlapping events for positive examples')
# loop though the events with picks
sorted_events_with_picks = [e for e in sorted_events if e[2] in events_with_picks.keys()]

for event_name, picks in tqdm(events_with_picks.items(), desc="Checking for overlapping picks"):
    start_time = catalogue_v14[catalogue_v14['name'] == event_name]['start_time'].values[0]
    end_time = catalogue_v14[catalogue_v14['name'] == event_name]['end_time'].values[0]
    start_time = UTCDateTime(start_time) - WINDOW_SIZE_SECONDS
    end_time = UTCDateTime(end_time) + WINDOW_SIZE_SECONDS

    for i in range(len(sorted_events_with_picks)):
        other_start, other_end, other_event_name = sorted_events_with_picks[i]
        other_start -= WINDOW_SIZE_SECONDS
        other_end += WINDOW_SIZE_SECONDS
        if other_event_name == event_name:
            continue
        if other_start < end_time and other_end > start_time:
            picks_with_overlap.append((event_name, other_event_name))

# convert to dictionary
picks_with_overlap_dict = {}
for event_name, other_event_name in picks_with_overlap:
    if event_name not in picks_with_overlap_dict:
        picks_with_overlap_dict[event_name] = []
    picks_with_overlap_dict[event_name].append(other_event_name)

print(f"Number of events with overlapping picks: {len(picks_with_overlap_dict)}")



# ========================================================================================================================
# check waveform data
# ========================================================================================================================

# load waveform data by day
print(f'\nStep 3: Check waveform data for events')
daily_data = insight_data_utils.load_waveform_data_by_day()

# check if we have waveform data for windows with and without picks

noise_traces_with_waveform_data = []
num_events_without_daily_data = 0
num_filled_with_zeros = 0
num_empty_traces_after_cropping = 0
for start_time, end_time, event_name, position in tqdm(negative_examples, desc="Checking waveform data for noise traces"):
    start_day = start_time.strftime('%Y-%m-%d')
    end_day = end_time.strftime('%Y-%m-%d')
    
    st = Stream()
    if start_day in daily_data:
        st = daily_data[start_day]
    if start_day != end_day and end_day in daily_data:
        st += daily_data[end_day]

    if len(st) == 0:
        #print(f'No waveform data for {start_day} around event {event_name} ({position}) from {start_time} to {end_time}.')
        num_events_without_daily_data += 1
        continue

    cropped = st.slice(start_time, end_time)
    cropped.copy().merge(method=1, fill_value=0)

    cropped = cropped.trim(start_time, end_time, pad=True, fill_value=0)
    if len(cropped) == 0:
        print(f'No waveform data for {start_day} around event {event_name} ({position}) from {start_time} to {end_time}.')
        num_empty_traces_after_cropping += 1
        continue

    # check if all traces in st are filled with zeros
    if all(np.all(trace.data == 0) for trace in cropped):
        #print(f'All traces are filled with zeros for {start_day} around event {event_name} ({position}) from {start_time} to {end_time}.')
        num_filled_with_zeros += 1
        continue

    noise_traces_with_waveform_data.append((start_time, end_time, event_name, position))

# check if we have waveform data for windows with picks
events_with_picks_and_waveform_data = []
for event_name, picks in tqdm(events_with_picks.items(), desc="Checking waveform data for events with picks"):
    start_time = catalogue_v14[catalogue_v14['name'] == event_name]['start_time'].values[0]
    end_time = catalogue_v14[catalogue_v14['name'] == event_name]['end_time'].values[0]
    start_time = UTCDateTime(start_time) - WINDOW_SIZE_SECONDS
    end_time = UTCDateTime(end_time) + WINDOW_SIZE_SECONDS

    start_day = start_time.strftime('%Y-%m-%d')
    end_day = end_time.strftime('%Y-%m-%d')
    
    st = Stream()
    if start_day in daily_data:
        st = daily_data[start_day]
    if start_day != end_day and end_day in daily_data:
        st += daily_data[end_day]

    if len(st) == 0:
        continue

    cropped = st.slice(start_time, end_time)
    cropped.copy().merge(method=1, fill_value=0)

    cropped = cropped.trim(start_time, end_time, pad=True, fill_value=0)
    if len(cropped) == 0:
        continue

    # check if all traces in st are filled with zeros
    if all(np.all(trace.data == 0) for trace in cropped):
        continue

    events_with_picks_and_waveform_data.append(event_name)

events_with_picks_and_waveform_data = list(set(events_with_picks_and_waveform_data))  # remove duplicates

print(f'Number of noise traces with waveform data: {len(noise_traces_with_waveform_data)}')
print(f"Number of events with picks and waveform data: {len(events_with_picks_and_waveform_data)}")

# remove events with picks that do not have waveform data
events_with_picks = {k: v for k, v in events_with_picks.items() if k in events_with_picks_and_waveform_data}


results = {
    'events_with_picks': events_with_picks, # dict of events that have annotated picks and corresponding waveform data
    'events_with_noise_traces': events_with_noise_traces, # list of event names that 1) have phase picks and 2) have noise traces before or after the event
    'picks_with_overlap': picks_with_overlap_dict, # dict of events that have overlapping picks with other events, useful for generating label sequences
    'noise_traces_with_waveform_data': noise_traces_with_waveform_data, # list of tuples (start_time, end_time, event_name, position) for noise traces that have waveform data
    # metadata
    'WINDOW_SIZE_SAMPLES': WINDOW_SIZE_SAMPLES,
    'FS': FS,
    'WINDOW_SIZE_SECONDS': WINDOW_SIZE_SECONDS,
}

# save results to file
with open(f'dataloaders/data/insight/dataset_prep_faster_{WINDOW_SIZE_SAMPLES}.pkl', 'wb') as f:
    pickle.dump(results, f)