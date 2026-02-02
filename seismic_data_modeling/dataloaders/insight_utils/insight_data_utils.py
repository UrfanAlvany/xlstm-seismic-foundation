import os
import pickle
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
from tqdm import tqdm


phase_list_P = ["p", "pP", "P", "P1", "Pg", "Pn", "PmP", "pwP", "pwPm", "PP", "PPP"]
phase_list_S = ["s", "S", "S1", "Sg", "SmS", "Sn", "SS", "ScS", "SSS"]

months_dict = {
    1: 'JAN',
    2: 'FEB',
    3: 'MARCH',
    4: 'APRIL',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEPT',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC'
}

def get_all_mseed_files(data_folder: str, format: str = 'ZNE') -> list:
    """
    Get all files in the specified folder that end with f'{format}.mseed'.
    :param data_folder: Path to the folder containing waveform data files.
    :param format: Format of the waveform data files, one of ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']. Default is 'ZNE'.
                    is extended to i.e. ZNE.mseed or UVW.mseed.
    :return: List of file paths matching the specified format.
    """
    file_extension = f'{format}.mseed'
    all_files = []
    for dirpath, dirnames, filenames in os.walk(data_folder):
        for filename in filenames:
            if filename.endswith(file_extension):
                all_files.append(os.path.join(dirpath, filename))
    return all_files


def load_raw_waveform_data(data_folder: str, format: str = 'ZNE', fs: int = 20, merge: bool = True) -> Stream:
    """
    Load raw waveform data from a specified folder. Merge all traces according to component and time.

    :param data_folder: Path to the folder containing waveform data files.
    :param format: Format of the waveform data files, one of ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']. Default is 'ZNE'.
    :param fs: target sampling frequency for resampling traces. Default is 20 Hz.
    :param merge: Whether to merge traces with the same station and channel. Default is True.
    :return: An ObsPy Stream object containing the loaded waveform data.
    """
    if format not in ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']:
        raise ValueError("Unsupported format. Supported formats are: 'ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC'.")
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"The specified data folder does not exist: {data_folder}")
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"The specified path is not a directory: {data_folder}")
    
    # Initialize an empty Stream object to hold the waveform data
    all_data = Stream()
    num_files = 0

    # Get all files to process for progress bar
    all_files = get_all_mseed_files(data_folder, format)

    # load all files
    for file_path in tqdm(all_files, desc="Loading MiniSEED files"):
        try:
            st = read(file_path)
            for trace in st:
                if trace.stats.sampling_rate != fs:
                    print('Resampling!')
                    trace.resample(fs)
            #st.merge(method=1, fill_value=0)  # Merge traces recorded on the same day
            all_data += st
            num_files += 1
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    if merge:
        print('Merging traces. This may take a while...')
        all_data.merge(method=1, fill_value=0)  # Merge traces with the same station and channel

    if 'ZNE' in format:
        all_data.sort(['component'], reverse=True)  # Sort traces by component (Z, N, E)
    
    # should load 214 files (for 2019)
    # some recording have not been converted to ZNE and are only present as UVW

    # print summary of loaded data
    print(f"Loaded {num_files} files from {data_folder} with format {format}.")

    return all_data


def load_waveform_data_by_day(
        data_folder: str = 'dataloaders/data/insight',
        format: str = 'ZNE',
        fs: int = 20,
) -> dict:
    """
    Load waveform data from a specified folder, grouped by day.
    
    :param data_folder: Path to the folder containing waveform data files.
    :param format: Format of the waveform data files, one of ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']. Default is 'ZNE'.
    :param fs: Target sampling frequency for resampling traces. Default is 20 Hz.
    :return: Dictionary mapping day strings (YYYY-MM-DD) to ObsPy Stream objects containing the loaded waveforms.
    """
    if format not in ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']:
        raise ValueError("Unsupported format. Supported formats are: 'ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC'.")
    
    daily_data = {}
    all_files = get_all_mseed_files(data_folder, format)
    num_files = 0

    for file_path in tqdm(all_files, desc="Loading MiniSEED files by day"):
        try:
            st = read(file_path)
            for trace in st:
                if trace.stats.sampling_rate != fs:
                    trace.resample(fs)
            # Extract the day from the start time of the first trace
            day_str = st[0].stats.starttime.strftime('%Y-%m-%d')
            
            # If the day is not in the dictionary, initialize it
            if day_str not in daily_data:
                daily_data[day_str] = Stream()
            
            # Add the traces to the corresponding day's stream
            daily_data[day_str] += st.merge(method=1, fill_value=0)  # Merge traces recorded on the same day
            num_files += 1
            
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    print(f"Loaded {num_files} files from {data_folder} with format {format}.")
    
    return daily_data


def get_metadata_inventory(path: str = '../data/dataless.XB.ELYSE.seed') -> Inventory:
    """
    Reads the inventory file and returns an Inventory object.
    
    TODO: double check inventory file.
    inv = obspy.read_inventory('../data/dataless.XB.ELYSE.seed') # 45.9MB
    inv = obspy.read_inventory('../data/ELYSE.dataless') # 5.7MB
    
    :param path: Path to the inventory file.
    :return: Inventory object.
    """
    try:
        inv = obspy.read_inventory(path)
        print(f"Inventory loaded from {path}")
        return inv
    except Exception as e:
        print(f"Failed to read inventory from {path}: {e}")
        return None
    

def load_InSight_catalogue(path: str = '../data/catalogs/events_InSIght_v14.pkl') -> DataFrame:
    """
    Loads the InSight catalogue from a pickle file.
    
    :param path: Path to the pickle file containing the catalogue.
    :return: DataFrame containing the catalogue data.
    """
    try:
        with open(path, 'rb') as f:
            catalogue = pickle.load(f)
        print(f"Catalogue loaded from {path}")
        return catalogue
    except Exception as e:
        print(f"Failed to load catalogue from {path}: {e}")
        return None


def get_sample_offset(event_time: UTCDateTime, waveform: obspy.Trace | Stream) -> int:
    """
    Calculate the sample offset for a given event time in a waveform trace or stream.
    
    :param event_time: UTCDateTime object representing the event time.
    :param waveform: obspy.Trace or Stream object containing the waveform data.
    :return: int, the sample offset from the start of the waveform.
    """
    if isinstance(waveform, Stream):
        waveform = waveform[0]
    
    fs = waveform.stats.sampling_rate
    start_time = waveform.stats.starttime
    offset = int((event_time - start_time) * fs)
    return offset


def get_event_picks(catalogue: DataFrame) -> dict:
    """
    Extract P and S picks from the InSight catalogue and return them as a dictionary.
    Picks are nummerated with an ID starting from 0.
    Every entry is itself a dictionary with the following added keys:
    - 'event_name': Name of the event
    - 'pick_type': Type of the pick ('P' or 'S')
    :param catalogue: DataFrame containing the InSight catalogue with events and their picks.
    :return: Dictionary with pick IDs as keys and pick information as values."""

    pick_dict = {}
    pick_id = 0

    # loop through each event in the catalogue
    for idx, event in catalogue.iterrows():
        event_name = event['name']
        picks = event['picks']

        # loop through all picks for the event
        for pick in picks:
            if pick['phase_hint'] in phase_list_P:
                pick_type = 'P'
                # update pick (dict) with event name and pick type
                pick['event_name'] = event_name
                pick['pick_type'] = pick_type
                pick_dict[pick_id] = pick
                pick_id += 1
            elif pick['phase_hint'] in phase_list_S:
                pick_type = 'S'
                pick['event_name'] = event_name
                pick['pick_type'] = pick_type
                pick_dict[pick_id] = pick
                pick_id += 1
            else:
                continue
    return pick_dict


def get_all_P_picks(pick_dict: dict) -> dict:
    """
    Extract all P picks from the pick dictionary.
    
    :param pick_dict: Catalogue dictionary containing P and S picks.
    :return: Dictionary containing only P picks.
    """
    p_picks = {k: v for k, v in pick_dict.items() if v['pick_type'] == 'P'}
    return p_picks


def get_all_S_picks(pick_dict: dict) -> dict:
    """
    Extract all S picks from the pick dictionary.
    :param pick_dict: Catalogue dictionary containing P and S picks.
    :return: Dictionary containing only S picks.
    """
    filtered = {k: v for k, v in pick_dict.items() if v['pick_type'] == 'S'}
    return filtered


def get_unique_events(pick_dict: dict) -> set:
    """
    Get a set of unique event names from the pick dictionary.
    :param pick_dict: Catalogue dictionary containing P and S picks.
    :return: Set of unique event names.
    """
    unique_events = set()
    for pick in pick_dict.values():
        unique_events.add(pick['event_name'])
    return unique_events


def get_number_of_unique_events(pick_dict: dict) -> int:
    """
    Get the number of unique events in the pick dictionary.
    
    :param pick_dict: Catalogue dictionary containing P and S picks.
    :return: Number of unique events.
    """
    return len(get_unique_events(pick_dict))


def fill_missing_uncertainities(pick_dict: dict, default_uncertainty: float = 10.0, overwrite: bool = False) -> None:
    """
    Fill missing uncertainties in the pick dictionary with a default value.
    Dict is filled in place
    
    :param pick_dict: Catalogue dictionary containing P and S picks.
    :param default_uncertainty: Default uncertainty value to fill in. Default is 10.0.
    :param overwrite: If True, overwrite existing uncertainties with the default value. Default is False.
    """
    for pick in pick_dict.values():
        if overwrite or'lower_uncertainty' not in pick or pick['lower_uncertainty'] is None:
            pick['lower_uncertainty'] = default_uncertainty
        if overwrite or 'upper_uncertainty' not in pick or pick['upper_uncertainty'] is None:
            pick['upper_uncertainty'] = default_uncertainty


def find_first_picks(events_with_picks: dict) -> dict:
    """
    If an event has multiple P or S picks, find the first P pick and the first S pick.
    :param events_with_picks: Catalogue dictionary containing P and S picks for every event.
    return: Dictionary with event names as keys and lists of first P and S picks as values.
    """

    filtered_events = {}

    # Loop through each event and its picks
    for event_name, picks in events_with_picks.items():
        p_picks = [pick for pick in picks if pick['pick_type'] == 'P']
        s_picks = [pick for pick in picks if pick['pick_type'] == 'S']

        filtered_events[event_name] = []
        
        # find earliest P and S picks
        if len(p_picks) > 0:
            filtered_events[event_name].append(min(p_picks, key=lambda x: x['time']))
        if len(s_picks) > 0:
            filtered_events[event_name].append(min(s_picks, key=lambda x: x['time']))

        '''
        # debug
        if len(p_picks) > 1:
            print('\nFound P picks to remove:')
            print([(p['time'], p['lower_uncertainty']) for p in p_picks])
            print(f'min time: {min(p_picks, key=lambda x: x["time"])}')
        '''
    
    return filtered_events

def find_most_certain_picks(events_with_picks: dict) -> dict:
    """
    If an event has multiple P or S picks, find the P and S pick with the lowest 'upper_uncertainty'.
    :param events_with_picks: Catalogue dictionary containing P and S picks for every event.
    return: Dictionary with event names as keys and lists of most certain P and S picks as values.
    """

    filtered_events = {}

    # Loop through each event and its picks
    for event_name, picks in events_with_picks.items():
        p_picks = [pick for pick in picks if pick['pick_type'] == 'P']
        s_picks = [pick for pick in picks if pick['pick_type'] == 'S']

        filtered_events[event_name] = []
        
        # find earliest P and S picks
        if len(p_picks) > 0:
            filtered_events[event_name].append(min(p_picks, key=lambda x: x['upper_uncertainty']))
        if len(s_picks) > 0:
            filtered_events[event_name].append(min(s_picks, key=lambda x: x['upper_uncertainty']))

        '''
        # debug
        if len(p_picks) > 1:
            print('\nFound P picks to remove:')
            print([(p['time'], p['upper_uncertainty'], p['phase_hint']) for p in p_picks])
            print(f'min upper_uncertainty: {min(p_picks, key=lambda x: x["upper_uncertainty"])}')
        '''
    
    return filtered_events


def get_dict_by_event_name(pick_dict: dict) -> dict:
    """
    Create a dictionary mapping event names to their respective picks.
    
    :param pick_dict: Catalogue dictionary containing P and S picks.
    :return: Dictionary mapping event names to lists of picks.
    """
    event_dict = {}
    for pick in pick_dict.values():
        event_name = pick['event_name']
        if event_name not in event_dict:
            event_dict[event_name] = []
        event_dict[event_name].append(pick)
    return event_dict


def change_dtype(st: Stream, dtype: str = 'float32') -> None:
    """
    Change the data type of the traces in a Stream object.
    Change is inplace, so references to the full dataset should be avoided.
    
    :param st: Stream object containing the traces to change dtype.
    :param dtype: Desired data type, e.g., 'float32' or 'float64'.
    :return: Stream object with traces converted to the specified data type.
    """
    for trace in st:
        trace.data = trace.data.astype(dtype)


def utcdatetime_to_filename(event_time: UTCDateTime, components: str = 'ZNE') -> str:
    """
    Convert a UTCDateTime object to a filename format.
    
    :param event_time: UTCDateTime object representing the event time.
    :param components: Format of the waveform data files, one of ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']. Default is 'ZNE'.
    :return: String representing the filename in the format 'YYYY/MONTH/DD/DD.components.mseed'.
    """
    assert components in ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC'], "Unsupported format. Supported formats are: 'ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC'."

    year = event_time.year
    month = event_time.month
    day = event_time.day

    file_name = f'{int(year)}/{months_dict[month]}/{day:02d}/{day:02d}.{components}.mseed'
    return file_name


def load_waveforms_for_events(
        event_dict: dict,
        data_folder: str = 'dataloaders/data/insight',
        format: str = 'ZNE',
        fs: int = 20,
        window_size_in_seconds: float = 1638.4, # 32768 samples at 20 Hz -> roughly 27 minutes
        num_windows_around_event: int = 2
) -> dict:
    """
    Load waveform data for specific events.
    
    :param event_dict: Dictionary mapping event names to lists of picks. (e.g. self.events_with_picks)
    :param data_folder: Path to the folder containing waveform data files.
    :param format: Format of the waveform data files, one of ['ZNE', 'UVW', 'ZNE_calib_ACC', 'UVW_calib_ACC']. Default is 'ZNE'.
    :param fs: Target sampling frequency for resampling traces. Default is 20 Hz.
    :param window_size_in_seconds: Size of the window around the event time to load waveforms. Default is 1638.4 seconds (27 minutes).
    :return: Dictionary mapping event names to ObsPy Stream objects containing the loaded waveforms.
    """
    waveforms = {}

    for event_name, picks in tqdm(event_dict.items(), desc="Loading waveforms for events"):

        # get event time
        start_time = min([pick['time'] for pick in picks]) - num_windows_around_event * window_size_in_seconds
        end_time = max([pick['time'] for pick in picks]) + num_windows_around_event * window_size_in_seconds

        # get file names. can be two files if the event spans over two days
        file_name_start = utcdatetime_to_filename(start_time, components=format)
        file_name_end = utcdatetime_to_filename(end_time, components=format)

        # if the event spans over two days, we need to load both files
        if file_name_start != file_name_end:
            file_names = [file_name_start, file_name_end]
        else:
            file_names = [file_name_start]
        
        # load the waveform data for the event
        for file_name in file_names:
            file_path = os.path.join(data_folder, file_name)
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist. Skipping event {event_name}.")
                continue
            try:
                st = read(file_path, starttime=start_time, endtime=end_time)
                if len(st) == 0:
                    print(f"No data found for event {event_name} in file {file_path}.")
                    continue
                # resample to target sampling frequency
                for trace in st:
                    if trace.stats.sampling_rate != fs:
                        trace.resample(fs)
                st.merge(method=1, fill_value=0)  # Merge traces with the same station and channel
                
                # if event_name in waveforms, append the new traces
                if event_name in waveforms:
                    waveforms[event_name] += st
                else:
                    # otherwise, create a new entry
                    waveforms[event_name] = st
                
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
    
    return waveforms


def ensure_crop_length(cropped: Stream, window_size_samples: int) -> Stream:
    """
    Ensure that the cropped stream has exactly self.window_size_samples samples.
    If the cropped data is shorter than the window size, it will be padded with zeros at the end.
    If it is longer, the end will be trimmed to the window size.

    :param cropped: Stream object containing the cropped waveform data.
    :param window_size_samples: Desired length of the waveform data in samples.
    :return: Stream object with traces adjusted to the window size.
    """
    if len(cropped[0].data) != window_size_samples:
        #print(f"Warning: Cropped data length {len(cropped[0].data)} does not match window size {window_size_samples}.")
        diff = window_size_samples - len(cropped[0].data)
        if diff > 0:
            # Pad the data with zeros if it's shorter than the window size
            padding = np.zeros(diff)
            for trace in cropped:
                trace.data = np.concatenate((trace.data, padding))
        else:
            # Trim the data if it's longer than the window size
            for trace in cropped:
                trace.data = trace.data[:window_size_samples]
    return cropped


def normalize_trace(st: Stream, method: str ='std') -> Stream:
    """
    Remove the mean and normalize ('peak') or standardize ('std') the traces in a Stream object.
    Normalization is done per trace resp. per component.
    The data is modified in place.
    
    @param st: Stream object containing the traces to normalize.
    @param method: Normalization method, either 'std' for standard deviation or 'peak' for maximum value.
    @return: Normalized Stream object.
    """
    if method not in ['std', 'peak']:
        raise ValueError("Normalization method must be either 'std' or 'peak'")
    
    # Create a copy to avoid modifying the original Stream
    # Necessary because .slice() produces references to the original data
    #st = st.copy()  

    eps = 1e-8

    for trace in st:
        if method == 'std':
            #print(trace.stats)
            #print(f'Std: {np.std(trace.data)}, Mean: {np.mean(trace.data)}')
            std_trace = np.std(trace.data)
            std_trace = np.where(std_trace == 0, 1, std_trace)
            if std_trace < eps:
                print(f'Warning: small standard deviation {std_trace}. Starttime: {trace.stats.starttime}')
                # If the standard deviation is zero, just remove the mean
                trace.data = trace.data - np.mean(trace.data)
            else:
                trace.data = (trace.data - np.mean(trace.data)) / std_trace
        elif method == 'max':
            peak_val = np.max(np.abs(trace.data))
            if peak_val < eps:
                # If the peak value is zero, just remove the mean
                trace.data = trace.data - np.mean(trace.data)
            else:
            # Normalize by peak value
                trace.data = (trace.data - np.mean(trace.data)) / np.max(np.abs(trace.data))
    
    return st
    

def fill_missing_channels(st: Stream, components: str = 'ZNE', strategy: str = 'copy', window_size: int = 32768) -> Stream:
    """
    The stream should have three components. First, check if all three components are present.
    If not, fill the missing components with zeros or copy the existing component data.
    Traces should be merged by component. A correct stream should have exactly three traces, one for each component.
    Here, we handle missing channels, not duplicates.

    :param st: Stream object containing the traces to check.
    :param components: String with expected components, e.g., 'ZNE' for vertical, north, and east components.
    :param strategy: Strategy to fill missing channels, either 'copy' to copy existing data or 'zero' to fill with zeros.
    :param window_size: Size of the window in samples to create traces for missing components.
    :return: Stream object with missing channels filled.
    """

    if strategy not in ['copy', 'zero']:
        raise ValueError("Strategy must be either 'copy' or 'zero'")

    expected_components = [comp for comp in components]

    # Check if all three components are present
    existing_components = set(trace.stats.component for trace in st)

    missing_components = set(expected_components) - existing_components

    # check if existing components data is not all zeros
    for comp in existing_components:
        if np.all(st.select(component=comp)[0].data == 0):
            missing_components.add(comp)
            st.remove(st.select(component=comp)[0])  # Remove zero traces

    existing_components -= missing_components  # Remove missing components from existing
    
    #assert len(missing_components) <= 2, "More than two components are missing. This should not happen."
    if len(missing_components) == 3:
        st = Stream()
        print("All components are missing. Creating empty traces for all components.")
        for c in components:
            # create a dummy trace with zeros
            tr = obspy.Trace()
            tr.stats.sampling_rate = 20
            tr.stats.component = c
            tr.data = np.zeros(window_size, dtype=np.float32)
            st += tr
        return st
    
    # Fill missing components
    for comp in missing_components:
        if strategy == 'copy':
            # select random existing component to copy
            existing_comp = np.random.choice(list(existing_components))
            new_trace = st.select(component=existing_comp)[0].copy()
            new_trace.stats.component = comp
            st += new_trace
        elif strategy == 'zero':
            # Create a new trace with zeros for the missing component
            new_trace = st.select(component=expected_components[0])[0].copy() # copy for stats
            new_trace.data = np.zeros_like(new_trace.data) # set to zeros
            new_trace.stats.component = comp # adjust component
            st += new_trace
    
    return st



if __name__ == "__main__":
    daily_data = load_waveform_data_by_day(
        data_folder='dataloaders/data/insight',
        format='ZNE',
        fs=20
    )
    print(f"Loaded {len(daily_data)} days of waveform data.")

    # print the first 5 days
    for day, st in list(daily_data.items())[:5]:
        print(f"Day: {day}, Number of traces: {len(st)}")
        print(f"First trace: {st[0].stats}")
        print('\n\n')