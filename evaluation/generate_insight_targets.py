from pandas import DataFrame
from pathlib import Path
from obspy import UTCDateTime
import argparse
from dataloaders.insight_loader import InsightDataset
import pandas as pd
from tqdm import tqdm
from obspy.core import Stream, Trace
import numpy as np


def get_datasets() -> dict[str, InsightDataset]:
    """
    Load the Insight datasets for train, dev, and test splits.
    Returns:
        dict: A dictionary containing the datasets for 'train', 'dev', and 'test'.
    """
    print('Loading Insight datasets...\n\n\n')
    
    datasets = {}
    for split in ['train', 'dev', 'test']:
        dataset = InsightDataset(
            data_dir='dataloaders/data/insight',
            fs=20,
            train=split,
            return_event_name=True,
            load_only_event_data=True,
            use_daily_datastructure=False,
            load_inventory=False,
            pre_slice=False,
            preload_waveform_data=False,
            merge_when_loading=False,
        )
        datasets[split] = dataset
    return datasets


def get_noise_window(event_name, catalogue, before_or_after, buffer=15, window_size=30.0) -> tuple[UTCDateTime, UTCDateTime]:
    """
    Get a noise window before or after an event based on the pick information 'start' or 'end' in the catalogue.

    @param event_name: Name of the event to get the noise window for.
    @param catalogue: The catalogue containing event information.
    @param before_or_after: Specify whether to get the window 'before' or 'after' the event.
    @param buffer: Buffer time in seconds in between the noise window and the event.
    e.g. if the window is before the event, there will be a buffer of 'buffer' seconds between the end of the noise window and the start of the event.
    @param window_size: Size of the noise window in seconds.
    @return: A tuple containing the start and end times of the noise window.
    """
    #before_or_after = np.random.choice(['before', 'after'])
    if before_or_after == 'before':
        end = [e for e in catalogue[catalogue['name'] == event_name]['picks'].values[0] if e['phase_hint'] == 'start'][0]['time']
        end = end - buffer 
        start = end - window_size
    else:
        start = [e for e in catalogue[catalogue['name'] == event_name]['picks'].values[0] if e['phase_hint'] == 'end'][0]['time']
        start = start + buffer 
        end = start + window_size

    return start, end


def generate_file_task1(datasets, target_directory, window_size=30.0, buffer=15.0):
    """
    Generate a CSV file with targets for event detection in the Insight dataset.
    The file will contain windows around P and S picks, as well as noise windows before and after the events.
    The windows are defined by the specified window size.
    The P and S picks are centered in the windows.

    @param datasets: Dictionary containing the Insight datasets for 'train', 'dev', and 'test'.
    @param target_directory: Directory where the generated CSV file will be saved.
    @param window_size: Size of the local windows in seconds
    @param buffer: Buffer time in seconds in between the noise window and the event.
    """
    save_dir = Path(target_directory)

    df = DataFrame(columns=[
        'event_name', # name of the event
        'trace_idx', # index from 0 to n-1
        'trace_split', # train, val or test
        'sampling_rate', # 20
        'start_time', # start sample of the window
        'end_time', # end sample of the window
        'trace_type', # marsquake or noise
    ])

    catalogue = datasets['train'].catalogue

    for split, dataset in datasets.items():
        print(f'Processing {split} set...')
        events_with_picks = dataset.events_with_picks

        for i, (event_name, picks) in enumerate(events_with_picks.items()):
            # add P and S picks
            for pick in picks:
                start = pick['time'] - window_size / 2
                end = pick['time'] + window_size / 2

                df.loc[len(df)] = {
                    'event_name': event_name,
                    'trace_idx': i,
                    'trace_split': split,
                    'sampling_rate': dataset.fs,
                    'start_time': start,
                    'end_time': end,
                    'trace_type': 'marsquake',
                }
            # add noise windows
            for before_or_after in ['before', 'after']:
                start, end = get_noise_window(
                    event_name=event_name, 
                    catalogue=catalogue,
                    before_or_after=before_or_after,
                    window_size=window_size,
                    buffer=buffer,
                    )
                
                df.loc[len(df)] = {
                    'event_name': event_name,
                    'trace_idx': i,
                    'trace_split': split,
                    'sampling_rate': dataset.fs,
                    'start_time': start,
                    'end_time': end,
                    'trace_type': 'noise',
                }

    # save the dataframe
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / 'task1.csv', index=False)
    print(f'Saved task1.csv to {save_dir / "task1.csv"}')


def generate_file_task23(datasets, target_directory, window_size=10.0):
    """
    Generate a CSV file with targets for phase discrimination and onset regression in the Insight dataset.
    The file will contain windows around P and S picks, with the specified window size.
    The P and S picks are centered in the windows, and the phase onset is recorded.

    @param datasets: Dictionary containing the Insight datasets for 'train', 'dev', and 'test'.
    @param target_directory: Directory where the generated CSV file will be saved.
    @param window_size: Size of the local windows in seconds
    """

    df = DataFrame(columns=[
        'event_name',  # name of the event
        'trace_idx',  # index from 0 to n-1
        'trace_split',  # train, val or test
        'sampling_rate',  # 20
        'start_time',  # start time of the window
        'end_time',  # end time of the window
        'phase_label', # P or S
        'full_phase_label', # full phase label (e.g. Pn, Pg, Sn, Sg)
        'phase_onset' # time of the phase onset
    ])

    # loop through the sets:
    for split, dataset in datasets.items():
        print(f'Processing {split} set...')
        events_with_picks = dataset.events_with_picks

        # loop through the events
        for i, (event_name, picks) in enumerate(events_with_picks.items()):
            
            # add P and S picks
            for pick in picks:
                
                start = pick['time'] - window_size / 2
                end = pick['time'] + window_size / 2

                df.loc[len(df)] = {
                    'event_name': event_name,
                    'trace_idx': i,
                    'trace_split': split,
                    'sampling_rate': dataset.fs,
                    'start_time': start,
                    'end_time': end,
                    'phase_label': pick['pick_type'],  # P or S
                    'full_phase_label': pick['phase_hint'],
                    'phase_onset': pick['time'],
                }

    # save the dataframe
    save_dir = Path(target_directory)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / 'task23.csv', index=False)
    print(f'Saved task23.csv to {save_dir / "task23.csv"}')


def generate_file_task4(target_directory, window_size_samples=32768, fs=20):
    # load deep catalogue
    deep_catalogue = pd.read_csv('dataloaders/data/insight/MQNet_DeepCatalogue.csv')
    # laod dataset with daily data structure
    dataset = InsightDataset(
        data_dir='dataloaders/data/insight',
        fs=fs,
        train='train',
        return_event_name=True,
        load_only_event_data=False,
        use_daily_datastructure=True, # load all data 
        load_inventory=False,
        pre_slice=False,
        preload_waveform_data=False,
        merge_when_loading=False,
    )

    v14_catalogue = dataset.catalogue

    results = DataFrame(columns=[
        'trace_idx',  # index from 0 to n-1
        'sampling_rate',  # 20
        'start_time',  # start time of the window
        'end_time',  # end time of the window
        'events', # name of events in the window
        'MQNet_bin', # true if the window contains an event in the deep catalogue, false otherwise
        'v14_bin', # true if the window contains an event in the v14 catalogue, false otherwise
        'contains_event', # true if the window contains an event in either catalogue, false otherwise
    ])

    fs = float(fs)

    global_start_time = UTCDateTime(2019, 6, 1, 1, 0, 0) # start time of the Insight dataset
    global_end_time = UTCDateTime(2022, 5, 26, 23, 0, 0) # end time of the Insight dataset
    inc_in_seconds = window_size_samples / fs / 2.0 # increment in seconds, half the window size

    print(f'\nGlobal start time: {global_start_time}')
    print(f'Global end time: {global_end_time}\n')

    # Estimate number of windows for tqdm
    total_seconds = (global_end_time - global_start_time)
    num_windows = int(total_seconds / inc_in_seconds)

    current_window_start = global_start_time

    # loop through the dataset in windows
    for _ in tqdm(range(num_windows), desc="Generating windows for task 4"):
        current_window_end = current_window_start + 2 * inc_in_seconds - 2.0 / fs

        # check if waveform data is available for the current window
        start_day = current_window_start.strftime('%Y-%m-%d')
        end_day = current_window_end.strftime('%Y-%m-%d')
        if not (start_day in dataset.daily_data or end_day in dataset.daily_data):
            # if no data is available, step the window forward
            current_window_start += inc_in_seconds
            continue
        else:
            st = Stream()
            if start_day == end_day or end_day not in dataset.daily_data:
                st += dataset.daily_data[start_day]
            elif start_day not in dataset.daily_data:
                st += dataset.daily_data[end_day]
            else:
                st += dataset.daily_data[start_day]
                st += dataset.daily_data[end_day]
        
            st = st.slice(
                starttime=current_window_start,
                endtime=current_window_end
            ).merge(method=1, fill_value=0)

            # check if the stream is empty
            if len(st) == 0:
                # if no data is available, step the window forward
                #print(f'Warning: No data available for the window from {current_window_start} to {current_window_end}. Skipping this window.')
                current_window_start += inc_in_seconds
                continue
            #if len(st) != 3:
                #print(f'Warning: Expected 3 traces, but got {len(st)} traces in the window from {current_window_start} to {current_window_end}. Skipping this window.')
            if all(np.all(trace.data == 0) for trace in st):
                # if all traces are empty, step the window forward
                current_window_start += inc_in_seconds
                continue


        # check if there are v14 events in the window
        mask = (v14_catalogue['end_time'] >= current_window_start) & (v14_catalogue['start_time'] <= current_window_end)
        v14_events = v14_catalogue[mask]
        v14_events = v14_events['name'].tolist()
        contains_v14_event = len(v14_events) > 0

        # check if there are deep catalogue events in the window
        mask = (deep_catalogue['utc_end'] >= current_window_start) & (deep_catalogue['utc_start'] <= current_window_end)
        deep_events = deep_catalogue[mask]
        deep_events = deep_events['event_name'].tolist()
        contains_deep_event = len(deep_events) > 0

        # add window to the results dataframe
        results.loc[len(results)] = {
            'trace_idx': len(results),
            'sampling_rate': fs,
            'start_time': current_window_start,
            'end_time': current_window_end,
            'events': v14_events + deep_events,  # names of events in the window
            'MQNet_bin': contains_deep_event,  # true if the window contains an event in the deep catalogue, false otherwise
            'v14_bin': contains_v14_event,  # true if the window contains an event in the v14 catalogue, false otherwise
            'contains_event': contains_deep_event or contains_v14_event,  # true if the window contains an event in either catalogue, false otherwise
        }
        # save every 1000th window to the disk
        if len(results) % 1000 == 0:
            # save the results dataframe
            save_dir = Path(target_directory)
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            results.to_csv(save_dir / 'task4.csv', index=False)
            #print(f'Saved task4.csv to {save_dir / "task4.csv"}')

        # step the window forward
        current_window_start += inc_in_seconds
        if current_window_start >= global_end_time:
            break

    print(f'\n\nGenerated {len(results)} windows for task 4.')
    # save the results dataframe
    save_dir = Path(target_directory)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(save_dir / 'task4.csv', index=False)
    print(f'Saved task4.csv to {save_dir / "task4.csv"}')



if __name__ == '__main__':
    """
    Main function to generate the Insight dataset target files for tasks 1 and 2/3.
    Argument parser takes the tasks (123 or 4), the target directory, the window size for task 1, 
    and the window size for task 2/3, the buffer time for task 1 and the window size for task4.
    The window sizes for tasks 123 are in seconds, the window size for task 4 is in samples.
    Usage:
        python -m evaluation.generate_insight_targets --tasks task123 task4 --target_directory dataloaders/data/insight --window_size_task1 30.0 --window_size_task23 10.0 --buffer 15.0 --window_size_task4 32768

    Only generate targets for task4:
        python -m evaluation.generate_insight_targets --tasks task4 --target_directory dataloaders/data/insight --window_size_task4 32768
    Only generate targets for task123:
        python -m evaluation.generate_insight_targets --tasks task123 --target_directory dataloaders/data/insight --window_size_task1 30.0 --window_size_task23 10.0 --buffer 15.0
    """
    parser = argparse.ArgumentParser(description='Generate Insight dataset target files for tasks 1 and 2/3.')
    parser.add_argument('--tasks', type=str, nargs='+', default=['task123'], help='Tasks to generate targets for. Options: task123, task4.')
    parser.add_argument('--target_directory', type=str, default='dataloaders/data/insight', help='Directory to save the generated target files.')
    parser.add_argument('--window_size_task1', type=float, default=30.0, help='Window size for task 1 in seconds.')
    parser.add_argument('--window_size_task23', type=float, default=10.0, help='Window size for task 2/3 in seconds.')
    parser.add_argument('--buffer', type=float, default=15.0, help='Buffer time in seconds for task 1 noise windows.')
    parser.add_argument('--window_size_task4', type=int, default=32768, help='Window size for task 4 in samples (default: 32768).')
    args = parser.parse_args()

    # check if files aleady exist
    target_dir = Path(args.target_directory)
    if 'task123' in args.tasks:
        if target_dir.exists() and ((target_dir / 'task1.csv').exists() or (target_dir / 'task23.csv').exists()):
            # ask user if they want to overwrite the files
            overwrite = input(f"Target files for tasks 1, 2 or 3 already exist in {target_dir}. Do you want to overwrite them? (y/n): ")
            if overwrite.lower() != 'y':
                print("Exiting without generating new files.")
                exit()
    if 'task4' in args.tasks:
        if target_dir.exists() and (target_dir / 'task4.csv').exists():
            # ask user if they want to overwrite the file
            overwrite = input(f"Target file for task 4 already exists in {target_dir}. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("Exiting without generating new files.")
                exit()
    
    if 'task123' in args.tasks:
        datasets = get_datasets()
    
        generate_file_task1(
            datasets=datasets,
            target_directory=args.target_directory,
            window_size=args.window_size_task1,
            buffer=args.buffer
            )

        generate_file_task23(
            datasets=datasets, 
            target_directory=args.target_directory,
            window_size=args.window_size_task23
            )
        # write arguments and current date to text file
        with open(target_dir / f'generation_info_task123.txt', 'w') as f:
            f.write(f"Generated on: {UTCDateTime.now()}\n")
            f.write(f"Target directory: {args.target_directory}\n")
            f.write(f"Window size for task 1: {args.window_size_task1} seconds\n")
            f.write(f"Window size for task 2/3: {args.window_size_task23} seconds\n")
            f.write(f"Buffer time for task 1: {args.buffer} seconds\n")

    if 'task4' in args.tasks:
        # generate file for task 4
        generate_file_task4(
            target_directory=args.target_directory,
            window_size_samples=args.window_size_task4,  # default value for task 4
            fs=20  # default sampling rate for Insight dataset
            )
        # write arguments and current date to text file
        with open(target_dir / f'generation_info_task4.txt', 'w') as f:
            f.write(f"Generated on: {UTCDateTime.now()}\n")
            f.write(f"Target directory: {args.target_directory}\n")
            f.write(f"Window size for task 4: {args.window_size_task4} samples\n")
            f.write(f"Sampling rate: {20} Hz\n")