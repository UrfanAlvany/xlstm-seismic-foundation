import re
import os
import glob
import numpy as np
import torch
import copy

CHANNELS = ['HHZ', 'HHN', 'HHE']
STATIONS = ['COVE', 'HDC3', 'RIFO']


def get_metadata(f_name: str) -> dict:
    full_path = copy.deepcopy(f_name)
    f_name = f_name.split('/')[-1].replace('..', '.')

    meta = f_name.split('.')
    station = meta[1]
    channel = meta[2]
    year = meta[3][:4]
    day = meta[3][4:7]
    meta = {
        'station': station,
        'channel': channel,
        'year': year,
        'day': day,
        'path': full_path
    }
    return meta


def filter_channel(file_list: list, channels: str | list) -> list:
    if isinstance(file_list[0], str):
        file_list = [get_metadata(f) for f in file_list]

    if isinstance(channels, str):
        channels = [channels]

    for c in channels:
        assert c in CHANNELS

    filtered = [f for f in file_list if f['channel'] in channels]
    return filtered


def filter_station(file_list: list, stations: str | list) -> list:
    if isinstance(file_list[0], str):
        file_list = [get_metadata(f) for f in file_list]

    if isinstance(stations, str):
        stations = [stations]

    for s in stations:
        assert s in STATIONS

    filtered = [f for f in file_list if f['station'] in stations]
    return filtered


def filter_year(file_list: list, years: str | int | list) -> list:
    if isinstance(file_list[0], str):
        file_list = [get_metadata(f) for f in file_list]
    if isinstance(years, str):
        years = [years]
    if isinstance(years, int):
        years = [str(years)]
    years = [str(year) for year in years]

    filtered = [f for f in file_list if f['year'] in years]
    return filtered


def filter_metadata(
        file_list: list,
        years: str | int | list = None,
        channels: str = None,
        stations: str = None,
) -> list:
    if years is not None:
        filtered = filter_year(file_list, years)
    else:
        filtered = file_list
    if channels is not None:
        filtered = filter_channel(filtered, channels)
    if stations is not None:
        filtered = filter_station(filtered, stations)

    return filtered


def sort_metadata_by_date(file_list: list) -> list:
    assert isinstance(file_list[0], dict)

    sorted_file_list = sorted(file_list, key=lambda d: d['year'] + d['day'])
    return sorted_file_list


def extract_sequences_from_metadata_list(file_list: list) -> list:
    """
    Extract sequences of consecutive days from metadata. file_list should already be metadata (not strings
    with file paths). To convert to metadata use [get_metadata(f) for f in file_list]. Does not work with multiple
    channels. The metadata should be filtered to only contain measurements from a single channel. To filter
    channels use filter_channel(file_list, channels=channel)
    :param file_list: list of metadata
    :return: nested list of sequences. Each sequence contains consecutive days from the same station
    """
    sorted_data = sort_metadata_by_date(file_list)

    sequences = []
    for entry in sorted_data:
        year_day = int(entry['year'] + entry['day'])
        if len(sequences) == 0:
            sequences.append([entry])
        else:
            add_to_existing_seq = False
            for seq in sequences:
                if int(seq[-1]['year'] + seq[-1]['day']) == year_day - 1 and seq[-1]['station'] == entry['station']:
                    seq.append(entry)
                    add_to_existing_seq = True
                    break
            if not add_to_existing_seq:
                sequences.append([entry])

    total_entries = sum([len(seq) for seq in sequences])
    assert total_entries == len(file_list)
    return sequences


def format_label(s):
    """
    Convert strings: 'processed_i4.HDC3.HHZ.2022287_0+.pt'->'HDC3.HHZ.2022287'.
    """
    # Remove leading parts before the first dot (and any extra dots)
    after_first_dot = s.split('.', 1)[1]
    # Remove the trailing parts after the last dot
    before_last_dot = re.split(r'\.[^\.]+$', after_first_dot)[0]
    # Replace multiple dots with a single dot, if needed
    single_dots = re.sub(r'\.+', '.', before_last_dot)
    return single_dots.split('_')[0]


def find_data_min_and_max(filename: str) -> tuple:
    file_list = glob.glob(os.path.join(filename, '*.pt'))

    d_min = torch.inf
    d_max = - torch.inf
    for f in file_list:
        data = torch.load(f)
        temp_min = torch.min(data)
        temp_max = torch.max(data)
        if temp_max > d_max:
            d_max = temp_max
        if temp_min < d_min:
            d_min = temp_min
    return d_min, d_max


if __name__ == '__main__':
    path = '../data/costa_rica/small_subset'
    #d_min, d_max = find_data_min_and_max(path)
    #print(f'd_min: {d_min}, d_max: {d_max}')
    data_min, data_max = find_data_min_and_max(path)
    print(max(abs(data_max), abs(data_min)))

