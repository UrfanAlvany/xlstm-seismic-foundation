import seaborn as sns
import matplotlib.pyplot as plt

palette = sns.color_palette("tab20b")
colors = {
    # baseline models
    'phasenet': palette[0],
    'pn': palette[0],
    'eqtransformer': palette[2],
    'eqt': palette[2],
    'seislm-base': palette[1],
    'seislm-large': palette[3],
    'convnet': palette[0],

    # small models
    's4d': palette[4],
    'mamba': palette[5],
    'hydra': palette[6],

    # large models
    'hydra-large': palette[12],
    'mamba-fine': palette[13],
    'mamba-2-fine': palette[14],
    'mamba-fine-512': palette[-1],
    'mamba-baar': palette[16],
    'hydra-downsample': palette[17],
    'hydra-fine': palette[17],
    'hydra-rand-init': palette[17],
    'mamba-rand-init': palette[13],
    # '': '',
}

linestyles = {
    # baseline models
    'phasenet': 'dotted',
    'eqtransformer': 'dotted',
    'seislm-base': 'dotted',
    'seislm-large': 'dotted',

    # small models
    's4d': 'solid',
    'mamba': 'solid',
    'hydra': 'solid',

    # large models
    'hydra-large': 'solid',
    'mamba-fine': 'solid',
    'mamba-2-fine': 'solid',
    'mamba-fine-512': 'solid',
    'mamba-baar': 'solid',
    'hydra-downsample': 'solid',
    'hydra-fine': 'solid',
    'hydra-rand-init': 'dotted',
    'mamba-rand-init': 'dotted',
}

markers = {
    # baseline models
    'phasenet': 'v',
    'eqtransformer': 'D',
    'seislm-base': 'p',
    'seislm-large': 's',

    # small models
    's4d': '<',
    'mamba': 'd',
    'hydra': 'o',

    # large models
    'hydra-large': '^',
    'mamba-fine': 'D',
    'mamba-2-fine': 's',
    'mamba-fine-512': 'd',
    'mamba-baar': 'o',
    'hydra-downsample': 'D',
    'hydra-fine': 'h',
    'hydra-rand-init': '^',
    'mamba-rand-init': 's',
}

formatted_names = {
    'phasenet': 'PhaseNet',
    'pn': 'PhaseNet',
    'eqtransformer': 'EQTransformer',
    'eqt': 'EQTransformer',
    'seislm-base': 'SeisLM-base',
    'seislm-large': 'SeisLM-large',
    's4d': 'S4D',
    'mamba': 'MAMBA',
    'hydra': 'HYDRA',
    'hydra-large': 'HYDRA-large',
    'mamba-fine': 'MAMBA-fine',
    'mamba-2-fine': 'MAMBA-2-fine',
    'mamba-baar': 'MAMBA-baar',
    'hydra-downsample': 'HYDRA-downsample',
    'mamba-fine-512': 'MAMBA-fine-512',
    'hydra-fine': 'HYDRA-fine',
    'hydra-rand-init': 'HYDRA-rand-init'
}

short_names = {
    'phasenet': 'PN',
    'pn': 'PN',
    'eqtransformer': 'EQT',
    'eqt': 'EQT',
    'seislm-base': 'SeisLM-b',
    'seislm-large': 'SeisLM-l',
    's4d': 'S4D',
    'mamba': 'MAMBA',
    'hydra': 'HYDRA',
    'hydra-large': 'HYDRA-l',
    'mamba-fine': 'MAMBA-f',
    'mamba-2-fine': 'MAMBA-2-f',
    'mamba-baar': 'MAMBA-baar',
    'hydra-downsample': 'HYDRA-d',
}

FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
TICK_FONT_SIZE = 10
A4_WIDTH = 8.27
A4_HEIGHT = 11.69

rc_params_update = {
    "font.size": LEGEND_FONT_SIZE,  # General font size
    "axes.titlesize": FONT_SIZE,  # Subplot titles
    "figure.titlesize": FONT_SIZE,  # Suptitle (main title)
    "legend.fontsize": LEGEND_FONT_SIZE,  # Legend font size
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    "pdf.fonttype": 42,  # Embed fonts in PDF for better compatibility
}

_name_aliases = {
    'pn': 'phasenet',
    'eqt': 'eqtransformer',
    'hydra-unet': 'hydra-large',
    'mamba-sash': 'mamba-fine',
    'hydra-down': 'hydra-downsample',
}


def _normalize_name(name: str) -> str:
    name = name.lower()
    name = name.replace("_", "-")
    name = name.replace(" ", "-")
    if name in _name_aliases.keys():
        name = _name_aliases[name]
    return name


def get_color(name: str):
    """
    Returns the color associated with the given model.
    :param name: model name (is converted to lower case and _ are replaced with -)
    :return: rgb color
    """
    return colors[_normalize_name(name)]


def get_marker(name: str):
    """
    Returns the marker associated with the given model.
    :param name: model name (is converted to lower case and _ are replaced with -)
    :return:
    """
    return markers[_normalize_name(name)]


def get_linestyle(name: str):
    """
    returns the linestyle of the given model. Baseline models are 'dotted', our own models are 'solid'.
    :param name: model name (is converted to lower case and _ are replaced with -)
    :return:
    """
    return linestyles[_normalize_name(name)]


def format_name(name: str):
    """
    Formats the name to standard form.
    :param name: model name
    :return: formatted name
    """
    return formatted_names[_normalize_name(name)]


def format_name_short(name: str) -> str:
    """
    Returns the short name for the given model. For example PhaseNet is PN.
    :param name: model name
    :return: short version of name
    """
    return short_names[_normalize_name(name)]


def create_title(dataset, task, phase=None, metric=None):
    """
    Formats and arranges a title for the plot. The format is: <phase>-<task> <metric> - <dataset>
    :param dataset: Name of the dataset. Is capitalized.
    :param task: Name of task. We remove underscores and replace with spaces. Capitalize first letters.
    :param phase: 'P' or 'S'. We capitalize.
    :param metric: Metric in the plot. We capitalize it.
    :return: formatted title
    """
    dataset = dataset.upper()
    if metric is not None:
        metric = metric.upper()
    if phase is not None:
        phase = phase.upper()
    task = task.replace('_', ' ')
    task = task.title()
    if phase is not None:
        title = f'{phase}-{task} {metric} - {dataset}' if metric is not None else f' {phase}-{task} - {dataset}'
    else:
        title = f'{task} {metric} - {dataset}' if metric is not None else f'{task} - {dataset}'
    return title


def get_figsize_wide_2():
    return A4_WIDTH, A4_WIDTH / 2


def get_figsize_wide_3():
    return A4_WIDTH, A4_WIDTH / 3


def get_figsize_square_1():
    return A4_WIDTH, A4_WIDTH


def get_figsize_square_2():
    return A4_WIDTH / 2, A4_WIDTH / 2


def get_figsize_square_3():
    return A4_WIDTH / 3, A4_WIDTH / 3


def get_figsize_square_4():
    return A4_WIDTH / 4, A4_WIDTH / 4
