import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from obspy import Stream
from plotting_utils.plotting_utils import *


def plot_example(
    cropped_st: torch.Tensor, 
    label_sequences: torch.Tensor, 
    clip: float = None, 
    title: str='',
    plot_over_time: bool = False,
    fs: int = 20
    ) -> None:
    """
    Plot the three-component waveform and the label sequences.

    @param cropped_st: The cropped waveform data as a torch tensor of shape (n_samples, 3).
    @param label_sequences: The label sequences as a torch tensor of shape (n_samples, 3).
    @param clip: Optional clipping value for the y-axis.
    @param title: Optional title for the plot.
    @param plot_over_time: If True, the x-axis will be in seconds; otherwise, it will be in sample indices.
    @param fs: Sampling frequency of the waveform data, used if plot_over_time is True.
    """

    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5 # Major x-axis ticks
    mpl.rcParams['ytick.major.width'] = 0.5 

    plt.rcParams.update(
        rc_params_update
    )
    
    n_samples = cropped_st.shape[0]
    sampling_rate = float(fs) if plot_over_time else 1.0  # Use fs if plotting over time, otherwise use 1.0 for sample index
    time = np.arange(n_samples) / sampling_rate

    fig = plt.figure(figsize=(A4_WIDTH, A4_WIDTH/4))
    ax = fig.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1]})

    # plot waveform data
    for i, c in enumerate(['Z', 'N', 'E']):
        ax[0].plot(cropped_st[:, i], label=c, alpha=0.7)
    

    ax[0].legend()
    
    # plot label sequences
    if clip is not None:
        ax[0].set_ylim(-clip, clip)

    if plot_over_time:
        ax[1].set_xlabel('Time (s)')
    else:
        ax[1].set_xlabel('Sample Index')
    
    ax[1].plot(label_sequences[:, 0], label='P', color='blue', alpha=0.7)
    ax[1].plot(label_sequences[:, 1], label='S', color='red', alpha=0.7)
    ax[1].plot(label_sequences[:, 2], label='Noise', color='green', alpha=0.7)
    ax[1].set_xlabel('Sample Index')
    ax[1].legend()
    ax[0].set_title(title)
    
    plt.show()


def plot_example_with_predictions(
    cropped_st: torch.Tensor, 
    label_sequences: torch.Tensor, 
    predicted_sequences: torch.Tensor, 
    clip: float = None, 
    title: str = '',
    plot_over_time: bool = False,
    fs: int = 20
) -> None:
    """
    Plot the three-component waveform, the expected label sequences, and the predicted label sequences.

    @param cropped_st: The cropped waveform data as a torch tensor of shape (n_samples, 3).
    @param label_sequences: The expected label sequences as a torch tensor of shape (n_samples, 3).
    @param predicted_sequences: The predicted label sequences as a torch tensor of shape (n_samples, 3).
    @param clip: Optional clipping value for the y-axis.
    @param title: Optional title for the plot.
    @param plot_over_time: If True, the x-axis will be in seconds; otherwise, it will be in sample indices.
    @param fs: Sampling frequency of the waveform data, used if plot_over_time is True.
    """
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5 # Major x-axis ticks
    mpl.rcParams['ytick.major.width'] = 0.5 

    plt.rcParams.update(
        rc_params_update
    )

    n_samples = cropped_st.shape[0]
    sampling_rate = float(fs) if plot_over_time else 1.0  # Use fs if plotting over time, otherwise use 1.0 for sample index
    time = np.arange(n_samples) / sampling_rate
    
    fig = plt.figure(figsize=(A4_WIDTH, A4_WIDTH/3))
    ax = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]})

    # plot waveform data
    for i, c in enumerate(['Z', 'N', 'E']):
        ax[0].plot(time, cropped_st[:, i], label=c, alpha=0.7)
    
    ax[0].legend()
    
    # plot label sequences
    if clip is not None:
        ax[0].set_ylim(-clip, clip)
    
    ax[1].plot(time, label_sequences[:, 0], label='P', color='blue', alpha=0.7)
    ax[1].plot(time, label_sequences[:, 1], label='S', color='red', alpha=0.7)
    ax[1].plot(time, label_sequences[:, 2], label='Noise', color='green', alpha=0.7)
    
    # plot predicted sequences
    ax[2].plot(time, predicted_sequences[:, 0], label='Predicted P', color='blue', alpha=0.7)
    ax[2].plot(time, predicted_sequences[:, 1], label='Predicted S', color='red', alpha=0.7)
    ax[2].plot(time, predicted_sequences[:, 2], label='Predicted Noise', color='green', alpha=0.7)

    if plot_over_time:
        ax[2].set_xlabel('Time (s)')
    else:
        ax[2].set_xlabel('Sample Index')
    ax[1].legend()
    ax[2].legend()
    
    ax[0].set_title(title)
    plt.tight_layout()
    
    plt.show()

