import torch
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import seisbench.data as sbd
import seisbench.generate as sbg
import pandas as pd

import evaluation.pick_eval as pe
from plotting_utils.plotting_utils import *


def get_model_dict(name, path, dataset, sample_len=None):
    if sample_len is None:
        res_path = path + f'/evals/eval_{dataset}/'
    else:
        res_path = path + f'/evals/sample_len_{sample_len}_{dataset}/'
    model_dict = {
        'name': name,
        'ckpt_path': path,
    }
    for task in ['1', '23']:
        for eval_set in ['train', 'dev', 'test']:
            file_name = f'{eval_set}_task{task}'
            file_path = res_path + file_name + '.csv'
            try:
                content = pd.read_csv(file_path)
            except:
                content = None
            model_dict.update({file_name: content})

            if content is not None:
                if task == '1':
                    res_event_det = pe.get_results_event_detection(file_path)
                    model_dict.update({f'{eval_set}_event_detection': res_event_det})
                if task == '23':
                    res_phase_ident = pe.get_results_phase_identification(file_path)
                    res_onset = pe.get_results_onset_determination(file_path)
                    model_dict.update({f'{eval_set}_phase_identification': res_phase_ident})
                    model_dict.update({f'{eval_set}_onset_determination': res_onset})
    return model_dict


def plot_ROC_single(model_dicts, dataset_name, task, colors, skip_train=True, skip_dev=True, figsize=(5, 5), xlim=[0, 0.3], ylim=[0.7, 1.0]):
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5  # Major x-axis ticks
    mpl.rcParams['ytick.major.width'] = 0.5

    if skip_train and skip_dev:
        eval_sets = ['test']
    elif skip_train:
        eval_sets = ['dev', 'test']
    else:
        eval_sets = ['train', 'dev', 'test']

    for i, eval_set in enumerate(eval_sets):
        fig, ax = plt.subplots(figsize=get_figsize_square_2())
        plt.rcParams.update(
            rc_params_update
        )
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style='ticks', rc=custom_params)
        plt.grid(False)
        title = create_title(dataset_name, task,)
        plt.suptitle(title)
        for model_dict in model_dicts:
            model_name = model_dict['name']
            try:
                task_results = model_dict[f'{eval_set}_{task}']
            except:
                task_results = None
            if task_results is not None:
                auc = task_results['auc']
                best_f1 = task_results['best_f1']
                label = '{:5s} {:1.3f}'.format(model_name, auc)
                f1_xy = task_results['best_f1_xy']
                # print(label)
                ax.plot(task_results['fpr'],
                        task_results['tpr'],
                        linewidth=3,
                        color=get_color(model_dict['name']),
                        #marker=get_marker(model_dict['name']),
                        )  #  , Best F1: {best_f1:.3f}
                ax.plot(
                    f1_xy[0],
                    f1_xy[1],
                    label=label,
                    color=get_color(model_dict['name']),
                    marker=get_marker(model_dict['name']),
                    markersize=10,
                )
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
        ax.legend(loc="lower right")
        #ax.set_title(eval_set.title())
    plt.tight_layout()
    plt.show()


def plot_p_dist_hist_single(model_dicts, h=2.5, res=0.45, bins=None):
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5  # Major x-axis ticks
    mpl.rcParams['ytick.major.width'] = 0.5
    phases = ['P', 'S']
    for model_dict in model_dicts:
        if np.isnan(model_dict['test_onset_determination']['P_onset_diff']).any():
            continue
        for i, phase in enumerate(phases):
            if phase == 'P':
                w = h + 0.5
            else:
                w = h + 1.2
            fig, ax = plt.subplots(figsize=(w, h))
            distances = model_dict['test_onset_determination']
            p_dist = distances[f'{phase}_onset_diff']
            # print(len(p_dist))
            p_dist_mask = np.abs(p_dist) > res
            high_res = np.sum(p_dist_mask) / len(p_dist)

            mae = np.mean(np.abs(p_dist))
            rmse = np.sqrt(np.mean(np.square(p_dist)))

            mean_dist = np.mean(p_dist)
            median_dist = np.median(p_dist)
            ax.axvline(mean_dist, color='orange', linestyle='dashed', linewidth=1, label='Mean Residual')
            ax.axvline(median_dist, color='r', linestyle='dashed', linewidth=1, label='Median Residual')

            if bins is None:
                bins = np.linspace(-0.4, 0.4, 80)
            ax.hist(p_dist, bins=bins)
            # ax.set_xlim([-2, 2])
            # ax.set_title(f'{phase} arrivals, RES: {high_res :.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}')
            ax.set_title(f'{model_dict["name"]}, {phase}')

            textstr = '\n'.join((
                f'{high_res:.2f}',
                f'{mae:.2f}',
                f'{rmse:.2f}',
            ))

            # Place the text box in the upper right corner of the plot
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.legend(loc='upper right')
            # plt.suptitle(model_dict['name'])
            # plt.tight_layout()
            plt.show()


def plot_p_dist_hist(model_dicts, h=2.5, res=0.45, bins=None):
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.major.width'] = 0.5  # Major x-axis ticks
    mpl.rcParams['ytick.major.width'] = 0.5
    phases = ['P', 'S']
    w = len(phases) * h + 1
    for model_dict in model_dicts:
        if np.isnan(model_dict['test_onset_determination']['P_onset_diff']).any():
            continue
        fig, axes = plt.subplots(1, 2, figsize=(w, h))
        for i, phase in enumerate(phases):
            ax = axes[i]
            distances = model_dict['test_onset_determination']
            p_dist = distances[f'{phase}_onset_diff']
            # print(len(p_dist))
            p_dist_mask = np.abs(p_dist) > res
            high_res = np.sum(p_dist_mask) / len(p_dist)

            mae = np.mean(np.abs(p_dist))
            rmse = np.sqrt(np.mean(np.square(p_dist)))

            mean_dist = np.mean(p_dist)
            median_dist = np.median(p_dist)
            ax.axvline(mean_dist, color='orange', linestyle='dashed', linewidth=1, label='Mean Residual')
            ax.axvline(median_dist, color='r', linestyle='dashed', linewidth=1, label='Median Residual')

            if bins is None:
                bins = np.linspace(-0.4, 0.4, 80)
            ax.hist(p_dist, bins=bins)
            # ax.set_xlim([-2, 2])
            # ax.set_title(f'{phase} arrivals, RES: {high_res :.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}')
            ax.set_title(f'{model_dict["name"]}, {phase}')

            textstr = '\n'.join((
                f'{high_res:.3f}',
                f'{mae:.3f}',
                f'{rmse:.3f}',
            ))

            # Place the text box in the upper right corner of the plot
            ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.legend(loc='upper right')
        # plt.suptitle(model_dict['name'])
        # plt.tight_layout()
        plt.show()