import torch
import torch.nn as nn
import torch.nn.functional as F
import tasks.metrics as M
import hydra
import omegaconf

from utils.config_utils import instantiate, to_list


class Task:
    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, **kwargs):
        self.dataset = dataset
        self.model = model
        if metrics is None: metrics = []
        self.metric_names = to_list(metrics)

        self.loss = instantiate(registry=M.metric_functions, config=loss, partial=True)
        if loss_val is not None:
            self.loss_val = instantiate(registry=M.metric_functions, config=loss_val, partial=True)
        else:
            self.loss_val = self.loss

    def metrics(self, x, y,):
        output_metrics = {
            metric: M.metric_functions[metric](x, y)
            for metric in self.metric_names if metric in M.metric_functions
        }
        return {**output_metrics}


task_registry = {
    'default': Task,
    'classification': Task,
    'regression': Task,
    'phase_pick': Task,
    'contrastive': Task,
}
