import os
import yaml
import torch
import torch.nn.functional as F
import re
import os
import glob
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
from train import LightningSequenceModel
from simple_train import SimpleSeqModel
from dataloaders.data_utils.costa_rica_utils import get_metadata
from omegaconf import OmegaConf


def moving_average(signal: torch.Tensor | np.ndarray, window_size: int = 10) -> torch.Tensor:
    """
    Calculate moving average over signal.
    :param signal: Signal to average. [1, signal_length] or [signal_length]
    :param window_size: Length of moving average. Defaults to 10
    :return: Averaged signal with the same length as the input
    """
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)

    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    signal = signal.float()

    # Create the convolution kernel with equal weights
    kernel = torch.ones(window_size) / window_size

    # Reshape the kernel to match the shape required for F.conv1d
    kernel = kernel.view(1, 1, -1)

    # Apply padding to the tensor to maintain the output size
    padding = window_size // 2
    padded_tensor = F.pad(signal, (padding, padding), mode='reflect')

    # Apply the convolution
    moving_avg = F.conv1d(padded_tensor.unsqueeze(0), kernel).squeeze(0)

    return moving_avg


def _extract_step_number(filename):
    match = re.search(r'step=(\d+)\.ckpt$', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint(
        checkpoint_path: str,
        location: str = 'cpu',
        return_path: bool = False,
        simple: bool = False,
        d_data: int = 3,
        return_random_init: bool = False,
) -> tuple[LightningSequenceModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu.
    If no checkpoint is specified, the folder is searched for checkpoints and the one with the highest
    step number is returned.
    :param return_path: if true, the path to the checkpoint will be returned
    :param location: device to map the checkpoint to (e.g. cuda or cpu). Defaults to 'cpu'
    :param checkpoint_path: path to checkpoint file. The hparams file is extracted automatically
    :param simple: Load SimpleSeqModel
    :param d_data: Data dimension for loading SimpleSeqModel, default is 3
    :param return_random_init: If true, the state dict will not be loaded and a randomly initialized model
    is returned
    :return: LightningSequenceModel, hparams
    """
    if not checkpoint_path.endswith('.ckpt'):
        # the path does not directly lead to checkpoint, we search for checkpoints in directory
        all_files = []

        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                file_path = os.path.join(root, file)
                step_number = _extract_step_number(file)
                if step_number is not None:
                    all_files.append((step_number, file_path))
        all_files.sort(key=lambda x: x[0])
        checkpoint_path = all_files[-1][1]

    hparam_path = '/'.join(checkpoint_path.split('/')[:-2]) + '/hparams.yaml'

    if not os.path.isfile(checkpoint_path):
        print('NO CHECKPOINT FOUND')
        return None
    
    # Try to load hyperparameters from separate hparams.yaml file first
    if not os.path.isfile(hparam_path):
        print('NO HPARAM YAML FOUND - extracting from checkpoint')
        # Fallback: Extract hyperparameters directly from checkpoint
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=location, weights_only=False)
            if 'hyper_parameters' in checkpoint_data:
                hparams = checkpoint_data['hyper_parameters']
                print('Successfully extracted hyperparameters from checkpoint')
            else:
                print('ERROR: No hyper_parameters found in checkpoint')
                hparams = None
        except Exception as e:
            print(f'ERROR loading checkpoint for hyperparameters: {e}')
            hparams = None
    else:
        # Load from separate hparams.yaml file (legacy approach)
        with open(hparam_path, 'r') as f:
            hparams = yaml.safe_load(f)

    print(f'Loading checkpoint from {checkpoint_path}')
    if hparams is not None:
        name = hparams['experiment_name']
        print(f'Experiment name: {name}')

    if simple:
        # Fix for fine-tuning checkpoints that don't have model._name_
        if '_name_' not in hparams.get('model', {}):
            print('no param _name_')
            # Fix for fine-tuning checkpoints: add missing model._name_
            if 'pretrained' in hparams.get('model', {}):
                print('Fine-tuning checkpoint detected - adding model._name_=contrastive_wrapper')
                # Need to convert to OmegaConf first and disable struct mode to add new keys
                from omegaconf import OmegaConf as OC
                hparams = OC.create(hparams) if not OC.is_config(hparams) else hparams
                OC.set_struct(hparams, False)  # Temporarily disable struct mode
                hparams['model']['_name_'] = 'contrastive_wrapper'
                OC.set_struct(hparams, True)  # Re-enable struct mode
        model = SimpleSeqModel(OmegaConf.create(hparams), d_data=d_data)
        if not return_random_init:
            model.load_state_dict(torch.load(checkpoint_path, map_location=location, weights_only=False)['state_dict'])
    else:
        model = LightningSequenceModel.load_from_checkpoint(checkpoint_path, map_location=location)
    if return_path:
        return model, hparams, checkpoint_path
    else:
        return model, hparams


def get_pipeline_components(pl_module: LightningSequenceModel):
    """
    Extract encoder, decoder and model backbone from LightningSequenceModel.
    The components are put in eval mode.
    :param pl_module: LightningSequenceModel (e.g. loaded from checkpoint)
    :return: Encoder, Decoder, Model
    """
    encoder = pl_module.encoder.eval()
    decoder = pl_module.decoder.eval()
    model = pl_module.model.eval()

    # if isinstance(model, Sashimi):
    # model.setup_rnn()

    return encoder, decoder, model


def print_hparams(hparams: dict):
    print(yaml.dump(hparams))


def get_model_summary(model: LightningSequenceModel, max_depth=1):
    summary = ModelSummary(model, max_depth=max_depth)
    return summary


def get_sorted_file_list(dir_path: str):
    file_paths = glob.glob(os.path.join(dir_path, '*.pt'))
    year_day_list = []
    for file in file_paths:
        name = file.split('/')[-1]
        meta = get_metadata(name)
        year_day_list.append((meta['year'], meta['day'], file))

    # sort list by year and day
    year_day_list = sorted(year_day_list, key=lambda x: x[0] + x[1])

    file_paths = [f for y, d, f in year_day_list]
    return file_paths


def checkpoint_testing():
    ckpt_path = '../wandb_logs/MA/2024-09-24__15_55_22'
    model, _ = load_checkpoint(ckpt_path)


if __name__ == '__main__':
    checkpoint_testing()
