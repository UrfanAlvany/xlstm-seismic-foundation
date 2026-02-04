import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np

from evaluation.eval_utils import load_checkpoint, get_pipeline_components, print_hparams, get_model_summary
from dataloaders.seisbench_auto_reg import get_eval_augmentations
from models.phasenet_wrapper import PhaseNetWrapper
import torch
import torch.nn as nn
from models.benchmark_models import SeisBenchModuleLit
from dataloaders.seisbench_auto_reg import phase_dict
from tqdm import tqdm
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import argparse
from sklearn import metrics

from dataloaders.insight_eval_loader import InsightEvalDataset
from torch.utils.data import DataLoader
from obspy import UTCDateTime
import ast


# set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def prepare_eval(ckpt_path: str, device: str = 'cpu') -> tuple[dict, dict]:
    """
    Prepare the evaluation by loading the checkpoint and extracting the model and the dataloaders.
    
    @param ckpt_path: Path to the checkpoint file.
    @param device: Device to load the model onto (default is 'cpu').
    @return: A dict containing the model components and configuration file.
    """

    # load the checkpoint
    ckpt, hparams = load_checkpoint(ckpt_path, location=device, simple=True)

    # extract model components (should be in eval mode))
    encoder, decoder, model = get_pipeline_components(ckpt)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    model = model.to(device)

    encoder.eval()
    decoder.eval()
    model.eval()

    model_dict = {
        'model': model,
        'encoder': encoder,
        'decoder': decoder,
        'hparams': hparams,
    }

    return model_dict


def get_predictions(
        model_dict: dict,
        x: torch.Tensor,
        device: str = 'cpu',
        return_logits: bool = False
) -> torch.Tensor:
    """
    Run input x through model pipeline and return predictions.

    @param model_dict: Dictionary containing the model components and configuration.
    @param x: Input tensor of shape (batch_size, sequence_length, num_channels).
    @param device: Device to run the model on (default is 'cpu').
    @param return_logits: If True, return raw logits; if False, return softmax probabilities (default is False).
    @return: Predictions tensor of shape (batch_size, sequence_length, num_classes)."""
    x = x.to(device)

    with torch.no_grad():
        encoded = model_dict['encoder'](x)
        out, _ = model_dict['model'](encoded)
        pred = model_dict['decoder'](out)
    if return_logits:
        return pred.detach().cpu()
    else:
        return F.softmax(pred, dim=-1).detach().cpu()
    

def get_picks_and_scores(
        pred: torch.Tensor,
        window_borders: torch.Tensor | None = None,
):
    """
    Extract picks and scores from the predictions.

    @param pred: Predictions tensor of shape (batch_size, sequence_length, 3), where the last dimension corresponds to P, S, and Noise classes.
    @param window_borders: Optional tensor [batch_size, 2] indicating the start and end indices for each sample in the batch.
    @return: A tuple containing:
        - score_detection: Tensor of shape (batch_size,) with detection scores.
        - score_p_or_s: Tensor of shape (batch_size,) with P/S ratio scores.
        - p_sample: Tensor of shape (batch_size,) with the index of the P pick for each sample.
        - s_sample: Tensor of shape (batch_size,) with the index of the S pick for each sample.
    """
    score_detection = torch.zeros(pred.shape[0])
    score_p_or_s = torch.zeros(pred.shape[0])
    p_sample = torch.zeros(pred.shape[0], dtype=int)
    s_sample = torch.zeros(pred.shape[0], dtype=int)
    
    for i in range(pred.shape[0]):
        if window_borders is not None:
            start_sample, end_sample = window_borders[i]
            local_pred = pred[i, start_sample:end_sample, :]
        else:
            local_pred = pred[i, :, :]

        score_detection[i] = torch.max(1 - local_pred[:, -1])
        score_p_or_s[i] = torch.max(local_pred[:, 0]) / torch.max(local_pred[:, 1])
        p_sample[i] = torch.argmax(local_pred[:, 0])
        s_sample[i] = torch.argmax(local_pred[:, 1])

    return score_detection, score_p_or_s, p_sample, s_sample


def save_pick_predictions(
        ckpt_path: str,
        target_path: str,
        sets: list[str] = ['dev', 'test'],
        tasks: list[str] = ['1', '23'],
        output_dir: str = 'eval',
        batch_size: int = 32,
        num_workers: int = 0,
):
    print('=' * 50, '\n' + ' ' * 13 + 'Pick Evaluation Insight' + '\n' + '=' * 50, '\n\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if ckpt_path.endswith('.ckpt'):
        save_path = '/'.join(ckpt_path.split('/')[:-2]) + f'/{output_dir}/'
    else:
        save_path = ckpt_path + f'/{output_dir}/'

    print(f"Will save predictions to directory: {save_path}")

    model_dict = prepare_eval(
        ckpt_path=ckpt_path,
        device=device,
    )

    hparams = model_dict['hparams']

    eval_dataset = InsightEvalDataset(
        data_dir='dataloaders/data/insight',
        targets=pd.DataFrame(),  # Empty DataFrame, will be set later
        task='task1',  # Default task, will be changed later
        fs=20,
        window_size_in_samples=hparams['dataset']['window_size_samples'],
        components=hparams['dataset']['components'],
        norm_type=hparams['dataset']['norm_type'],
    )

    if tasks == ['4']:
        sets = ['full_data']

    # loop through the sets:
    for eval_set in sets:
        print(f"\nEvaluating on {eval_set} set\n")

        for task in tasks:
            print(f'Processing task {task}')
            task_csv = Path(target_path) / f'task{task}.csv'

            if not task_csv.is_file():
                print(f"Task {task} targets not found at {task_csv}. Skipping...")
                continue

            # load the targets for the current task
            task_targets = pd.read_csv(task_csv)

            # filter the targets for the current evaluation set
            if task in ['1', '23']:
                task_targets = task_targets[task_targets['trace_split'] == eval_set]

            # prepare dataloader
            eval_dataset.set_new_task(
                task=f'task{task}',
                targets=task_targets
            )
            loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            all_event_detections = []
            all_phase_discriminations = []
            all_p_onsets = []
            all_s_onsets = []
            all_start_times = []
            all_window_border_indices = []

            # loop through the batches with tqdm
            for batch in tqdm(loader, desc=f"Processing {eval_set} set for task {task}"):

                waveforms, start_times, window_border_indices = batch
                # get predictions
                pred = get_predictions(model_dict, waveforms, device=device, return_logits=False)
                
                # get picks and scores
                if task == '4':
                    # for task 4, we dont use the window borders
                    ev_det, p_disc, p_on, s_on = get_picks_and_scores(pred=pred, window_borders=None)
                else:
                    ev_det, p_disc, p_on, s_on = get_picks_and_scores(pred=pred, window_borders=window_border_indices)
                
                # append to the lists
                all_event_detections.append(ev_det)
                all_phase_discriminations.append(p_disc)
                all_p_onsets.append(p_on)
                all_s_onsets.append(s_on)
                all_start_times += start_times
                all_window_border_indices.append(window_border_indices)
            
            # concatenate the results
            all_event_detections = torch.cat(all_event_detections)
            all_phase_discriminations = torch.cat(all_phase_discriminations)
            all_p_onsets = torch.cat(all_p_onsets)
            all_s_onsets = torch.cat(all_s_onsets)
            all_window_border_indices = torch.cat(all_window_border_indices)
            
            # add results to DataFrame
            task_targets['score_detection'] = all_event_detections
            task_targets['score_p_or_s'] = all_phase_discriminations
            task_targets['p_sample_pred'] = all_p_onsets
            task_targets['s_sample_pred'] = all_s_onsets
            task_targets['trace_start_time'] = all_start_times

            # convert start times to UTCDateTime and calculate prediction times based on the sample indices
            fs = float(hparams['dataset']['fs']) # sampling frequency

            # convert start times to UTCDateTime
            # start times refer to the start of the window within a trace, not to be confused with the trace start time
            utc_start_times = [UTCDateTime(ts) for ts in task_targets['start_time']] 
            
            # calculate prediction times for P and S phases
            # the sample indices are relative to the start of the evaluation window.
            p_prediction_times = [str(t + p.item() / fs) for t, p in zip(utc_start_times, all_p_onsets)]
            s_prediction_times = [str(t + s.item() / fs) for t, s in zip(utc_start_times, all_s_onsets)]
            task_targets['p_prediction_time'] = p_prediction_times
            task_targets['s_prediction_time'] = s_prediction_times

            # save predictions to CSV
            file_name = Path(save_path) / f"{eval_set}_task{task}.csv"
            file_name.parent.mkdir(parents=True, exist_ok=True)
            task_targets.to_csv(file_name, index=False)
            print(f"Saved predictions for {eval_set} set to {file_name}")
            

def get_results_event_detection(pred_path: str) -> dict:
    """
    Calculate evaluation metrics for event detection from predictions.
    
    @param pred_path: Path to the CSV file containing predictions.
    @return: A dictionary containing evaluation metrics such as AUC, FPR, TPR, precision, recall, F1 score, and best F1 score.
    """
    pred_path = Path(pred_path)
    pred = pd.read_csv(pred_path)

    pred['trace_type_bin'] = pred['trace_type'] == 'marsquake' # True for marsquakes, False for noise
    pred["score_detection"] = pred["score_detection"].fillna(0.0)  # Fill NaN values with 0.0

    fpr, tpr, roc_thresh = metrics.roc_curve(
        pred['trace_type_bin'],
        pred['score_detection'],
    )
    prec, recall, thr = metrics.precision_recall_curve(
        pred['trace_type_bin'],
        pred['score_detection'],
    )
    auc = metrics.roc_auc_score(
        pred['trace_type_bin'],
        pred['score_detection'],
    )

    f1 = 2 * prec * recall / (prec + recall)
    f1_threshold = thr[np.nanargmax(f1)]
    best_f1 = np.max(f1)

    best_f1_idx = np.argmax(f1)
    best_f1_xy = (0, 0) #(fpr[best_f1_idx], tpr[best_f1_idx])

    return {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'prec': prec,
        'recall': recall,
        'f1': f1,
        'f1_threshold': f1_threshold,
        'best_f1': best_f1,
        'best_f1_xy': best_f1_xy,
        'best_f1_idx': best_f1_idx,
    }


def get_results_event_task4(pred_path: str, ignore_SF_events: bool = True, remove_duplicates: bool = False) -> dict:
    """
    Calculate evaluation metrics for event detection from predictions for task 4.
    
    @param pred_path: Path to the CSV file containing predictions.
    @param ignore_SF_events: If True, ignore events that are only SF events (i.e. events named T[xxxx][z]).
    @param remove_duplicates: If True, remove duplicate events based on their detection scores.
    @return: A dictionary containing 3 sets of evaluation metrics for different experiments:
        - 'v14': Metrics for original catalgue events (catalogue_v14) (S[xxxx][z] and T[xxxx][z]).
        - 'MQC': Metrics for events from the MQNet_DeepCatalogue. (D[xxxx][z]).
        - 'all': Metrics for both original and MQNet events.
    """
    # load predictions
    pred_path = Path(pred_path)
    pred = pd.read_csv(pred_path)

    experiments = {
        'v14': 'v14_bin',
        'MQC': 'MQNet_bin',
        'all': 'contains_event',
    }
    # TODO: deal with super high frequency events named T[xxxx][z]
    # we do not train on these, at least for not.

    if ignore_SF_events:
        if type(pred.iloc[0]['events']) is str:
            # convert events from string to list if they are stored as strings
            # this is necessary because the events are stored as strings in the CSV file
            pred['events'] = pred['events'].apply(
                ast.literal_eval
            )
        # mark all events that are only SF events (i.e. events named T[xxxx][z])
        pred['only_SF_events'] = pred['events'].apply(
            lambda lst: bool(lst) and all(isinstance(x, str) and x.startswith('T') for x in lst)
        )
        
        # romove all events that are only SF events
        #pred = pred[not pred['only_SF_events']]

        # change labels for v14_bin, if the only event in that window is an SF event.
        pred.loc[pred['only_SF_events'], 'v14_bin'] = False

    if remove_duplicates:
        # Convert lists to tuples for grouping
        pred['events_tuple'] = pred['events'].apply(tuple)

        # Mask for empty events lists
        empty_mask = pred['events'].apply(lambda x: len(x) == 0)

        # Keep all rows with empty events (negative examples)
        df_empty = pred[empty_mask]

        # For non-empty events, keep only the row with max score_detection per unique events list
        df_non_empty = pred[~empty_mask]
        idx = df_non_empty.groupby('events_tuple')['score_detection'].idxmax()
        df_non_empty_dedup = df_non_empty.loc[idx]

        # Combine and reset index
        pred = pd.concat([df_empty, df_non_empty_dedup]).reset_index(drop=True)

    results = {}

    for experiment, column in experiments.items():
        pred['score_detection'] = pred['score_detection'].fillna(0.0)  # Fill NaN values with 0.0

        fpr, tpr, roc_thresh = metrics.roc_curve(
            pred[column],
            pred['score_detection'],
        )
        prec, recall, thr = metrics.precision_recall_curve(
            pred[column],
            pred['score_detection'],
        )
        auc = metrics.roc_auc_score(
            pred[column],
            pred['score_detection'],
        )
        f1 = 2 * prec * recall / (prec + recall)
        f1_threshold = thr[np.nanargmax(f1)]
        best_f1 = np.max(f1)
        best_f1_idx = np.argmax(f1)
        best_f1_xy = (0, 0) 

        results[experiment] = {
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'prec': prec,
            'recall': recall,
            'f1': f1,
            'f1_threshold': f1_threshold,
            'best_f1': best_f1,
            'best_f1_xy': best_f1_xy,
            'best_f1_idx': best_f1_idx,
        }
    return results


def get_results_phase_identification(pred_path) -> dict:
    """
    Calculate evaluation metrics for phase identification from predictions.

    @param pred_path: Path to the CSV file containing predictions.
    @return: A dictionary containing evaluation metrics such as AUC, FPR, TPR, precision, recall, F1 score, MCC, and best F1 score.
    """
    # load predictions
    pred_path = Path(pred_path)
    pred = pd.read_csv(pred_path)

    pred["phase_label_bin"] = pred["phase_label"] == "P" # True for P phase, False for S phase
    pred["score_p_or_s"] = pred["score_p_or_s"].fillna(0)
    
    fpr, tpr, roc_thresh = metrics.roc_curve(
        pred["phase_label_bin"], pred["score_p_or_s"]
    )
    prec, recall, thr = metrics.precision_recall_curve(
        pred["phase_label_bin"], pred["score_p_or_s"]
    )
    f1 = 2 * prec * recall / (prec + recall)
    f1_threshold = thr[np.nanargmax(f1)]
    best_f1 = np.nanmax(f1)
    best_f1_idx = np.argmax(f1)
    best_f1_xy = (0, 0)#(fpr[best_f1_idx], tpr[best_f1_idx])

    auc = metrics.roc_auc_score(
        pred["phase_label_bin"], pred["score_p_or_s"]
    )

    mcc_thrs = np.sort(pred["score_p_or_s"].values)
    mcc_thrs = mcc_thrs[np.linspace(0, len(mcc_thrs) - 1, 50, dtype=int)]
    mccs = []
    for thr in mcc_thrs:
        mccs.append(
            metrics.matthews_corrcoef(
                pred["phase_label_bin"], pred["score_p_or_s"] > thr
            )
        )
    mcc = np.max(mccs)
    mcc_thr = mcc_thrs[np.argmax(mccs)]

    return {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'prec': prec,
        'recall': recall,
        'f1': f1,
        'f1_threshold': f1_threshold,
        'best_f1': best_f1,
        'mcc': mcc,
        'mcc_thr': mcc_thr,
        'best_f1_xy': best_f1_xy,
        'best_f1_idx': best_f1_idx,
    }


def get_results_onset_determination(pred_path):
    """
    Calculate evaluation metrics for onset determination from predictions.
    
    @param pred_path: Path to the CSV file containing predictions.
    @return: A dictionary containing the differences (in seconds) between predicted and ground truth onset times for P and S phases.
    """
    # load predictions
    pred_path = Path(pred_path)
    pred = pd.read_csv(pred_path)
    
    results = {}
    gt_column = 'phase_onset'

    for phase in ['P', 'S']:
        pred_phase = pred[pred["phase_label"] == phase]
        
        prediction_column = f"{phase.lower()}_prediction_time"
        pred_time = pd.to_datetime(pred_phase[prediction_column])
        gt_time = pd.to_datetime(pred_phase[gt_column])
        diff = (pred_time - gt_time).dt.total_seconds() # convert to seconds
        
        results[f'{phase}_onset_diff'] = diff
    
    return results



def evaluate_test_ckpt():
    path = 'wandb_logs/MA/2025-06-13__09_08_08'

    save_pick_predictions(
        ckpt_path=path,
        target_path='dataloaders/data/insight',
        sets=['train', 'dev', 'test'],
        output_dir='eval',
        batch_size=16,
        num_workers=0,
    )

if __name__ == '__main__':
    # parse arguments for ckpt_path, target_path, sets, tasks, output_dir, batch_size, num_workers
    parser = argparse.ArgumentParser(description='Evaluate pick predictions from a checkpoint.')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the checkpoint file.')
    parser.add_argument('--target_path', type=str, default='dataloaders/data/insight', help='Path to the target CSV files (default: dataloaders/data/insight).')
    parser.add_argument('--sets', type=str, nargs='+', default=['dev', 'test'], help='Sets to evaluate (default: dev test).')
    parser.add_argument('--tasks', type=str, nargs='+', default=['1', '23'], help='Tasks to evaluate (default: 1 23).')
    parser.add_argument('--output_dir', type=str, default='eval', help='Directory to save the predictions (default: eval).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation (default: 32).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading (default: 0).')
    args = parser.parse_args()

    save_pick_predictions(
        ckpt_path=args.ckpt_path,
        target_path=args.target_path,
        sets=args.sets,
        tasks=args.tasks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # example usage of this script:
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/MA/2025-06-13__09_08_08 --target_path dataloaders/data/insight --sets dev test --output_dir eval --batch_size 16 --num_workers 0

    # only evaluate task 4 (example):
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-06-13__12_49_23/checkpoints/callback-epoch=64-step=3185.ckpt --target_path dataloaders/data/insight --sets full_data --tasks 4 --output_dir eval --batch_size 32 --num_workers 2
    
    # s4d supervised
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-06-13__12_49_23/checkpoints/callback-epoch=64-step=3185.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    
    # Hydra complex supervised
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-09__17_03_30/checkpoints/callback-epoch=96-step=4753.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-09__17_03_30/checkpoints/callback-epoch=96-step=4753.ckpt --target_path dataloaders/data/insight --output_dir eval --batch_size 32 --num_workers 2 --tasks 4
    
    # finetune hydra wav2vec
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-10__10_55_42/checkpoints/callback-epoch=32-step=3185.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-10__10_55_42/checkpoints/callback-epoch=32-step=3185.ckpt --target_path dataloaders/data/insight --output_dir eval --batch_size 32 --num_workers 2 --tasks 4
    
    # finetune hydra wav2vec with quantization
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-11__16_01_03/checkpoints/callback-epoch=20-step=2032.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-11__16_01_03/checkpoints/callback-epoch=20-step=2032.ckpt --target_path dataloaders/data/insight --output_dir eval --batch_size 32 --num_workers 2 --tasks 4
    
    # finetune hydra wav2vec UNET 32k sample len
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-15__12_10_00/checkpoints/callback-epoch=20-step=3969.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 16 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-15__12_10_00/checkpoints/callback-epoch=20-step=3969.ckpt --target_path dataloaders/data/insight --output_dir eval --batch_size 16 --num_workers 2 --tasks 4
    
    # finetune hydra wav2vec UNET with quantization 16k sample len
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-15__12_31_37/checkpoints/callback-epoch=23-step=2278.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 16 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-15__12_31_37/checkpoints/callback-epoch=23-step=2278.ckpt --target_path dataloaders/data/insight/task4_16k --output_dir eval --batch_size 16 --num_workers 2 --tasks 4
    
    # hydra wav2vec UNET with quantization 16k sample len RAND INIT
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-16__14_59_50/checkpoints/callback-epoch=77-step=7642.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 16 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-16__14_59_50/checkpoints/callback-epoch=77-step=7642.ckpt --target_path dataloaders/data/insight/task4_16k --output_dir eval --batch_size 16 --num_workers 2 --tasks 4
    
    # S4D 16k sample len
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-06-13__11_49_06/checkpoints/callback-epoch=58-step=1416.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-06-13__11_49_06/checkpoints/callback-epoch=58-step=1416.ckpt --target_path dataloaders/data/insight/task4_16k --output_dir eval --batch_size 32 --num_workers 2 --tasks 4
    
    # hydra ts2vec fine-tuned
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-06-30__11_31_00/checkpoints/callback-epoch=72-step=7105.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    
    # Pure Mamba Patch regression fine-tuned
    # python3 -m evaluation.pick_eval_insight --ckpt_path wandb_logs/mars/2025-07-07__11_14_48/checkpoints/callback-epoch=41-step=2057.ckpt --target_path dataloaders/data/insight/targets_long --output_dir eval --batch_size 32 --num_workers 2 --sets train dev test
    
    