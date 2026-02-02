#!/usr/bin/env python3
"""
Direct evaluation script for fine-tuned xLSTM phase picking models.
Bypasses the PhasePickerLit wrapper which has issues with xLSTM state_dict loading.
"""
import sys
import argparse
import torch
import tempfile
from omegaconf import OmegaConf

sys.path.insert(0, '/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling')

from simple_train import SimpleSeqModel
import evaluation.pick_eval as pe


def load_xlstm_checkpoint(ckpt_path, device='cuda'):
    """
    Load xLSTM fine-tuned checkpoint, fixing config issues.
    Based on eval_foreshock_simple.py approach.
    """
    print(f"Loading checkpoint from {ckpt_path}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['hyper_parameters']

    # Fix config
    OmegaConf.set_struct(cfg, False)

    # Disable pretrained loading (the file path may not exist during eval)
    if 'pretrained' in cfg.model:
        print(f"Original pretrained path: {cfg.model.pretrained}")
        cfg.model.pretrained = None

    # Add encoder config if missing or incomplete
    if 'encoder' not in cfg or '_name_' not in cfg.encoder:
        print("Adding missing encoder config")
        cfg.encoder = OmegaConf.create({
            '_name_': 'conv-down-encoder-contrastive',
            'kernel_size': 3,
            'n_layers': 2,
            'dim': 256,
            'stride': 2,
        })

    # Remove pretrained from encoder if present
    if 'pretrained' in cfg.encoder:
        cfg.encoder.pop('pretrained')

    # Detect actual d_model from checkpoint if present
    detected_d_model = None
    for key in ['encoder.final_projection.weight', 'encoder.mask_emb']:
        if key in ckpt['state_dict']:
            detected_d_model = int(ckpt['state_dict'][key].shape[0])
            break

    if detected_d_model is not None:
        print(f"Detected d_model from checkpoint: {detected_d_model}")
        cfg.model.d_model = detected_d_model

    # Harmonize kernel chunk size for evaluation
    try:
        if int(cfg.model.get('chunk_size', 128)) > 128:
            print("Overriding model.chunk_size -> 128 for evaluation")
            cfg.model.chunk_size = 128
    except Exception:
        pass

    OmegaConf.set_struct(cfg, True)

    # Update checkpoint with fixed config
    ckpt['hyper_parameters'] = cfg

    # Save temp fixed checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ckpt') as tmp:
        torch.save(ckpt, tmp.name)
        tmp_path = tmp.name

    print(f"Loading model from fixed checkpoint (strict=False)...")
    # Load model (strict=False to handle any architecture differences)
    model = SimpleSeqModel.load_from_checkpoint(tmp_path, map_location=device, strict=False)
    model.eval()

    print("Model loaded successfully")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Direct evaluation for xLSTM fine-tuned phase picking models"
    )
    parser.add_argument(
        "checkpoint", type=str, help="Path to model checkpoint"
    )
    parser.add_argument(
        '--target_dataset', type=str, default='GEOFON',
        help="Name of the target dataset (ETHZ, GEOFON, STEAD)"
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        '--sets', type=str, default='test',
        help="Comma-separated splits to evaluate (train,dev,test)"
    )
    parser.add_argument(
        '--norm_type', type=str, default='std',
        help="Type of amplitude norm for data loader (peak, std)"
    )
    parser.add_argument(
        '--save_tag', type=str, default='eval_direct',
        help="Subfolder tag under evals/"
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_xlstm_checkpoint(args.checkpoint, device=device)

    # Prepare paths
    ckpt_path = args.checkpoint
    if '.ckpt' in ckpt_path:
        ckpt_path = '/'.join(ckpt_path.split('/')[:-2])

    target_path = 'evaluation/eval_tasks/' + args.target_dataset

    print(f"\nEvaluating on {args.target_dataset} dataset, splits: {args.sets}")
    print(f"Checkpoint directory: {ckpt_path}")
    print(f"Target path: {target_path}")
    print(f"Save tag: {args.save_tag}\n")

    # Run evaluation using the same function as pick_evaluation_script.py
    pe.save_pick_predictions(
        model=model,
        target_path=target_path,
        ckpt_path=ckpt_path,
        sets=args.sets,
        save_tag=args.save_tag,
        batch_size=64,
        num_workers=args.num_workers
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
