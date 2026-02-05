import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

from dataloaders.foreshock_aftershock_lit import ForeshockAftershockLitDataset
from simple_train import SimpleSeqModel, load_checkpoint
from hydra import compose, initialize_config_dir


# Display labels for 9-class and other configurations
DISPLAY_LABELS = {
    2: ["Foreshock", "Aftershock"],
    4: ["FEQ1", "FEQ2", "AEQ1", "AEQ2"],
    8: ["FEQ1", "FEQ2", "FEQ3", "FEQ4", "AEQ1", "AEQ2", "AEQ3", "AEQ4"],
    9: ["FEQ1", "FEQ2", "FEQ3", "FEQ4", "Visso", "AEQ1", "AEQ2", "AEQ3", "AEQ4"],
}


def build_config(exp_name: str):
    # Resolve paths relative to this file so it works from any CWD
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.join(proj_dir, 'configs')
    cfg_base_path = os.path.join(cfg_dir, 'config.yaml')
    cfg_exp_path = os.path.join(cfg_dir, 'experiment', f'{exp_name}.yaml')
    if not os.path.isfile(cfg_base_path):
        raise FileNotFoundError(f"Missing base config: {cfg_base_path}")
    if not os.path.isfile(cfg_exp_path):
        raise FileNotFoundError(f"Missing experiment config: {cfg_exp_path}")
    # Compose via Hydra to resolve defaults
    with initialize_config_dir(config_dir=cfg_dir, job_name="eval_foreshock", version_base="1.1"):
        cfg = compose(config_name='config.yaml', overrides=[
            f'experiment={exp_name}',
            'train.disable_pretraining=true',
        ])
    # Provide full encoder config if missing (defensive)
    if 'encoder' not in cfg or getattr(cfg.encoder, '_name_', None) is None:
        cfg.encoder = OmegaConf.create({
            '_name_': 'conv-down-encoder-contrastive',
            'kernel_size': 3,
            'n_layers': 2,
            'dim': 256,
            'stride': 2,
        })
    return cfg


@torch.no_grad()
def evaluate(ckpt_path: str, data_dir: str, num_classes: int = 9, batch_size: int = 32,
             save_fig: bool = True, save_results: bool = True, output_dir: str = None):
    """
    Evaluate foreshock-aftershock classification model.

    Args:
        ckpt_path: Path to model checkpoint
        data_dir: Path to foreshock data directory
        num_classes: Number of classification classes (2, 4, 8, or 9)
        batch_size: Batch size for evaluation
        save_fig: Whether to save confusion matrix figure
        save_results: Whether to save results JSON
        output_dir: Directory to save outputs (default: ../evaluation_results/)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint hparams to mirror dataset args
    print("Loading dataset...")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get('hyper_parameters')
    ds_kwargs = {
        'data_dir': data_dir,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'event_split_method': 'temporal',
        'component_order': 'ZNE',
        'seed': 42,
        'remove_class_overlapping_dates': False,
        'train_frac': 0.70,
        'val_frac': 0.10,
        'test_frac': 0.20,
        'dimension_order': 'NWC',
        'demean_axis': 1,
        'amp_norm_axis': 1,
        'amp_norm_type': 'std',
        'num_workers': 0,
        'collator': None,
        # Ensure per-sample sequence length divisible by 2048 for pool=[4,4] & chunk_size=128
        'seq_len_multiple': 2048,
    }
    try:
        # Merge dataset overrides from checkpoint if available
        if isinstance(cfg, dict) and 'dataset' in cfg:
            ds_cfg = cfg['dataset']
            # OmegaConf-friendly access
            def _get(d, k, default=None):
                try:
                    return d.get(k, default)
                except Exception:
                    return getattr(d, k, default)
            ds_kwargs.update({
                'num_classes': int(_get(ds_cfg, 'num_classes', num_classes)),
                'batch_size': int(_get(ds_cfg, 'batch_size', batch_size)),
                'event_split_method': _get(ds_cfg, 'event_split_method', 'temporal'),
                'component_order': _get(ds_cfg, 'component_order', 'ZNE'),
                'seed': int(_get(ds_cfg, 'seed', 42)),
                'train_frac': float(_get(ds_cfg, 'train_frac', 0.70)),
                'val_frac': float(_get(ds_cfg, 'val_frac', 0.10)),
                'test_frac': float(_get(ds_cfg, 'test_frac', 0.20)),
                'dimension_order': _get(ds_cfg, 'dimension_order', 'NWC'),
                'demean_axis': _get(ds_cfg, 'demean_axis', 1),
                'amp_norm_axis': _get(ds_cfg, 'amp_norm_axis', 1),
                'amp_norm_type': _get(ds_cfg, 'amp_norm_type', 'std'),
                'num_workers': int(_get(ds_cfg, 'num_workers', 0)),
            })
    except Exception:
        pass

    ds = ForeshockAftershockLitDataset(**ds_kwargs)
    test_loader = ds.test_loader
    print(f"Test set: {len(test_loader.dataset)} samples")

    # Try exact restore of LightningModule first
    print("Loading model...")
    model = None
    # Path 1: Our load_checkpoint helper (reads hparams.yaml alongside ckpt)
    try:
        # Ensure constructor skips any pretraining init while restoring from Lightning ckpt
        SimpleSeqModel.SKIP_PRETRAINED_INIT = True
        model, _ = load_checkpoint(ckpt_path, location=device)
        model.eval()
        model.to(device)
        print("[eval] Exact checkpoint restore via load_checkpoint succeeded.")
    except Exception as e:
        print(f"[eval] load_checkpoint path failed: {e}")
        # Path 2: Lightning load_from_checkpoint with skip flag
        try:
            SimpleSeqModel.SKIP_PRETRAINED_INIT = True
            model = SimpleSeqModel.load_from_checkpoint(ckpt_path, map_location=device)
            model.eval()
            model.to(device)
            print("[eval] Exact Lightning restore succeeded.")
        except Exception as e2:
            print(f"[eval] Exact restore failed: {e2}")
            model = None
        finally:
            try:
                SimpleSeqModel.SKIP_PRETRAINED_INIT = False
            except Exception:
                pass
    finally:
        try:
            SimpleSeqModel.SKIP_PRETRAINED_INIT = False
        except Exception:
            pass

    if model is None:
        # Disable struct mode to modify
        OmegaConf.set_struct(cfg, False)

        # Detect actual feature dim (d_model) from checkpoint encoder weights
        detected_d_model = None
        sd = state.get('state_dict', {})
        if 'encoder.final_projection.weight' in sd:
            detected_d_model = int(sd['encoder.final_projection.weight'].shape[0])
        elif 'encoder.mask_emb' in sd:
            detected_d_model = int(sd['encoder.mask_emb'].shape[0])
        # Detect decoder input dimension from checkpoint if present
        dec_in_ckpt = None
        if 'decoder.linear.weight' in sd and sd['decoder.linear.weight'].dim() == 2:
            dec_in_ckpt = int(sd['decoder.linear.weight'].shape[1])
        elif 'decoder.net.0.weight' in sd and sd['decoder.net.0.weight'].dim() == 3:
            dec_in_ckpt = int(sd['decoder.net.0.weight'].shape[1])

        if detected_d_model is not None:
            print(f"\n[eval] Detected d_model from ckpt: {detected_d_model}")
            cfg.model.d_model = detected_d_model

        if dec_in_ckpt is not None:
            try:
                cfg.decoder.manual_input_dim = dec_in_ckpt
                print(f"[eval] Detected decoder input dim from ckpt: {dec_in_ckpt}\n")
            except Exception:
                pass

        # Disable pretrained loading (the file doesn't exist during eval anyway)
        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None

        # Fix encoder config - add full config if missing
        if 'encoder' not in cfg or '_name_' not in cfg.encoder:
            # Get the actual encoder config from the pretrained checkpoint that was used
            cfg.encoder = OmegaConf.create({
                '_name_': 'conv-down-encoder-contrastive',
                'kernel_size': 3,
                'n_layers': 2,
                'dim': 256,
                'stride': 2,
            })

        # Harmonize kernel chunk size for evaluation: ensure it works at all stages
        # Some checkpoints used chunk_size=256 during training; center stage length is 128,
        # which breaks the Triton kernel heuristic that requires L % target_chunk_size == 0.
        try:
            if int(cfg.model.get('chunk_size', 128)) > 128:
                print("\n[eval] Overriding model.chunk_size -> 128 for evaluation compatibility\n")
                cfg.model.chunk_size = 128
        except Exception:
            pass

        # Remove pretrained flag from encoder (not a valid init arg)
        if 'pretrained' in cfg.encoder:
            cfg.encoder.pop('pretrained')

        OmegaConf.set_struct(cfg, True)

        # Instantiate model and load weights (robust fallback)
        model = SimpleSeqModel(cfg, d_data=3)
        # Load checkpoint with shape-safe filtering: drop keys that don't match current shapes
        current_sd = model.state_dict()
        filtered = {}
        mismatched = 0
        for k, v in state['state_dict'].items():
            if k in current_sd and current_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                mismatched += 1
        if mismatched:
            print(f"[eval] Skipped {mismatched} key(s) with mismatched shapes during load.")
        model.load_state_dict(filtered, strict=False)
        try:
            model.model.pretraining = False
            model.encoder.pretraining = False
        except Exception:
            pass
        model.eval()
        model.to(device)
        print("Model loaded successfully (fallback path)")

    # Run inference
    print("Running inference...")
    all_preds = []
    all_targets = []
    total_loss = 0.0
    n_samples = 0

    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        logits, targets = model.forward((x, y), batch_idx)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item() * targets.shape[0]
        n_samples += targets.shape[0]
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    test_loss = total_loss / max(1, n_samples)
    test_acc = accuracy_score(all_targets, all_preds)

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    cm_percentage = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_display = np.rint(cm_percentage).astype(int)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Total Samples: {n_samples}")
    print("="*60)

    # Per-class accuracy
    per_class_acc = cm_display.diagonal()
    labels = DISPLAY_LABELS.get(num_classes, [f"Class{i}" for i in range(num_classes)])
    print("\nPER-CLASS ACCURACY:")
    print("-"*60)
    for label, acc in zip(labels, per_class_acc):
        bar = '█' * int(acc / 5)
        print(f"{label:8s} | {acc:3d}% {bar}")
    print("-"*60)
    print(f"MEAN     | {per_class_acc.mean():.2f}%")
    print("="*60 + "\n")

    # Save outputs
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix figure
    if save_fig:
        fig, ax = plt.subplots(figsize=(12, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_display,
            display_labels=labels,
        )
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Reds")
        ax.set_title(
            f"Confusion Matrix (xLSTM-UNet) | Accuracy: {test_acc*100:.2f}%",
            fontsize=16,
            fontweight='bold'
        )
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f'foreshock_xlstm_confusion_{num_classes}class.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {fig_path}")
        plt.close(fig)

    # Save results JSON
    if save_results:
        results = {
            'checkpoint': ckpt_path,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'num_samples': int(n_samples),
            'num_classes': num_classes,
            'per_class_accuracy': per_class_acc.tolist(),
            'class_labels': labels,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_percentage': cm_display.tolist(),
        }

        results_path = os.path.join(output_dir, f'foreshock_xlstm_results_{num_classes}class.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to: {results_path}")

    return test_loss, test_acc, cm, cm_display


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate xLSTM-UNet on foreshock-aftershock classification (SeisLM-style)'
    )
    parser.add_argument('--ckpt', required=True, help='Path to finetuned foreshock checkpoint (.ckpt)')
    parser.add_argument(
        '--data_dir',
        default=os.environ.get('SEIS_DATA_DIR'),
        help='Foreshock data directory (or set env var SEIS_DATA_DIR)',
    )
    parser.add_argument('--num_classes', type=int, default=9, choices=[2, 4, 8, 9],
                        help='Number of classes (2, 4, 8, or 9)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--no_save_fig', action='store_true', help='Do not save confusion matrix figure')
    parser.add_argument('--no_save_results', action='store_true', help='Do not save results JSON')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ../evaluation_results/)')
    args = parser.parse_args()

    if not args.data_dir:
        raise SystemExit("Missing --data_dir (or set SEIS_DATA_DIR).")

    evaluate(
        args.ckpt,
        args.data_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        save_fig=not args.no_save_fig,
        save_results=not args.no_save_results,
        output_dir=args.output_dir
    )
