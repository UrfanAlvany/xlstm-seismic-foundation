<div align="center">

# WaveXLSTM

### Extended LSTM for Self-Supervised Seismology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**Getting Started**](#quick-start) | [**Architecture**](#architecture) | [**Experiments**](#experiments) | [**Results**](#results) | [**Citation**](#citation)

</div>

---

## What is WaveXLSTM?

**WaveXLSTM** applies Extended Long Short-Term Memory (xLSTM) to seismology, using a **U-Net architecture with mLSTM/sLSTM blocks** for self-supervised learning on seismic waveforms.

Building on the success of [SeisLM](https://github.com/seisbench/seislm) and inspired by [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477), we pretrain on **5M+ seismic waveforms** from 8 global datasets using contrastive learning with Gumbel-Softmax vector quantization. The pretrained model transfers effectively to downstream tasks including **phase picking** and **foreshock-aftershock classification**.

Key contributions:
- **xLSTM U-Net** combining bidirectional mLSTM blocks with skip connections for multi-scale temporal modeling
- **TFLA Triton kernels** enabling efficient mLSTM computation with linear memory scaling

---

## Key Features

### Architecture

- **xLSTM U-Net Backbone** with bidirectional mLSTM/sLSTM blocks and skip connections
- **Multi-scale pooling** (4096 → 1024 → 256 tokens) capturing both fine-grained arrivals and long-range structure
- **TFLA Triton kernels** for hardware-efficient mLSTM with BF16 mixed precision
- **Configurable scale**: 270K (supervised) to 20M (self-supervised) parameters

### Pretraining

**mLSTM-CR-large (Contrastive):**
- **8 datasets**: ETHZ, GEOFON, STEAD, INSTANCE, MLAAPDE, Iquique, PNW, OBST2024
- **Training**: 40 epochs, ~7 days on 4×H200, batch size 64 (global 1024)
- **Wav2Vec 2.0-style objective**: InfoNCE contrastive loss + Gumbel-Softmax VQ
- **Quantizer**: 2 groups × 320 codewords = 640 codes, ~510 perplexity (80% usage)

**mLSTM-Seq-large (Masked Reconstruction):**
- **2 datasets**: STEAD + MLAAPDE (~2M waveforms)
- **Training**: 45 epochs, ~7 days on 2×H200, batch size 128
- **Masking**: 75% masking ratio, span length 10

### Downstream Tasks

| Task | Description | Classes |
|------|-------------|---------|
| **Phase Picking** | P/S wave arrival detection | Regression / 3-class |
| **Foreshock Classification** | Temporal classification relative to mainshock | 2 or 9 classes |

---

## Architecture

```
                             WaveXLSTM Architecture
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   Input Waveform                                                    │
    │   [B, 4096, 3]                                                      │
    │        │                                                            │
    │        ▼                                                            │
    │   ┌─────────────┐                                                   │
    │   │ Conv Encoder│  Patch embedding (temporal convolutions)          │
    │   │   d=128     │                                                   │
    │   └──────┬──────┘                                                   │
    │          │                                                          │
    │          ▼                                                          │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │                    xLSTM U-Net Backbone                     │   │
    │   │                                                             │   │
    │   │  ┌───────────────┐                     ┌───────────────┐    │   │
    │   │  │ Encoder       │                     │ Decoder       │    │   │
    │   │  │               │                     │               │    │   │
    │   │  │ Stage 0       │─────────────────────│ Stage 0       │    │   │
    │   │  │ 4096 → 1024   │    Skip Connection  │ 1024 → 4096   │    │   │
    │   │  │ d=128         │                     │ d=128         │    │   │
    │   │  │ mLSTM ×3      │                     │ mLSTM ×3      │    │   │
    │   │  │               │                     │               │    │   │
    │   │  │ Stage 1       │─────────────────────│ Stage 1       │    │   │
    │   │  │ 1024 → 256    │    Skip Connection  │ 256 → 1024    │    │   │
    │   │  │ d=256         │                     │ d=256         │    │   │
    │   │  │ mLSTM ×3      │                     │ mLSTM ×3      │    │   │
    │   │  │               │                     │               │    │   │
    │   │  └───────┬───────┘                     └───────┬───────┘    │   │
    │   │          │                                     ▲            │   │
    │   │          ▼                                     │            │   │
    │   │     ┌─────────────────────────────────────────┐│            │   │
    │   │     │           Bottleneck (d=512)            ││            │   │
    │   │     │           256 tokens, mLSTM ×3          │┘            │   │
    │   │     └─────────────────────────────────────────┘             │   │
    │   │                                                             │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │          │                                                          │
    │          ▼                                                          │
    │   ┌─────────────┐                                                   │
    │   │  Task Head  │  (Classification / Regression / Contrastive)      │
    │   └──────┬──────┘                                                   │
    │          │                                                          │
    │          ▼                                                          │
    │      Output                                                         │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

**Model Configurations:**

| Config | d_model | Layers | Parameters | Use Case |
|--------|---------|--------|------------|----------|
| Small (sLSTM/mLSTM) | 12-64 | 5×3 | ~270K | Supervised phase picking |
| mLSTM-CR-large | 128 | 5×3 | **20.3M** | Contrastive pretraining |
| mLSTM-Seq-large | 176 | 24 | **12.3M** | Masked reconstruction |

*Layer notation: 5×3 = 5 stages × 3 blocks per stage = 15 total blocks*

---

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-username/wavexlstm.git
cd wavexlstm

# Create conda environment
conda create -n wavexlstm python=3.10
conda activate wavexlstm

# Install PyTorch (CUDA 11.8 example)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install pytorch-lightning hydra-core wandb seisbench
pip install einops triton
```

### 2. Install xLSTM with TFLA Kernels

```bash
# Install xLSTM
pip install xlstm

# Install TFLA Triton kernels for efficient mLSTM
pip install tfla-triton
```

### 3. Download SeisBench Datasets

```python
import seisbench.data as sbd

# Download pretraining datasets (run once)
datasets = ['ethz', 'geofon', 'stead', 'instance', 'iquique', 'pnw']
for name in datasets:
    data = sbd.WaveformDataset(name, download=True)
```

### 4. Run Experiments

**Pretraining (Contrastive Learning):**
```bash
python simple_train.py experiment=contrastive/xlstm_unet_seisbench \
    trainer.devices=4 \
    trainer.max_epochs=50
```

**Fine-tune on Phase Picking:**
```bash
python simple_train.py experiment=phase_picking/fine_tune_xlstm_ethz_seislm \
    model.checkpoint_path=/path/to/pretrained.ckpt \
    trainer.max_epochs=30
```

**Fine-tune on Foreshock Classification:**
```bash
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet \
    model.checkpoint_path=/path/to/pretrained.ckpt \
    trainer.max_epochs=15
```

---

## Project Structure

```
wavexlstm/
├── simple_train.py              # Main training script (PyTorch Lightning)
├── configs/
│   ├── experiment/              # Experiment configurations
│   │   ├── contrastive/         # Pretraining configs
│   │   ├── phase_picking/       # Phase picking fine-tuning
│   │   ├── fore_aftershock/     # Foreshock classification
│   │   └── xlstm_large/         # Large-scale foundation models
│   ├── model/                   # Model architecture configs
│   ├── dataset/                 # Dataset configurations
│   ├── optimizer/               # Optimizer configs (AdamW, etc.)
│   └── scheduler/               # LR scheduler configs
├── models/
│   ├── xlstm_unet.py            # xLSTM U-Net backbone (core architecture)
│   ├── contrastive_wrapper.py   # Wav2Vec 2.0-style contrastive wrapper
│   └── quantizers.py            # Gumbel-Softmax vector quantization
├── dataloaders/
│   ├── seisbench_loader.py      # SeisBench dataset loaders
│   └── foreshock_loader.py      # Foreshock-aftershock data pipeline
├── tasks/
│   ├── encoders/                # Input encoders (convolutional, etc.)
│   ├── decoders/                # Task-specific heads
│   └── losses/                  # Loss functions (InfoNCE, etc.)
├── evaluation/                  # Evaluation scripts and metrics
├── docs/                        # Technical documentation
└── notebooks/                   # Analysis notebooks
```

---

## Experiments

### Configuration Reference

| Task | Config Path | Description |
|------|-------------|-------------|
| **Pretraining** | | |
| Contrastive (Large) | `contrastive/xlstm_unet_seisbench_large` | 20.3M params, 8 datasets |
| Contrastive (Small) | `contrastive/xlstm_unet_seisbench` | ~270K params |
| **Phase Picking** | | |
| ETHZ | `phase_picking/fine_tune_xlstm_ethz_seislm` | P/S arrival detection |
| GEOFON | `phase_picking/fine_tune_xlstm_geofon_seislm` | P/S arrival detection |
| STEAD | `phase_picking/fine_tune_xlstm_stead_seislm` | P/S arrival detection |
| **Foreshock Classification** | | |
| xLSTM U-Net | `fore_aftershock/finetune_xlstm_unet` | 9-class temporal classification |
| Pure mLSTM variants | `fore_aftershock/finetune_pure_mlstm_*` | Multiple configurations |

### Training Commands

```bash
# Pretraining with multi-GPU
python simple_train.py experiment=contrastive/xlstm_unet_seisbench \
    trainer.devices=4 \
    trainer.strategy=ddp \
    data.batch_size=64

# Foreshock classification
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet \
    model.checkpoint_path=/path/to/pretrained.ckpt

# Debug run (fast iteration)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet \
    trainer.fast_dev_run=True
```

---

## Results

### Foreshock-Aftershock Classification (2016 Central Italy)

| Model | Accuracy | Parameters |
|-------|----------|------------|
| ConvNet baseline | 58.33% | - |
| SeisLM-base | 65.11% | 11.4M |
| SeisLM-large | 74.22% | 93.7M |
| mLSTM-Seq-large | 67.17% | 12.3M |
| **mLSTM-CR-large** | **76.96%** | **20.3M** |

*mLSTM-CR-large achieves SOTA (+2.74% over SeisLM-large) with 78% fewer parameters*

### Phase Picking (ETHZ Dataset)

| Model | Event AUC | Phase AUC | P-RMSE (s) | S-RMSE (s) |
|-------|-----------|-----------|------------|------------|
| PhaseNet | 0.990 | 0.998 | 0.297 | 0.467 |
| EQTransformer | 0.960 | 0.998 | 0.355 | 0.519 |
| sLSTM-small | 0.992 | 0.998 | **0.214** | 0.465 |
| mLSTM-small | **0.993** | **0.999** | 0.228 | **0.425** |

### Pretraining Metrics

| Model | Params | Datasets | Codebook Perplexity |
|-------|--------|----------|---------------------|
| mLSTM-CR-large | 20.3M | 8 | ~510 (80% usage) |
| mLSTM-Seq-large | 12.3M | 2 | N/A (reconstruction) |

---

## Citation

If you use WaveXLSTM in your research, please cite:

```bibtex
@mastersthesis{wavexlstm2025,
  title     = {WaveXLSTM: Extended LSTM for Self-Supervised Seismology},
  author    = {Alvani, Urfan},
  school    = {University of Basel},
  year      = {2025},
  note      = {Master's Thesis}
}
```

Please also cite the foundational works:

```bibtex
@article{beck2024xlstm,
  title   = {xLSTM: Extended Long Short-Term Memory},
  author  = {Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and others},
  journal = {arXiv preprint arXiv:2405.04517},
  year    = {2024}
}

@article{seislm2024,
  title   = {SeisLM: a Foundation Model for Seismic Waveforms},
  author  = {Münchmeyer, Jannes and others},
  journal = {arXiv preprint},
  year    = {2024}
}

@inproceedings{woollam2022seisbench,
  title     = {SeisBench: A Toolbox for Machine Learning in Seismology},
  author    = {Woollam, Jack and others},
  booktitle = {Seismological Research Letters},
  year      = {2022}
}
```

---

## Acknowledgments

This project builds on excellent open-source work:

- **[SeisBench](https://github.com/seisbench/seisbench)** - Seismological ML benchmark and data loaders
- **[xLSTM](https://github.com/NX-AI/xlstm)** - Extended LSTM implementation by NX-AI
- **[TFLA](https://github.com/NX-AI/tfla)** - Triton Fused Linear Attention kernels
- **[PyTorch Lightning](https://lightning.ai/)** - Training framework
- **[Hydra](https://hydra.cc/)** - Configuration management
- **[Weights & Biases](https://wandb.ai/)** - Experiment tracking

---

<div align="center">

**[Back to Top](#wavexlstm)**

</div>
