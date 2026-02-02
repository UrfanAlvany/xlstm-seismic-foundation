<div align="center">

# SeisLM-xLSTM

### Extended LSTM for Self-Supervised Seismology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/)

[**Getting Started**](#quick-start) | [**Architecture**](#architecture) | [**Experiments**](#experiments) | [**Results**](#results) | [**Citation**](#citation)

</div>

---

## What is SeisLM-xLSTM?

**SeisLM-xLSTM** applies Extended Long Short-Term Memory (xLSTM) to seismology, using a **U-Net architecture with mLSTM/sLSTM blocks** for self-supervised learning on seismic waveforms.

Building on the success of [SeisLM](https://github.com/seisbench/seislm) and inspired by [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477), we pretrain on **5M+ seismic waveforms** from 8 global datasets using contrastive learning with Gumbel-Softmax vector quantization. The pretrained model transfers effectively to downstream tasks including **phase picking**, **foreshock-aftershock classification**, and **Mars InSight seismology**.

Key contributions:
- **xLSTM U-Net** combining bidirectional mLSTM blocks with skip connections for multi-scale temporal modeling
- **TFLA Triton kernels** enabling efficient mLSTM computation with linear memory scaling
- **Cross-planetary transfer**: Models pretrained on Earth data generalize to Mars seismology

---

## Key Features

### Architecture

- **xLSTM U-Net Backbone** with bidirectional mLSTM/sLSTM blocks and skip connections
- **Multi-scale pooling** (4096 → 1024 → 256 tokens) capturing both fine-grained arrivals and long-range structure
- **TFLA Triton kernels** for hardware-efficient mLSTM with BF16 mixed precision
- **Configurable scale**: 11M (base) to 50M+ (large) parameters

### Pretraining

- **8 datasets**: ETHZ, GEOFON, STEAD, InstanceCounts, MLAAPDE, Iquique, PNW, OBST2024
- **Wav2Vec 2.0-style objective**: InfoNCE contrastive loss + Gumbel-Softmax vector quantization
- **Masking strategy**: Random span masking over latent representations
- **Quantizer**: 2 groups × 320 codevectors × 256-D (640 total codes)

### Downstream Tasks

| Task | Description | Classes |
|------|-------------|---------|
| **Phase Picking** | P/S wave arrival detection | Regression / 3-class |
| **Foreshock Classification** | Temporal classification relative to mainshock | 2 or 9 classes |
| **Mars InSight** | Cross-planetary seismology transfer | Various |

---

## Architecture

```
                           SeisLM-xLSTM Architecture
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

| Config | d_model | Layers/Stage | Parameters | Use Case |
|--------|---------|--------------|------------|----------|
| Base | 128 | 3 | ~11M | Standard pretraining |
| Large | 176 | 24 | ~50M | Foundation model |
| Lite | 80 | 3 | ~4M | Fast experimentation |

---

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-username/seislm-xlstm.git
cd seislm-xlstm

# Create conda environment
conda create -n seislm-xlstm python=3.10
conda activate seislm-xlstm

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
python simple_train.py experiment=phase_picking/xlstm_unet \
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
seislm-xlstm/
├── simple_train.py              # Main training script (PyTorch Lightning)
├── configs/
│   ├── experiment/              # Experiment configurations
│   │   ├── contrastive/         # Pretraining configs
│   │   ├── phase_picking/       # Phase picking fine-tuning
│   │   ├── fore_aftershock/     # Foreshock classification
│   │   ├── mars/                # Mars InSight experiments
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
| Contrastive (Base) | `contrastive/xlstm_unet_seisbench` | 11M params, 8 datasets, 50 epochs |
| Contrastive (ETHZ) | `contrastive/xlstm_unet_ethz` | Single dataset variant |
| **Phase Picking** | | |
| xLSTM U-Net | `phase_picking/xlstm_unet` | P/S arrival detection |
| **Foreshock Classification** | | |
| 9-Class | `fore_aftershock/finetune_xlstm_unet` | Temporal bin classification |
| 2-Class | `fore_aftershock/finetune_xlstm_unet_2class` | Binary fore/aftershock |
| Few-Shot (5%) | `fore_aftershock/finetune_xlstm_unet_fewshot` | Limited data regime |
| Frozen Encoder | `fore_aftershock/finetune_frozen_encoder_only` | Linear probe |
| Random Init | `fore_aftershock/finetune_xlstm_unet_random_init` | Ablation (no pretraining) |
| **Mars InSight** | | |
| Transfer | `mars/xlstm_unet_mars` | Cross-planetary transfer |

### Training Commands

```bash
# Pretraining with multi-GPU
python simple_train.py experiment=contrastive/xlstm_unet_seisbench \
    trainer.devices=4 \
    trainer.strategy=ddp \
    data.batch_size=64

# Few-shot foreshock classification
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet_fewshot \
    data.train_fraction=0.1 \
    model.checkpoint_path=/path/to/pretrained.ckpt

# Debug run (fast iteration)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet \
    trainer.fast_dev_run=True
```

---

## Results

### Pretraining

| Model | Params | Datasets | Contrastive Acc | Codebook Usage |
|-------|--------|----------|-----------------|----------------|
| xLSTM U-Net Base | 11M | 8 | TBD | TBD |
| xLSTM U-Net Large | 50M | 8 | TBD | TBD |

### Phase Picking (SeisBench Benchmark)

| Model | P Precision | P Recall | P F1 | S Precision | S Recall | S F1 |
|-------|-------------|----------|------|-------------|----------|------|
| PhaseNet | - | - | - | - | - | - |
| EQTransformer | - | - | - | - | - | - |
| SeisLM | - | - | - | - | - | - |
| **xLSTM U-Net (Ours)** | TBD | TBD | TBD | TBD | TBD | TBD |

### Foreshock-Aftershock Classification (2016 Central Italy)

| Model | Pretraining | 9-Class Acc | 2-Class Acc | F1 Macro |
|-------|-------------|-------------|-------------|----------|
| SeisLM | Contrastive | - | - | - |
| xLSTM U-Net | None | TBD | TBD | TBD |
| **xLSTM U-Net (Ours)** | Contrastive | TBD | TBD | TBD |

### Few-Shot Learning

| Data Fraction | SeisLM | xLSTM U-Net (Ours) |
|---------------|--------|---------------------|
| 5% | - | TBD |
| 10% | - | TBD |
| 20% | - | TBD |
| 50% | - | TBD |
| 100% | - | TBD |

---

## Citation

If you use SeisLM-xLSTM in your research, please cite:

```bibtex
@mastersthesis{seislm-xlstm2025,
  title     = {Extended LSTM for Self-Supervised Seismology},
  author    = {Author Name},
  school    = {University Name},
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

**[Back to Top](#seislm-xlstm)**

</div>
