# 🌋 mLSTM-UNet Foreshock-Aftershock Fine-tuning Guide

## 📚 Table of Contents
1. [Quick Start](#quick-start)
2. [What is Foreshock Classification?](#what-is-foreshock-classification)
3. [Configuration Files](#configuration-files)
4. [Training Commands](#training-commands)
5. [SeisLM-Style Experiments](#seislm-style-experiments)
6. [Hyperparameter Guide](#hyperparameter-guide)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites
1. **Pretrained checkpoint**: Contrastive pretrained mLSTM-UNet
   - Example: `wandb_logs/mars/2025-07-07__17_36_30/checkpoints/best.ckpt`
2. **Dataset**: Foreshock-aftershock NRCA data
   - Location: `dataloaders/data/foreshock_aftershock_NRCA/`
   - Files: `dataframe_pre_NRCA.csv`, `dataframe_post_NRCA.csv`, `dataframe_visso_NRCA.csv`

### Basic Training Command
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  model.pretrained=wandb_logs/mars/2025-07-07__17_36_30
```

---

## 🎯 What is Foreshock Classification?

### Scientific Background
**Foreshock-aftershock classification** aims to distinguish seismic events based on their **temporal relationship** to a main earthquake. This task is scientifically important for understanding:
- Stress evolution in the Earth's crust
- Precursory patterns before large earthquakes
- Post-seismic relaxation processes

### The 2016 Central Italy Sequence
Our dataset focuses on the **2016 Norcia earthquake** (M6.5):
- **Main event**: October 30, 2016, 07:40:17 UTC
- **Visso foreshock**: October 26, 2016 (M5.9) - 4 days before main shock
- **Network**: NRCA (Near-Realtime Central Apennines)

### Classification Schemes

#### 2-Class Setup
- **Class 0**: Pre-shock events (before main shock)
- **Class 1**: Post-shock events (after main shock)

#### 9-Class Setup (Most Challenging)
Events are binned by **Time-To-Failure (TTF)** relative to main shock:
- **Classes 0-3**: 4 temporal bins of pre-shock events (earliest → latest)
- **Class 4**: Visso main foreshock (M5.9)
- **Classes 5-8**: 4 temporal bins of post-shock events (earliest → latest)

**Why this matters**: The model must learn subtle temporal patterns in seismic waveforms that correlate with the earthquake cycle, NOT just event magnitude or location.

### Key Dataset Features
1. **Temporal event-based splitting**: Train/val/test use different earthquakes (prevents data leakage)
2. **Balanced classes**: Equal number of waveforms per class
3. **Edge-middle split**: Training uses early/late events, validation uses middle events (forces temporal interpolation)

---

## 📁 Configuration Files

### Created Configs

| Config File | Purpose | Num Classes | Use Case |
|------------|---------|-------------|----------|
| `finetune_xlstm_unet.yaml` | **Main config** | 9 | Full fine-tuning, baseline |
| `finetune_xlstm_unet_2class.yaml` | Binary classification | 2 | Simpler task, faster experiments |
| `finetune_xlstm_unet_fewshot.yaml` | Few-shot learning | 9 | Sample efficiency testing |
| `finetune_xlstm_unet_random_init.yaml` | No pretraining | 9 | Transfer learning ablation |

### Config Structure (Hydra)
```
configs/
├── experiment/fore_aftershock/
│   ├── finetune_xlstm_unet.yaml          # Main config
│   └── ...
├── model/xlstm_unet.yaml                 # Model architecture
├── dataset/foreshock_aftershock.yaml     # Dataset config
├── task/classification.yaml              # Task definition
└── optimizer/adamw_hydra.yaml            # Optimizer config
```

---

## 🎮 Training Commands

### 1. Baseline: Full Fine-tuning (9-class)
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  model.pretrained=<YOUR_CHECKPOINT_PATH>
```

**Expected**:
- Training: ~176 batches/epoch (with batch_size=16)
- Time: ~5-10 min/epoch (depending on GPU)
- Convergence: 10-15 epochs

---

### 2. Binary Classification (2-class)
Simpler task for quick validation:
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet_2class \
  model.pretrained=<YOUR_CHECKPOINT_PATH>
```

**Expected**:
- Higher accuracy than 9-class
- Faster convergence
- Good for sanity checking

---

### 3. Frozen Encoder (Train Only Decoder)
SeisLM-style ablation:
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  encoder.freeze=True \
  model.freeze=True
```

**Purpose**: Measure quality of pretrained representations

---

### 4. Progressive Unfreezing
Freeze initially, unfreeze at epoch 5:
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  train.unfreeze_at_epoch=5 \
  train.unfrozen_lr_mult=0.1
```

**Best practice** from seisLM to prevent overfitting

---

### 5. Few-Shot Learning (5% Data)
Test sample efficiency:
```bash
# 5% of training data
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet_fewshot \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  dataset.train_frac=0.035

# 10% of training data
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet_fewshot \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  dataset.train_frac=0.07

# 20% of training data
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet_fewshot \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  dataset.train_frac=0.14

# 50% of training data
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet_fewshot \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  dataset.train_frac=0.35
```

**SeisLM comparison**: They test 5%, 10%, 20%, 50%, 100% fractions

---

### 6. Random Initialization (No Pretraining)
Ablation to measure transfer learning benefit:
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet_random_init
```

**Expected**: Lower accuracy and slower convergence than pretrained

---

### 7. Multi-GPU Training
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  trainer.devices=4 \
  trainer.strategy=ddp
```

---

### 8. Quick Debug Run
Fast sanity check:
```bash
python simple_train.py \
  experiment=fore_aftershock/finetune_xlstm_unet \
  model.pretrained=<YOUR_CHECKPOINT_PATH> \
  trainer.limit_train_batches=0.1 \
  trainer.limit_val_batches=0.1 \
  trainer.max_epochs=2
```

---

## 🔬 SeisLM-Style Experiments

### Complete Ablation Suite

#### Experiment Set 1: Architecture Ablations
```bash
# 1. Full fine-tuning (baseline)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT>

# 2. Frozen encoder
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT> encoder.freeze=True model.freeze=True

# 3. Progressive unfreezing
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT> train.unfreeze_at_epoch=5

# 4. Random init (no pretraining)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet_random_init
```

#### Experiment Set 2: Few-Shot Learning
```bash
# Train on 5%, 10%, 20%, 50%, 100% data
for frac in 0.035 0.07 0.14 0.35 0.7; do
  python simple_train.py \
    experiment=fore_aftershock/finetune_xlstm_unet_fewshot \
    model.pretrained=<CKPT> \
    dataset.train_frac=$frac \
    experiment_name=XLSTM_UNET_foreshock_frac_${frac}
done
```

#### Experiment Set 3: Num Classes Comparison
```bash
# 2-class (binary)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet_2class model.pretrained=<CKPT>

# 4-class
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT> \
  dataset.num_classes=4 decoder.num_classes=4

# 8-class
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT> \
  dataset.num_classes=8 decoder.num_classes=8

# 9-class (default)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT>
```

---

## ⚙️ Hyperparameter Guide

### SeisLM-Aligned Hyperparameters

| Parameter | SeisLM Value | Our Config | Notes |
|-----------|--------------|------------|-------|
| **Learning Rate** | 4e-4 | 4e-4 | For Wav2Vec2 fine-tuning |
| **Weight Decay** | 0.1 | 0.1 | L2 regularization |
| **Batch Size** | 16 | 16 | Fits in 24GB GPU |
| **Max Epochs** | 15 | 15 | For fine-tuning |
| **Warmup** | 0% | 0% | No warmup for fine-tuning |
| **LR Schedule** | Cosine | Cosine | With annealing |
| **Gradient Clip** | 1.0 | 1.0 | Stabilize training |
| **Decoder Dropout** | 0.2 | 0.2 | Head regularization |
| **Label Smoothing** | 0.0 | 0.0 | (can enable with 0.1) |

### When to Adjust Hyperparameters

**Increase LR** (e.g., 1e-3) when:
- Training from random init
- Small dataset / few-shot
- Model not converging

**Increase Dropout** (e.g., 0.3) when:
- Overfitting (train acc >> val acc)
- Few-shot learning
- Small dataset

**Increase Epochs** (e.g., 50) when:
- Training from scratch
- Few-shot learning
- Model still improving at epoch 15

**Add Label Smoothing** (e.g., 0.1) when:
- Model too confident (high softmax scores)
- Overfitting on training set

---

## 📊 Expected Results

### Performance Baselines

#### seisLM Results (Wav2Vec2, from paper/repo):
- **2-class**: ~75-85% accuracy
- **9-class**: ~40-50% accuracy (chance = 11.1%)
- **Few-shot (5%)**: ~30-35% accuracy

#### Your mLSTM-UNet Should:
- **Match or exceed** seisLM on 9-class (>50%)
- **Outperform** random init by 10-15%
- **Show sample efficiency** in few-shot (>35% with 5% data)

### Confusion Matrix Insights
For 9-class, expect:
- **High accuracy** on classes 0-3 (early pre-shocks) and 5-8 (aftershocks)
- **Lower accuracy** on class 4 (Visso) - harder to distinguish
- **Off-diagonal errors** mostly between adjacent temporal bins

### Training Curves
- **Loss**: Should decrease smoothly, converge by epoch 10-12
- **Accuracy**: Train ~60-70%, Val ~50-55% (9-class)
- **Val loss**: Should not increase after epoch 5 (if overfitting, add dropout)

---

## 🛠️ Troubleshooting

### Issue: "Pretrained checkpoint not found"
**Solution**:
```bash
# Check checkpoint path
ls -lh <YOUR_CHECKPOINT_PATH>

# If using wandb_logs, ensure path ends with /checkpoints/best.ckpt or similar
model.pretrained=wandb_logs/mars/2025-07-07__17_36_30/checkpoints/best.ckpt
```

### Issue: "Dataset not found"
**Solution**:
```bash
# Check dataset location
ls -lh dataloaders/data/foreshock_aftershock_NRCA/

# Should contain:
# - dataframe_pre_NRCA.csv
# - dataframe_post_NRCA.csv
# - dataframe_visso_NRCA.csv
```

### Issue: OOM (Out of Memory)
**Solutions**:
1. Reduce batch size:
   ```bash
   dataset.batch_size=8 loader.batch_size=8
   ```
2. Enable gradient checkpointing:
   ```bash
   model.gradient_checkpointing=True
   ```
3. Use smaller model:
   ```bash
   model.d_model=64 model.n_layers=2
   ```

### Issue: Model not converging
**Solutions**:
1. Check data normalization matches pretraining:
   ```bash
   dataset.amp_norm_type='std'  # or 'peak'
   ```
2. Increase learning rate:
   ```bash
   optimizer.lr=1e-3
   ```
3. Add warmup:
   ```bash
   scheduler.warmup_fraction=0.1
   ```

### Issue: Overfitting (train acc >> val acc)
**Solutions**:
1. Increase dropout:
   ```bash
   decoder.dropout=0.3 model.dropout=0.2
   ```
2. Add label smoothing:
   ```bash
   train.label_smoothing=0.1
   ```
3. Freeze encoder:
   ```bash
   encoder.freeze=True model.freeze=True
   ```
4. Reduce training data:
   ```bash
   dataset.train_frac=0.5  # Use 50% of data
   ```

---

## 📈 Monitoring & Evaluation

### WandB Logging
Experiments are automatically logged to WandB with tags:
- `num_classes_9`
- `train_frac_1.0`
- `model_xlstm_unet`

### Key Metrics to Track
1. **Accuracy**: Primary metric (balanced classes)
2. **Loss**: Should decrease smoothly
3. **Per-class F1**: Identify which classes are hardest
4. **Confusion Matrix**: Understand error patterns

### Evaluation Commands
```bash
# Evaluate best checkpoint
python pick_evaluation_script.py \
  --checkpoint outputs/<EXPERIMENT_DIR>/checkpoints/best.ckpt \
  --dataset foreshock_aftershock

# Generate confusion matrix
python -c "
import seaborn as sns
import matplotlib.pyplot as plt
# Load predictions and targets
sns.heatmap(confusion_matrix, annot=True)
plt.savefig('confusion_matrix.png')
"
```

---

## 🎓 Scientific Impact

If mLSTM-UNet outperforms seisLM:
1. **First xLSTM application** to foreshock-aftershock classification
2. **Evidence** that long-range dependencies matter for stress evolution
3. **Transfer learning validation** across seismic tasks
4. **Sample efficiency** for practical deployment

### Potential Paper Sections
- **Ablation Study**: Pretrained vs random init
- **Few-Shot Analysis**: Sample efficiency curves
- **Architecture Comparison**: xLSTM vs Transformer (seisLM)
- **Temporal Pattern Visualization**: Attention/CAM on waveforms

---

## 📝 Quick Reference Card

```bash
# Baseline (9-class, full data, pretrained)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT>

# Binary (2-class)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet_2class model.pretrained=<CKPT>

# Few-shot (5% data)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet_fewshot model.pretrained=<CKPT> dataset.train_frac=0.035

# Random init (no pretraining)
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet_random_init

# Frozen encoder
python simple_train.py experiment=fore_aftershock/finetune_xlstm_unet model.pretrained=<CKPT> encoder.freeze=True model.freeze=True
```

---

## 🤝 Comparison to seisLM

| Aspect | seisLM | Your mLSTM-UNet |
|--------|--------|-----------------|
| **Backbone** | Wav2Vec2 (Transformer) | xLSTM U-Net |
| **Pretraining** | Contrastive + VQ | Contrastive + VQ ✅ |
| **Decoder** | DoubleConvBlock | SequenceClassifier (double-conv) ✅ |
| **LR** | 4e-4 | 4e-4 ✅ |
| **Epochs** | 15 | 15 ✅ |
| **Batch Size** | 16 | 16 ✅ |
| **Data Split** | Temporal event-based | Temporal event-based ✅ |

**Advantage**: Your Hydra config system enables easier hyperparameter sweeps!

---

## 🚀 Next Steps

1. **Update checkpoint path** in configs
2. **Run baseline experiment** (9-class, full data)
3. **Compare to random init** (measure transfer learning benefit)
4. **Run few-shot experiments** (5%, 10%, 20%, 50%)
5. **Analyze results** (confusion matrices, per-class F1)
6. **Compare to seisLM** (if results available)
7. **Write paper** 📝

Good luck! 🌟
