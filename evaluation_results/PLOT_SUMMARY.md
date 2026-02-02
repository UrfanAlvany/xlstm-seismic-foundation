# Updated Phase Picking Results - Publication Quality Plots

**Generated:** October 14, 2025
**Status:** ✅ Complete - All models integrated successfully!

## 📊 What's New

Successfully integrated **SeisLM benchmark results** from the pickle file with our **mLSTM models** to create comprehensive comparison plots.

### Models Included

#### ETHZ Dataset (5 models):
1. **mLSTM_Sequential** (Ours) - Sequential pretraining
2. **mLSTM_CR** (Ours) - Contrastive + Random pretraining
3. **PhaseNet** (Baseline) - Published baseline model
4. **SeisLM_base** (Benchmark) - Transformer-based foundation model
5. **SeisLM_large** (Benchmark) - Larger transformer model

#### GEOFON Dataset (4 models):
1. **mLSTM_CR** (Ours) - Fine-tuned on GEOFON
2. **PhaseNet** (Baseline)
3. **SeisLM_base** (Benchmark)
4. **SeisLM_large** (Benchmark)

## 🎨 Beautiful Color Scheme

Each model has a distinctive visual style for easy identification:

- **mLSTM_Sequential**: Blue circle, solid line (━)
- **mLSTM_CR**: Green triangle, dash-dot line (━·━)
- **PhaseNet**: Orange square, dashed line (- - -)
- **SeisLM_base**: Purple diamond, dotted line (···)
- **SeisLM_large**: Red inverted triangle, solid line (━)

## 📁 Generated Files

### Main Comparison Plots
- `ethz_all_tasks_combined.pdf` (29 KB) - **Publication-ready PDF**
- `ethz_all_tasks_combined.png` (373 KB) - High-res PNG
- `geofon_all_tasks_combined.pdf` (27 KB) - **Publication-ready PDF**
- `geofon_all_tasks_combined.png` (346 KB) - High-res PNG

### Plot Structure
Each combined figure contains 4 subplots (2×2 grid):
1. **Event Detection** (AUC) - Top Left
2. **Phase Identification** (AUC) - Top Right
3. **P Onset Determination** (RMSE in seconds) - Bottom Left
4. **S Onset Determination** (RMSE in seconds) - Bottom Right

All plots show performance across 5 training fractions: 0.05, 0.10, 0.20, 0.50, 1.00

## 🔬 Data Sources

### Our Models (mLSTM)
- Loaded from evaluation CSV files in `seismic_data_modeling/wandb_logs/mars/`
- Real evaluation results on test sets

### Benchmark Models (PhaseNet, SeisLM)
- Loaded from `notebooks/all_datasets_results.pkl`
- Published results from SeisLM paper
- Real PhaseNet performance (not dummy data!)

## ✨ Key Features

- **Publication-quality**: 300 DPI, PDF + PNG formats
- **Consistent styling**: Follows SeisLM paper conventions
- **Clear legends**: All models clearly labeled
- **Proper formatting**: Embedded fonts (PDF fonttype 42)
- **Grid lines**: Enhanced readability
- **Bold titles**: Clear section identification

## 🚀 How to Regenerate

To regenerate the plots with updated data:

```bash
source ~/.bashrc
conda activate xlstm311
python run_updated_plots.py
```

## 📝 Notes

- PhaseNet results are now REAL data from the pickle file (not placeholder values)
- SeisLM models provide strong baselines for comparison
- All plots automatically adapt to available models
- Missing data points are handled gracefully (shown as gaps)

## 📍 Location

All files saved to:
```
/scicore/home/dokman0000/alvani0000/final_seismology/evaluation_results/
```

---

**Implementation Details:**
- Notebook updated: `notebooks/phasepick_seislm_style_plot.ipynb`
- Standalone script: `run_updated_plots.py`
- Helper module: `notebooks/load_seislm_pickle.py`
