#!/bin/bash
# Quick evaluation script for foreshock checkpoints
# Usage: bash run_foreshock_eval.sh

set -eo pipefail

# Activate environment
source ~/.bashrc
conda activate xlstm_official_240 || conda activate xlstm311 || exit 1

cd /scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling

# Paths
CKPT_EPOCH4="/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/wandb_logs/mars/2025-10-19__19_08_46__j59201230/checkpoints/callback-epoch=4-step=874.ckpt"
CKPT_EPOCH14="/scicore/home/dokman0000/alvani0000/final_seismology/seismic_data_modeling/wandb_logs/mars/2025-10-19__19_08_46__j59201230/checkpoints/epoch=14-step=2537.ckpt"
DATA_DIR="/scicore/home/dokman0000/alvani0000/seis_data"
OUTPUT_DIR="/scicore/home/dokman0000/alvani0000/final_seismology/evaluation_results"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Foreshock-Aftershock Evaluation (SeisLM-style)"
echo "============================================"
echo ""

# Evaluate epoch 4 checkpoint
if [ -f "$CKPT_EPOCH4" ]; then
    echo "🔬 Evaluating Epoch 4 checkpoint..."
    echo "Checkpoint: $CKPT_EPOCH4"
    echo ""
    python evaluate_foreshock.py \
        --ckpt "$CKPT_EPOCH4" \
        --data_dir "$DATA_DIR" \
        --num_classes 9 \
        --batch_size 32 \
        --output_dir "$OUTPUT_DIR"

    echo ""
    echo "✅ Epoch 4 evaluation complete!"
    echo ""
else
    echo "⚠️  Epoch 4 checkpoint not found: $CKPT_EPOCH4"
    echo ""
fi

# Evaluate epoch 14 checkpoint
if [ -f "$CKPT_EPOCH14" ]; then
    echo "🔬 Evaluating Epoch 14 checkpoint..."
    echo "Checkpoint: $CKPT_EPOCH14"
    echo ""
    python evaluate_foreshock.py \
        --ckpt "$CKPT_EPOCH14" \
        --data_dir "$DATA_DIR" \
        --num_classes 9 \
        --batch_size 32 \
        --output_dir "$OUTPUT_DIR"

    echo ""
    echo "✅ Epoch 14 evaluation complete!"
    echo ""
else
    echo "⚠️  Epoch 14 checkpoint not found: $CKPT_EPOCH14"
    echo ""
fi

echo "============================================"
echo "✅ All evaluations complete!"
echo "============================================"
echo ""
echo "📊 Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -lh "$OUTPUT_DIR"/foreshock_xlstm_* 2>/dev/null || echo "  (No files found yet)"
echo ""
echo "Next steps:"
echo "  1. Review confusion matrices: $OUTPUT_DIR/foreshock_xlstm_confusion_*.png"
echo "  2. Check JSON results: $OUTPUT_DIR/foreshock_xlstm_results_*.json"
echo "  3. Compare with SeisLM baseline"
echo ""
