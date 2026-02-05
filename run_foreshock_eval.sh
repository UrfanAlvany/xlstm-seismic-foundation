#!/usr/bin/env bash
# Evaluate one or more foreshock checkpoints with `evaluate_foreshock.py`.
#
# Usage:
#   export SEIS_DATA_DIR=/path/to/foreshock/data
#   bash run_foreshock_eval.sh /path/to/ckpt1.ckpt [/path/to/ckpt2.ckpt ...]
#
# Optional:
#   OUTPUT_DIR=./evaluation_results bash run_foreshock_eval.sh /path/to/ckpt.ckpt

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./evaluation_results}"
DATA_DIR="${SEIS_DATA_DIR:-}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Missing SEIS_DATA_DIR. Set it to your foreshock dataset directory."
  exit 1
fi

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash run_foreshock_eval.sh /path/to/ckpt1.ckpt [/path/to/ckpt2.ckpt ...]"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

for CKPT in "$@"; do
  if [[ ! -f "${CKPT}" ]]; then
    echo "Skipping missing checkpoint: ${CKPT}"
    continue
  fi

  echo "Evaluating: ${CKPT}"
  python evaluate_foreshock.py \\
    --ckpt "${CKPT}" \\
    --data_dir "${DATA_DIR}" \\
    --num_classes 9 \\
    --batch_size 32 \\
    --output_dir "${OUTPUT_DIR}"
done

echo "Done. Outputs in: ${OUTPUT_DIR}"
