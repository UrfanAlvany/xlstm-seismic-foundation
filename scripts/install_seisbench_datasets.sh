#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash install_seisbench_datasets.sh Iquique PNW OBST2024
#
# Ensures SEISBENCH_DATA is set (defaults to ~/seis_data) and downloads the
# specified SeisBench datasets into that base directory.

: "${SEISBENCH_DATA:=$HOME/seis_data}"
export SEISBENCH_DATA
mkdir -p "$SEISBENCH_DATA"
echo "[info] SEISBENCH_DATA=$SEISBENCH_DATA"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DatasetName> [<DatasetName> ...]" >&2
  echo "Example: $0 Iquique PNW OBST2024" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "$SCRIPT_DIR/install_seisbench_datasets.py" "$@"

