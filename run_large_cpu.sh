#!/usr/bin/env bash
set -euo pipefail
# Activate your Python environment before running. First argument: worker processes.
# An existing output directory is rejected to prevent overwriting a previous run.
workers="${1:-8}"
output_path="${2:-results/cpu-scale}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python experiment.py --config config_scale.json --workers "$workers" --out "$output_path"
python analyze.py --results "$output_path"
