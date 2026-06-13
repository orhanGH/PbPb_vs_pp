#!/bin/bash
#SBATCH --job-name=pbpb_pp_cpu_check
#SBATCH --output=outputs/logs/pbpb_pp_cpu_check_%j.out
#SBATCH --error=outputs/logs/pbpb_pp_cpu_check_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

DATA_ROOT="/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50"

mkdir -p outputs/logs
mkdir -p outputs/splits

echo "============================================================"
echo "PbPb_vs_pp CPU check"
echo "============================================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Data root: ${DATA_ROOT}"
echo "Python: $(which python3)"
python3 --version
echo "============================================================"

echo "Inspecting dataset..."
python3 scripts/inspect_dataset.py \
  --data-root "${DATA_ROOT}" \
  --n 1

echo "Creating file-level splits..."
python3 scripts/make_splits.py \
  --data-root "${DATA_ROOT}" \
  --output-dir outputs/splits \
  --seed 42 \
  --test-size 0.20 \
  --n-folds 4

echo "Checking observable dataset..."
python3 main.py --mode check_obsv

echo "Checking particle dataset..."
python3 main.py --mode check_parts

echo "Done."
