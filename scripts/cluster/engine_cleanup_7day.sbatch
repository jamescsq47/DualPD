#!/usr/bin/env bash
#SBATCH --job-name=dualpd-cleanup-7d
#SBATCH --partition=pi_songhan
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail
# Slurm spools the batch script, so resolve the maintained checkout explicitly.
project_root="${DUALPD_ROOT:-/home/shangy/dualpd/DualPD}"
state_dir="${DUALPD_VALIDATION_ROOT:?set DUALPD_VALIDATION_ROOT to the isolated run root}"
exec "${project_root}/.venv/bin/python" \
  "${project_root}/scripts/cluster/engine_cleanup_allocation.py" \
  --state-dir "${state_dir}"
