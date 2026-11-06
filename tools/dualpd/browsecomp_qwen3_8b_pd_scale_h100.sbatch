#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-pd-scale
#SBATCH --partition=pi_songhan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=/home/shangy/dualpd/DualPD/examples/pd/runs-host/qwen3-8b-pd-scale-%j.log

set -euo pipefail

export PYTHONUNBUFFERED=1
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12 \
  /home/shangy/dualpd/DualPD/tools/dualpd/browsecomp_qwen3_8b_pd_scale_h100_matrix.py
