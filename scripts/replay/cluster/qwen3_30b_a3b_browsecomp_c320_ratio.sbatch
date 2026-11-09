#!/usr/bin/env bash
#SBATCH --job-name=dualpd-qwen30-ratio
#SBATCH --partition=pi_songhan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --time=08:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/c320-ratio-%x-%j.out

set -euo pipefail

: "${PD_RATIO:?submit with PD_RATIO=2p2d or PD_RATIO=3p1d}"
export PYTHONUNBUFFERED=1
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python \
  /home/shangy/dualpd/DualPD/scripts/replay/cluster/qwen3_30b_a3b_browsecomp_c320_ratio_supervisor.py
