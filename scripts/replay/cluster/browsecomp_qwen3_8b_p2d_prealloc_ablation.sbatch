#!/usr/bin/env bash
#SBATCH --job-name=bc8b-p2d-prealloc
#SBATCH --partition=pi_songhan
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-prealloc-c576-20261027/slurm-%j.out

set -euo pipefail
mkdir -p /orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-prealloc-c576-20261027
cd /home/shangy/dualpd
export PYTHONUNBUFFERED=1
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12 \
  DualPD/scripts/replay/cluster/browsecomp_qwen3_8b_p2d_prealloc_ablation.py
