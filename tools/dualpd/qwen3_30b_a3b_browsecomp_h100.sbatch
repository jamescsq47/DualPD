#!/usr/bin/env bash
#SBATCH --job-name=dualpd-qwen30ba3b
#SBATCH --partition=pi_songhan
#SBATCH --nodelist=node2801
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --time=1-12:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-30b-a3b-tp2-12-matrix-20261012/slurm-%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python \
  /home/shangy/dualpd/DualPD/tools/dualpd/qwen3_30b_a3b_browsecomp_h100_supervisor.py
