#!/usr/bin/env bash
#SBATCH --job-name=dualpd-qwen30ba3b-extra
#SBATCH --partition=pi_songhan
#SBATCH --exclude=node1800,node2700,node2701
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --time=1-00:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-30b-a3b-tp2-c128-c384-20261013/slurm-%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python \
  /home/shangy/dualpd/DualPD/tools/dualpd/qwen3_30b_a3b_browsecomp_h100_continuation_supervisor.py
