#!/usr/bin/env bash
#SBATCH --job-name=bc8b-c512-nsys
#SBATCH --partition=pi_songhan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/nsys-browsecomp-qwen3-8b-h100-4p4d-c512-w300-m100-20261025/slurm-%j.out

set -euo pipefail
cd /home/shangy/dualpd
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python \
  DualPD/scripts/replay/cluster/browsecomp_qwen3_8b_c512_nsys.py
