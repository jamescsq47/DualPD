#!/usr/bin/env bash
# Keep node2800 available for follow-up experiments after the Qwen3-30B matrix.
#SBATCH --job-name=dualpd-hold-node2800
#SBATCH --partition=pi_songhan
#SBATCH --nodelist=node2800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/hold-node2800-after-qwen3-30b-%j.log

set -euo pipefail

echo "holding $(hostname) for follow-up experiments; pid=$$"
while true; do
  sleep 300
done
