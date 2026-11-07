#!/usr/bin/env bash
# Allocate two whole H100 nodes and keep them for qwen35_slurm.sh.  The model
# groups themselves are launched as explicit overlapping Slurm steps; this
# batch shell owns no GPU process.
#SBATCH --job-name=dualpd-qwen35-2n
#SBATCH --partition=pi_songhan
#SBATCH --account=mit_general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=224
#SBATCH --gres=gpu:h100:8
# Qwen3.5-122B HiCache/Mooncake needs room for the 512 GiB Mooncake segment,
# per-rank Attention/Mamba host pools, model/runtime state, and the verifier.
# 1000G previously hit the Slurm cgroup limit and left the extern step in OOM
# cleanup; reserve the validated large-memory-node budget by default.
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/shangy/dualpd/slurm-qwen35-%j.out

set -euo pipefail
sleep 604800
