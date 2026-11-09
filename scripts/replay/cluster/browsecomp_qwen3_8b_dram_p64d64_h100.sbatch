#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-dram-p64d64
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-p64d64-c512-20261022/slurm-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
audit_ready=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-p64d64-c512-20261022/audit-go.json
if [[ ! -f "${audit_ready}" ]]; then
  echo "Independent audit GO missing; refusing to launch"
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
nvidia-smi topo -m
lscpu
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12 \
  /home/shangy/dualpd/DualPD/scripts/replay/cluster/browsecomp_qwen3_8b_dram_p64d64_h100.py
