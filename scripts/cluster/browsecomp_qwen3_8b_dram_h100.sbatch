#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-dram-c512
#SBATCH --partition=pi_songhan
#SBATCH --nodelist=node2700
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/slurm-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
audit_ready=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/audit-go-pss-isolation.json
while [[ ! -f "${audit_ready}" ]]; do
  if (( SECONDS > 900 )); then
    echo "Independent audit readiness deadline exceeded; releasing allocation"
    exit 1
  fi
  sleep 5
done
exec /orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12 \
  /home/shangy/dualpd/DualPD/scripts/cluster/browsecomp_qwen3_8b_dram_h100_matrix.py
