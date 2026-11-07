#!/usr/bin/env bash
#SBATCH --job-name=bc8b-p2d-paired-r10
#SBATCH --partition=pi_songhan
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-paired-credit-c576-20261028-r10/slurm-%j.out

set -euo pipefail
control=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-paired-credit-c576-20261028-r10
mkdir -p "$control"
cd /home/shangy/dualpd
export PYTHONUNBUFFERED=1
python_bin=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12
"$python_bin" -m pytest -q slime/tests/test_late_binding_router.py \
  -k 'paired_prefill_credit or parent_direct_shadow_credit'
exec "$python_bin" \
  slime/scripts/cluster/browsecomp_qwen3_8b_p2d_paired_credit_ablation.py
