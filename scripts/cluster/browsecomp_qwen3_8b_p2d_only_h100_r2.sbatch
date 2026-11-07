#!/usr/bin/env bash
#SBATCH --job-name=bc8b-p2d-r2
#SBATCH --partition=pi_songhan
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r2/slurm-%j.out

set -euo pipefail
control=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r2
mkdir -p "${control}"
cd /home/shangy/dualpd
export PYTHONUNBUFFERED=1
python_bin=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12
PYTHONPATH=/home/shangy/dualpd/sglang-src/pd/python:/home/shangy/dualpd/DualPD/examples/pd \
  "${python_bin}" -m pytest -q \
  slime/tests/test_late_binding_router.py \
  -k 'direct_admission_skips_d_with_full_transfer_window or p_ready_host_staging or p2d_host_metadata'
exec "${python_bin}" \
  slime/scripts/cluster/browsecomp_qwen3_8b_p2d_only_h100_r2.py
