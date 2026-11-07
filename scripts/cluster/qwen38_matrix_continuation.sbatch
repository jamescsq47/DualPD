#!/usr/bin/env bash
# Resume the Qwen3.8 matrix on a successor two-node allocation.  Resource
# requests are supplied by sbatch so this file only owns the controller.
set -euo pipefail

controller=/home/shangy/dualpd/DualPD/scripts/cluster/qwen38_matrix_autopilot.py
python=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12
control_dir=/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen38-27b-tp2-2p6d-matrix-autopilot
lock="$control_dir/controller.lock"
log="$control_dir/controller.log"

# The old login-node controller can hold the shared lock for at most one poll
# interval after its allocation ends.  Wait for that exact lock instead of
# racing or killing a process by name.
while ! flock -n "$lock" -c true; do
  sleep 60
done

export DUALPD_MATRIX_JOB_ID="$SLURM_JOB_ID"
mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ "${#allocated_nodes[@]}" -ne 2 ]]; then
  echo "expected exactly two allocated nodes, got ${#allocated_nodes[@]}" >&2
  exit 2
fi
export DUALPD_MATRIX_NODE0="${allocated_nodes[0]}"
export DUALPD_MATRIX_NODE1="${allocated_nodes[1]}"
exec "$python" -u "$controller" --interval-seconds 1800 >>"$log" 2>&1
