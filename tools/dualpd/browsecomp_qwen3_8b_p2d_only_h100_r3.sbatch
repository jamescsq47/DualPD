#!/usr/bin/env bash
#SBATCH --job-name=dualpd-h100-7d
#SBATCH --partition=pi_songhan
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=7-00:00:00
#SBATCH --output=/orcd/pool/004/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r3/slurm-%j.out

set -uo pipefail
control=/orcd/pool/004/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r3
mkdir -p "${control}"
cd /home/shangy/dualpd
export PYTHONUNBUFFERED=1
python_bin=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12

PYTHONPATH=/home/shangy/dualpd/sglang-src/pd/python:/home/shangy/dualpd/DualPD/examples/pd \
  "${python_bin}" -m pytest -q \
  slime/examples/pd/tests/test_late_binding_router.py \
  -k 'direct_admission_skips_d_with_full_transfer_window or p_ready_host_staging or p2d_host_metadata'
test_rc=$?

controller_rc=125
if [[ ${test_rc} -eq 0 ]]; then
  "${python_bin}" slime/tools/dualpd/browsecomp_qwen3_8b_p2d_only_h100_r3.py
  controller_rc=$?
fi

"${python_bin}" - "${control}/allocation-holder.json" "${test_rc}" "${controller_rc}" <<'PY'
import json
import os
from pathlib import Path
import socket
import sys
import time

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "epoch": time.time(),
    "hostname": socket.gethostname(),
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "test_rc": int(sys.argv[2]),
    "controller_rc": int(sys.argv[3]),
    "status": "holding_allocation",
}, indent=2) + "\n")
temporary.replace(path)
PY

# The user requested one reusable seven-day node.  Keep the clean allocation
# alive after this case completes or pauses; watchdog-driven repairs and later
# experiments are launched into this same Slurm job.
while true; do
  sleep 300
done
