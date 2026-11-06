#!/usr/bin/env bash
#SBATCH --job-name=q3-8b-dram-total64-128
#SBATCH --partition=pi_songhan,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-global64-128-c512-20261028/slurm-%j.log
set -euo pipefail
export PYTHONUNBUFFERED=1
dram_control=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-global64-128-c512-20261028
dram_frozen=/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022/code/new_method
dram_python=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12
"${dram_python}" -c 'import json,sys; assert json.load(open(sys.argv[1])).get("verdict") == "GO", "Independent audit GO missing"' "${dram_control}/audit-go.json"
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
nvidia-smi topo -m
cd "${dram_frozen}"
PYTHONPATH="${dram_frozen}" "${dram_python}" -m pytest -q --disable-warnings --maxfail=1 \
  --junitxml="${dram_control}/frozen-lifecycle-tests.xml" \
  sglang/srt/disaggregation/test_agentic_kv_lifecycle.py \
  sglang/srt/disaggregation/test_agentic_host_transfer.py \
  sglang/srt/disaggregation/test_agentic_group_transfer.py \
  sglang/srt/disaggregation/test_agentic_tp_host_cancel_control.py \
  sglang/srt/disaggregation/test_agentic_tp_host_commit_order.py \
  sglang/srt/disaggregation/test_agentic_startup_prewarm.py
exec "${dram_python}" /home/shangy/dualpd/DualPD/tools/dualpd/browsecomp_qwen3_8b_dram_global64_128_h100.py
