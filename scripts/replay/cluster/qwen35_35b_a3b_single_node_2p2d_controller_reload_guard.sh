#!/usr/bin/env bash
set -euo pipefail

control=/orcd/scratch/orcd/003/shangy/dualpd/runs/qwen35-35b-a3b-tp2-single-node-2p2d-c64c128c192-tuned-20261016
old_controller_pid=761498
python=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12
controller=/home/shangy/dualpd/DualPD/scripts/replay/cluster/qwen35_35b_a3b_single_node_2p2d_matrix_autopilot.py
old_run=q35a3b-2p2d-a-ft30-l4l3-swe500-c256-20261016-r3

while true; do
  read -r status index current progress < <(
    "$python" - "$control/state.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1]))
print(
    state.get("status", ""),
    state.get("index", -1),
    state.get("current_run_id", ""),
    state.get("progress", 0),
)
PY
  )
  if [[ "$status" == running \
        && "$index" -ge 10 \
        && "$current" != "$old_run" \
        && "$progress" -ge 1 ]] \
      && ! pgrep -P "$old_controller_pid" >/dev/null 2>&1; then
    kill -TERM "$old_controller_pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$old_controller_pid" 2>/dev/null || break
      sleep 1
    done
    exec "$python" "$controller" --interval-seconds 300 \
      >>"$control/controller.log" 2>&1
  fi
  sleep 30
done
