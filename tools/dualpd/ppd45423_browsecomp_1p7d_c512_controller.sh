#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shangy/dualpd
TOOLS=${ROOT}/slime/tools/dualpd
PYTHON=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python
BASE=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423
CONTROL=${BASE}/aligned-1p7d-c512
RUNS=${BASE}/runs
RUN_ID=ppd45423-browsecomp-local-1p7d-c512-w300-m1200-r1
RUN_DIR=${RUNS}/${RUN_ID}
PORTS=(28900 29000 29100 29101 29102 29103 29104 29105 29106 \
  29200 29300 29301 29302 29303 29304 29305 29306 \
  29500 29600 29700 29701 29702 29703 29704 29705 29706)

mkdir -p "${CONTROL}"
printf '%s\n' "$$" >"${CONTROL}/controller.pid"

write_state() {
  local status="$1" detail="${2:-}"
  STATUS_VALUE="${status}" DETAIL_VALUE="${detail}" CONTROL_VALUE="${CONTROL}" \
    RUN_ID_VALUE="${RUN_ID}" "${PYTHON}" - <<'PY'
import json, os, time
from pathlib import Path
p = Path(os.environ["CONTROL_VALUE"]) / "state.json"
value = {
    "epoch": time.time(), "status": os.environ["STATUS_VALUE"],
    "detail": os.environ["DETAIL_VALUE"], "concurrency": 512,
    "layout": "1p7d", "run_id": os.environ["RUN_ID_VALUE"],
}
t = p.with_suffix(".tmp")
t.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
t.replace(p)
PY
}

hard_gate() {
  local port
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | grep -q .; then return 1; fi
  for port in "${PORTS[@]}"; do
    if ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q .; then return 1; fi
  done
  if pgrep -af 'apptainer|squashfuse|fuse-overlayfs' | grep -v 'pgrep -af' | grep -q .; then return 1; fi
}

if ! hard_gate; then
  write_state failed "pre-case hard cleanup gate failed"
  exit 1
fi

write_state running "starting ${RUN_ID}"
if ! env LAYOUT=1p7d PAIR_COUNT=1 CONCURRENCY=512 \
    WARMUP_SECONDS=300 MEASURE_SECONDS=1200 TURN_CAP=2048 \
    "${TOOLS}/ppd45423_browsecomp_smoke.sh" local "${RUN_ID}" \
    >"${CONTROL}/case.log" 2>&1; then
  write_state failed "1p7d c512 case launcher failed"
  exit 1
fi

if ! hard_gate; then
  write_state failed "post-case hard cleanup gate failed"
  exit 1
fi

"${PYTHON}" "${ROOT}/slime/examples/pd/scripts/tools/summarize_agentic_steady.py" \
  "${RUN_DIR}" >"${RUN_DIR}/summarize.log"
write_state complete "c512 1p7d complete and clean"
