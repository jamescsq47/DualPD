#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shangy/dualpd
TOOLS=${ROOT}/DualPD/scripts/replay/cluster
PYTHON=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python
BASE=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423
CONTROL=${BASE}/aligned-2p6d-c576
RUNS=${BASE}/runs
PRIOR_RUN=${RUNS}/ppd45423-browsecomp-local-4p4d-c512-w300-m1200-r1
RUN_ID=ppd45423-browsecomp-local-2p6d-c576-w300-m1200-r3
RUN_DIR=${RUNS}/${RUN_ID}
PORTS=(28900 29000 29100 29101 29102 29103 29104 29105 \
  29200 29201 29300 29301 29302 29303 29304 29305 \
  29500 29501 29600 29601 29700 29701 29702 29703 29704 29705)

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
    "detail": os.environ["DETAIL_VALUE"], "concurrency": 576,
    "layout": "2p6d", "run_id": os.environ["RUN_ID_VALUE"],
}
t = p.with_suffix(".tmp")
t.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
t.replace(p)
PY
}

hard_gate() {
  local port
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | grep -q .; then
    return 1
  fi
  for port in "${PORTS[@]}"; do
    if ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | grep -q .; then
      return 1
    fi
  done
  if pgrep -af 'apptainer|squashfuse|fuse-overlayfs' | grep -v 'pgrep -af' | grep -q .; then
    return 1
  fi
}

if [[ "${SKIP_PRIOR_WAIT:-0}" != "1" ]]; then
  write_state waiting "waiting for c512 4p4d to complete and clean"
  deadline=$((SECONDS + 7200))
  while (( SECONDS < deadline )); do
    if [[ -f "${PRIOR_RUN}/state.json" ]] && \
      "${PYTHON}" -c 'import json,sys;raise SystemExit(json.load(open(sys.argv[1])).get("stage")!="complete_clean")' \
        "${PRIOR_RUN}/state.json"; then
      break
    fi
    sleep 10
  done
  if (( SECONDS >= deadline )); then
    write_state failed "timed out waiting for c512 cleanup"
    exit 1
  fi

  if [[ ! -f "${PRIOR_RUN}/agentic_steady_summary.json" ]]; then
    "${PYTHON}" "${ROOT}/DualPD/scripts/tools/summarize_agentic_steady.py" \
      "${PRIOR_RUN}" >"${PRIOR_RUN}/summarize.log"
  fi
fi

if ! hard_gate; then
  write_state failed "pre-case hard cleanup gate failed"
  exit 1
fi

write_state running "starting ${RUN_ID}"
if ! env LAYOUT=2p6d PAIR_COUNT=1 CONCURRENCY=576 \
    WARMUP_SECONDS=300 MEASURE_SECONDS=1200 TURN_CAP=2048 \
    "${TOOLS}/ppd45423_browsecomp_smoke.sh" local "${RUN_ID}" \
    >"${CONTROL}/case.log" 2>&1; then
  write_state failed "2p6d case launcher failed"
  exit 1
fi

if ! hard_gate; then
  write_state failed "post-case hard cleanup gate failed"
  exit 1
fi

"${PYTHON}" "${ROOT}/DualPD/scripts/tools/summarize_agentic_steady.py" \
  "${RUN_DIR}" >"${RUN_DIR}/summarize.log"
write_state complete "c576 2p6d complete and clean"
