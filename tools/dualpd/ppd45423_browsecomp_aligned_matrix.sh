#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/shangy/dualpd
TOOLS=${ROOT}/slime/tools/dualpd
PYTHON=/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python
CONTROL=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/aligned-matrix
RUNS=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/runs
CONCURRENCIES=(384 512 576)
PORTS=(28900 29000 29100 29101 29102 29103 29200 29201 29202 29203 \
  29300 29301 29302 29303 29500 29501 29502 29503 29600 29601 29602 29603 \
  29700 29701 29702 29703)

mkdir -p "${CONTROL}"
printf '%s\n' "$$" >"${CONTROL}/controller.pid"

write_state() {
  local status="$1" concurrency="${2:-}" detail="${3:-}"
  STATUS_VALUE="${status}" CONCURRENCY_VALUE="${concurrency}" DETAIL_VALUE="${detail}" \
    CONTROL_VALUE="${CONTROL}" "${PYTHON}" - <<'PY'
import json, os, time
from pathlib import Path
p=Path(os.environ['CONTROL_VALUE'])/'state.json'
value={'epoch':time.time(),'status':os.environ['STATUS_VALUE'],
       'current_concurrency':int(os.environ['CONCURRENCY_VALUE']) if os.environ['CONCURRENCY_VALUE'] else None,
       'detail':os.environ['DETAIL_VALUE']}
t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');t.replace(p)
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

for concurrency in "${CONCURRENCIES[@]}"; do
  run_id="ppd45423-browsecomp-local-4p4d-c${concurrency}-w300-m1200-r1"
  run_dir="${RUNS}/${run_id}"
  if [[ -f "${run_dir}/state.json" ]] && \
    "${PYTHON}" -c 'import json,sys;raise SystemExit(json.load(open(sys.argv[1])).get("stage")!="complete_clean")' \
      "${run_dir}/state.json"; then
    continue
  fi
  if ! hard_gate; then
    write_state failed "${concurrency}" "pre-case hard cleanup gate failed"
    exit 1
  fi
  write_state running "${concurrency}" "starting ${run_id}"
  if ! env PAIR_COUNT=4 WARMUP_SECONDS=300 MEASURE_SECONDS=1200 \
      CONCURRENCY="${concurrency}" TURN_CAP=2048 \
      "${TOOLS}/ppd45423_browsecomp_smoke.sh" local "${run_id}" \
      >"${CONTROL}/c${concurrency}.log" 2>&1; then
    write_state failed "${concurrency}" "case launcher failed"
    exit 1
  fi
  if ! hard_gate; then
    write_state failed "${concurrency}" "post-case hard cleanup gate failed"
    exit 1
  fi
  "${PYTHON}" "${ROOT}/slime/examples/pd/scripts/tools/summarize_agentic_steady.py" \
    "${run_dir}" >"${run_dir}/summarize.log"
  write_state transition "${concurrency}" "case complete and clean"
done

write_state complete "" "all baseline-aligned PPD cases complete"
