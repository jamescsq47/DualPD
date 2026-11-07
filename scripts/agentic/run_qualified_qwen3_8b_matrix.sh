#!/usr/bin/env bash
set -euo pipefail

# Continue a qualified Qwen3-8B workload from its completed c384 comparison
# into a 3-concurrency x 4-method matrix.  The c384 runs may be produced by a
# preceding pilot; this script waits for and validates them before deciding.

PD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PD_DIR}/.." && pwd)"
SLURM_ALLOCATION_ID="${SLURM_ALLOCATION_ID:?set SLURM_ALLOCATION_ID}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:?set WORKLOAD_CONFIG}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
RUN_PREFIX="${RUN_PREFIX:?set RUN_PREFIX}"
NEW_C384_DIR="${NEW_C384_DIR:?set NEW_C384_DIR}"
QUALIFY_RATIO="${QUALIFY_RATIO:-1.8}"
REQUESTS="${REQUESTS:-8192}"
WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
STEP_MEMORY="${STEP_MEMORY:-1750G}"

complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/requests.jsonl" ]] || return 1
  jq -e \
    '(.status == "steady_churn")
     and (.warmup_seconds >= 299)
     and (.measurement_seconds >= 1199)
     and ((.state_at_measurement_end.failures - .state_at_measurement_start.failures) == 0)' \
    "${run_dir}/closed_loop_boundaries.json" >/dev/null 2>&1
}

wait_for_complete_run() {
  local run_dir="$1"
  while ! complete_run "${run_dir}"; do
    if ! squeue -h -j "${SLURM_ALLOCATION_ID}" | rg -q .; then
      echo "allocation ${SLURM_ALLOCATION_ID} ended while waiting for ${run_dir}" >&2
      exit 1
    fi
    sleep 30
  done
  python "${PD_DIR}/scripts/tools/summarize_agentic_steady.py" "${run_dir}" >/dev/null
}

assert_gpu_and_ports_idle() {
  srun --jobid="${SLURM_ALLOCATION_ID}" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=1G bash -lc '
      set -euo pipefail
      if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | rg -q "[0-9]"; then
        echo "GPU processes remain before the next matrix case" >&2
        exit 1
      fi
      if ss -ltn | rg -q ":(3920[0-7]|39210|39300|39301|39320|39321|39400|39401|39420|39421|39510|39520|40300|40320|40400|40420)\\b"; then
        echo "serving ports remain before the next matrix case" >&2
        exit 1
      fi
    '
}

run_case() {
  local method="$1"
  local concurrency="$2"
  local run_dir="${RUN_ROOT}/${RUN_PREFIX}-${method}-c${concurrency}-w300-m1200-h100"
  if complete_run "${run_dir}"; then
    python "${PD_DIR}/scripts/tools/summarize_agentic_steady.py" "${run_dir}" >/dev/null
    return
  fi
  if [[ -e "${run_dir}" ]]; then
    echo "refusing to overwrite partial run: ${run_dir}" >&2
    exit 1
  fi
  assert_gpu_and_ports_idle
  srun --jobid="${SLURM_ALLOCATION_ID}" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=192 --mem="${STEP_MEMORY}" --gres=gpu:h100:8 bash -lc "
      cd '${PD_DIR}'
      METHOD='${method}' \\
      WORKLOAD_CONFIG='${WORKLOAD_CONFIG}' \\
      MAX_INFLIGHT='${concurrency}' REQUESTS='${REQUESTS}' \\
      WARMUP_SECONDS='${WARMUP_SECONDS}' MEASURE_SECONDS='${MEASURE_SECONDS}' \\
      RUN_DIR='${run_dir}' \\
      bash scripts/agentic/run_qwen3_8b_case.sh
    "
  complete_run "${run_dir}" || {
    echo "incomplete matrix run: ${run_dir}" >&2
    exit 1
  }
  python "${PD_DIR}/scripts/tools/summarize_agentic_steady.py" "${run_dir}" >/dev/null
}

NO_REVERSE_C384_DIR="${RUN_ROOT}/${RUN_PREFIX}-no-reverse-c384-w300-m1200-h100"
HICACHE_C384_DIR="${RUN_ROOT}/${RUN_PREFIX}-hicache-c384-w300-m1200-h100"
COLOCATED_C384_DIR="${RUN_ROOT}/${RUN_PREFIX}-colocated-c384-w300-m1200-h100-revised"

wait_for_complete_run "${NEW_C384_DIR}"
wait_for_complete_run "${NO_REVERSE_C384_DIR}"
wait_for_complete_run "${HICACHE_C384_DIR}"
wait_for_complete_run "${COLOCATED_C384_DIR}"

ratio="$({
  jq -r '.agent_s' "${NEW_C384_DIR}/agentic_steady_summary.json"
  jq -r '.agent_s' "${NO_REVERSE_C384_DIR}/agentic_steady_summary.json"
} | awk 'NR==1{new=$1} NR==2{printf "%.9f", new/$1}')"
if ! awk -v ratio="${ratio}" -v threshold="${QUALIFY_RATIO}" 'BEGIN { exit !(ratio >= threshold) }'; then
  echo "workload did not qualify: new/no-reverse=${ratio}, threshold=${QUALIFY_RATIO}"
  exit 0
fi

echo "workload qualified: new/no-reverse=${ratio}; completing c512/c640 matrix"
for concurrency in 512 640; do
  for method in new no_reverse hicache colocated; do
    run_case "${method}" "${concurrency}"
  done
done

echo "qualified 12-case matrix complete: ${RUN_PREFIX}"
