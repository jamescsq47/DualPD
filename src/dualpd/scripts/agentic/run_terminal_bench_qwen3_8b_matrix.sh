#!/usr/bin/env bash
set -euo pipefail

# Continue the already-running c384 colocated pilot, perform the c384 method
# comparison, and automatically complete the 12-case matrix if it qualifies.

PD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SLURM_ALLOCATION_ID="${SLURM_ALLOCATION_ID:-25279337}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/terminal_bench_2_1_pilot.yaml}"
RUN_ROOT="${RUN_ROOT:-${PD_DIR}/runs-host/agentic-qwen3-8b}"
RUN_PREFIX="${RUN_PREFIX:-tb2}"
REPORT="${REPORT:-${PD_DIR}/runs-host/TERMINAL_BENCH_2_1_QWEN3_8B_H100.md}"
QUALIFY_RATIO="${QUALIFY_RATIO:-1.8}"
STEP_MEMORY="${STEP_MEMORY:-1750G}"
DOCKER_BIN="${DOCKER_BIN:-/orcd/scratch/orcd/003/shangy/dualpd/tools/docker-rootless-28.5.2/bin/docker}"
DOCKER_HOST_VALUE="${DOCKER_HOST_VALUE:-unix:///tmp/dualpd-docker-25279337/docker.sock}"

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

refresh_report() {
  python "${PD_DIR}/scripts/tools/render_agentic_matrix_report.py" \
    --run-root "${RUN_ROOT}" --prefix "${RUN_PREFIX}" --output "${REPORT}" \
    --concurrencies "$@"
}

summarize_run() {
  local run_dir="$1"
  python "${PD_DIR}/scripts/tools/summarize_agentic_steady.py" "${run_dir}" >/dev/null
}

wait_for_initial_run() {
  local run_dir="$1"
  until complete_run "${run_dir}"; do
    if ! squeue -h -j "${SLURM_ALLOCATION_ID}" | rg -q .; then
      echo "allocation ${SLURM_ALLOCATION_ID} ended while waiting for ${run_dir}" >&2
      exit 1
    fi
    sleep 30
  done
  summarize_run "${run_dir}"
  refresh_report 384 512 640
}

node_is_idle() {
  srun --jobid="${SLURM_ALLOCATION_ID}" --overlap --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=1G bash -lc '
      set -euo pipefail
      if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | rg -q "[0-9]"; then
        exit 1
      fi
      if ss -ltn | rg -q ":(3920[0-7]|39210|39300|39301|39320|39321|39400|39401|39420|39421|39510|39520|40300|40320|40400|40420|59603|59651|59652|59680|59690)\\b"; then
        exit 1
      fi
      export DOCKER_HOST="'"${DOCKER_HOST_VALUE}"'"
      [[ "$("'"${DOCKER_BIN}"'" ps -q | wc -l)" -eq 0 ]]
    '
}

wait_for_idle() {
  local waited=0
  until node_is_idle; do
    sleep 10
    waited=$((waited + 10))
    if (( waited % 60 == 0 )); then
      echo "still waiting for GPU processes, serving ports, or Docker tasks to drain (${waited}s)"
    fi
  done
}

run_case() {
  local method="$1"
  local dir_method="$2"
  local concurrency="$3"
  local all_concurrencies="$4"
  local run_dir="${RUN_ROOT}/${RUN_PREFIX}-${dir_method}-c${concurrency}-w300-m1200-h100"
  if complete_run "${run_dir}"; then
    summarize_run "${run_dir}"
    refresh_report ${all_concurrencies}
    return
  fi

  local attempt=1
  while (( attempt <= 3 )); do
    wait_for_idle
    if [[ -e "${run_dir}" ]]; then
      mv "${run_dir}" "${run_dir}.failed-attempt-${attempt}-$(date +%Y%m%d-%H%M%S)"
    fi
    echo "starting method=${method} concurrency=${concurrency} attempt=${attempt}"
    if srun --jobid="${SLURM_ALLOCATION_ID}" --overlap --nodes=1 --ntasks=1 \
      --cpus-per-task=192 --mem="${STEP_MEMORY}" --gres=gpu:h100:8 bash -lc "
        cd '${PD_DIR}'
        METHOD='${method}' \\
        WORKLOAD_CONFIG='${WORKLOAD_CONFIG}' \\
        MAX_INFLIGHT='${concurrency}' REQUESTS=8192 \\
        WARMUP_SECONDS=300 MEASURE_SECONDS=1200 \\
        RUN_DIR='${run_dir}' \\
        bash scripts/agentic/run_qwen3_8b_case.sh
      " && complete_run "${run_dir}"; then
      summarize_run "${run_dir}"
      refresh_report ${all_concurrencies}
      return
    fi
    echo "case failed or incomplete; preserving attempt and retrying after cleanup" >&2
    attempt=$((attempt + 1))
  done
  echo "failed after three attempts: ${run_dir}" >&2
  exit 1
}

all_concurrencies="384 512 640"
colocated_c384="${RUN_ROOT}/${RUN_PREFIX}-colocated-c384-w300-m1200-h100"
wait_for_initial_run "${colocated_c384}"

run_case new new 384 "${all_concurrencies}"
run_case no_reverse no-reverse 384 "${all_concurrencies}"
run_case hicache hicache 384 "${all_concurrencies}"

new_rate="$(jq -r .agent_s "${RUN_ROOT}/${RUN_PREFIX}-new-c384-w300-m1200-h100/agentic_steady_summary.json")"
no_reverse_rate="$(jq -r .agent_s "${RUN_ROOT}/${RUN_PREFIX}-no-reverse-c384-w300-m1200-h100/agentic_steady_summary.json")"
ratio="$(awk -v new="${new_rate}" -v base="${no_reverse_rate}" 'BEGIN {printf "%.9f", new/base}')"
if ! awk -v ratio="${ratio}" -v threshold="${QUALIFY_RATIO}" 'BEGIN {exit !(ratio >= threshold)}'; then
  echo "Terminal-Bench pilot did not qualify: new/no-reverse=${ratio}, threshold=${QUALIFY_RATIO}"
  exit 0
fi

# At c384, a >=60% peak indicates that c640 would approach/exceed saturation;
# use the lower 256/384/512 ladder in that case.
kv_peak="$(jq -r '[.prefill.full_token_usage_peak_per_gpu, .decode.full_token_usage_peak_per_gpu] | max' \
  "${colocated_c384}/agentic_steady_summary.json")"
if awk -v peak="${kv_peak}" 'BEGIN {exit !(peak >= 0.60)}'; then
  all_concurrencies="256 384 512"
  extra_concurrencies="256 512"
else
  all_concurrencies="384 512 640"
  extra_concurrencies="512 640"
fi
echo "qualified at ${ratio}x; KV peak=${kv_peak}; matrix=${all_concurrencies}"

for concurrency in ${extra_concurrencies}; do
  run_case new new "${concurrency}" "${all_concurrencies}"
  run_case no_reverse no-reverse "${concurrency}" "${all_concurrencies}"
  run_case hicache hicache "${concurrency}" "${all_concurrencies}"
  run_case colocated colocated "${concurrency}" "${all_concurrencies}"
done

refresh_report ${all_concurrencies}
echo "Terminal-Bench 12-case matrix complete: ${REPORT}"
