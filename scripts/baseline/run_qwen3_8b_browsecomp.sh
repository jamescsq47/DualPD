#!/usr/bin/env bash
# Public native baseline defaults; run_pd_case.sh still owns service lifecycle.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Edit concurrency here rather than exporting it in every launch command.
BROWSECOMP_CONCURRENCY=512
mode="${1:-no_reverse}"
case "${mode}" in
  -h|--help)
    echo "Usage: bash scripts/baseline/run_qwen3_8b_browsecomp.sh [no_reverse|hicache]"
    echo "Set MODEL_PATH, PD_DATA_ROOT and BASELINE_ENV_BIN (clean baseline bin directory)."
    exit 0 ;;
  no_reverse) export CASE_MODE=no_reverse ;;
  hicache) export CASE_MODE=native_mooncake ;;
  *) echo "Unknown baseline: ${mode}" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Expected at most one baseline argument" >&2; exit 2
fi
: "${MODEL_PATH:=${PD_MODEL_ROOT:+${PD_MODEL_ROOT}/Qwen3-8B}}"
: "${MODEL_PATH:?Set MODEL_PATH or PD_MODEL_ROOT}"
: "${PD_DATA_ROOT:?Set PD_DATA_ROOT}"
: "${BASELINE_ENV_BIN:?Set BASELINE_ENV_BIN to the clean baseline environment bin directory}"
# Never inherit the modified engine overlay or its agentic switches.
unset SGLANG_OVERLAY_ROOT DUALPD_ENGINE_ROOT
for key in ${!SGLANG_AGENTIC_@} ${!SGLANG_PD_@}; do
  unset "${key}"
done
export PYTHONPATH="${PROJECT_ROOT}"
export PD_ENV_BIN="${BASELINE_ENV_BIN}"
export MODEL_PATH PD_DATA_ROOT
export HF_HOME="${HF_HOME:-${PD_DATA_ROOT}/huggingface-cache}"
defaults=(
  "QA_DATA=${PD_DATA_ROOT}/browsecomp/bc_train.jsonl" 'WORKLOAD_CONFIG='
  'PREFILL_GPUS=0 2 4 6' 'DECODE_GPUS=1 3 5 7'
  'PREFILL_TP_SIZE=1' 'DECODE_TP_SIZE=1' 'TP1_LOCAL_NUMA=true'
  'PREFILL_PORTS=27100 27120 27200 27220'
  'PREFILL_BOOTSTRAP_PORTS=28100 28120 28200 28220'
  'DECODE_PORTS=27101 27121 27201 27221'
  'ROUTER_PORT=27510' 'ROUTER_PROMETHEUS_PORT=27520'
  'PREFILL_MEM_FRACTION_STATICS=0.80 0.80 0.80 0.80'
  'DECODE_MEM_FRACTION_STATICS=0.80 0.80 0.80 0.60'
  'SEARCH_GPU=7' 'SEARCH_PORT=8750' 'START_SEARCH_SERVER=true'
  'SEARCH_START_AFTER_MODELS=true' 'REQUESTS=680'
  "MAX_INFLIGHT=${BROWSECOMP_CONCURRENCY}" 'MATH_RATIO=0'
  'CLOSED_LOOP=true' 'WARMUP_SECONDS=300' 'MEASURE_SECONDS=1200'
  'PRESERVE_SOURCE_ORDER=true' 'DISPATCH_POLICY=fixed'
  "SCHEDULE_FILE=${PROJECT_ROOT}/configs/workloads/fixed_browsecomp_source_order_n680.json"
  'SEED=2026' 'TEMPERATURE=0' 'TOP_P=1' 'TOP_K=-1'
  'MODEL_CONTEXT_LENGTH=40960' 'MODEL_MAX_RESPONSE_LENGTH=36864'
  'MODEL_REASONING_PARSER=' 'MODEL_TOOL_CALL_PARSER='
  'P_HICACHE_SIZE=128' 'D_HICACHE_SIZE=56'
)
for setting in "${defaults[@]}"; do
  key="${setting%%=*}"
  if [[ ! -v "${key}" ]]; then
    printf -v "${key}" '%s' "${setting#*=}"
  fi
  export "${key}"
done
export RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/browsecomp-qwen3-8b-${mode}-c${MAX_INFLIGHT}-$(date -u +%Y%m%dT%H%M%S)-$$}"
exec bash "${PROJECT_ROOT}/runtime/launch/baseline/run_pd_case.sh"
