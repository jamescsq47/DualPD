#!/usr/bin/env bash
# Public single-node Qwen3-8B defaults; older topology/replay entries stay intact.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
workload="${1:-browsecomp}"
case "${workload}" in
  -h|--help)
    echo "Usage: bash scripts/new_method/run_qwen3_8b.sh [browsecomp|swe]"
    echo "Set MODEL_PATH and PD_DATA_ROOT; optional RUN_DIR, MAX_INFLIGHT and other environment overrides."
    exit 0 ;;
  browsecomp|swe) ;;
  *) echo "Unknown workload: ${workload}" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Expected at most one workload argument" >&2
  exit 2
fi
: "${MODEL_PATH:=${PD_MODEL_ROOT:+${PD_MODEL_ROOT}/Qwen3-8B}}"
: "${MODEL_PATH:?Set MODEL_PATH or PD_MODEL_ROOT}"
: "${PD_DATA_ROOT:?Set PD_DATA_ROOT to the prepared dataset directory}"

defaults=(
  "DUALPD_ENGINE_ROOT=${PROJECT_ROOT}/3rdparty/sglang"
  "PD_ENV_BIN=$(dirname -- "$(command -v python)")"
  'PREFILL_GPUS=0 2 4 6' 'DECODE_GPUS=1 3 5 7'
  'PREFILL_TP_SIZE=1' 'DECODE_TP_SIZE=1' 'SEARCH_GPU=7'
  'SEARCH_START_AFTER_MODELS=true' 'MEM_FRACTION_STATIC=0.80'
  'MATH_RATIO=0' 'PRESERVE_SOURCE_ORDER=true' 'SEED=2026'
  'TEMPERATURE=0' 'TOP_P=1' 'TOP_K=-1'
  'MAX_CONTEXT_LENGTH=40960' 'MAX_RESPONSE_LENGTH=36864'
  'WARMUP_REQUESTS=0' 'WARMUP_SECONDS=300' 'MAX_WARMUP_SECONDS=420'
  'MEASURE_SECONDS=1200' 'FAST_TOOL_THRESHOLD_SECONDS=1' 'DIRECT_WAIT_SECONDS=1'
  'D2P_HOST_ARENA_GIB_PER_P=128' 'P2D_HOST_ARENA_GIB_PER_P=64'
  'P2D_HOST_STAGING=true'
  'SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND=memfd'
  'SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND=memfd'
  'SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB=768'
  'SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA=1'
  'P_H2D_MAX_INFLIGHT=2' 'SGLANG_AGENTIC_KV_D2H_INFLIGHT=2'
  'SGLANG_AGENTIC_KV_P2D_D2H_WORKERS=2' 'SGLANG_AGENTIC_KV_P2D_H2D_WORKERS=2'
  'SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS=4096'
  'SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS=4096'
  'SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS=4096'
  'SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS=4096'
  'SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH=32'
  'SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW=8'
)
if [[ "${workload}" == browsecomp ]]; then
  defaults+=(
    "QA_DATA=${PD_DATA_ROOT}/browsecomp/bc_train.jsonl"
    "SCHEDULE_FILE=${PROJECT_ROOT}/configs/workloads/fixed_browsecomp_source_order_n680.json"
    'WORKLOAD_CONFIG=' 'PD_SKIP_SEARCH=0' 'MODEL_REASONING_PARSER='
    'MODEL_TOOL_CALL_PARSER=' 'EXPERIMENT_CONFIG='
    'DECODE_MEM_FRACTION_STATICS=0.80 0.80 0.80 0.60'
    'REQUESTS=680' 'MAX_INFLIGHT=256' 'CLOSED_LOOP=1'
  )
else
  defaults+=(
    "WORKLOAD_CONFIG=${PROJECT_ROOT}/configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml"
    'PD_SKIP_SEARCH=1' 'MODEL_REASONING_PARSER=qwen3'
    'MODEL_TOOL_CALL_PARSER=qwen25' 'EXPERIMENT_CONFIG='
    'DECODE_MEM_FRACTION_STATICS=0.80 0.80 0.80 0.80'
    'REQUESTS=500' 'MAX_INFLIGHT=64' 'CLOSED_LOOP=0'
  )
fi
for setting in "${defaults[@]}"; do
  key="${setting%%=*}"
  # Preserve even explicitly empty caller values, like the profile loader.
  if [[ ! -v "${key}" ]]; then
    printf -v "${key}" '%s' "${setting#*=}"
  fi
  export "${key}"
done
export MODEL_PATH PD_DATA_ROOT
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-${DUALPD_ENGINE_ROOT}/python}"
export HF_HOME="${HF_HOME:-${PD_DATA_ROOT}/huggingface-cache}"
export RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/${workload}-qwen3-8b-c${MAX_INFLIGHT}-$(date -u +%Y%m%dT%H%M%S)-$$}"
source "${PROJECT_ROOT}/scripts/env.sh"
exec bash "${SCRIPT_DIR}/run_4p4d_numa_case.sh"
