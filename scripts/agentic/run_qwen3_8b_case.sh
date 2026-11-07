#!/usr/bin/env bash
set -euo pipefail

# One comparable Qwen3-8B serving case for lightweight multi-turn workloads.
# METHOD: colocated | new | no_reverse | hicache

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PD_DIR}/.." && pwd)"

METHOD="${METHOD:-colocated}"
WORKLOAD_CONFIG="${WORKLOAD_CONFIG:?WORKLOAD_CONFIG is required}"
MAX_INFLIGHT="${MAX_INFLIGHT:-384}"
RUN_DIR="${RUN_DIR:-${PD_DIR}/runs-host/agentic-qwen3-8b/${METHOD}-c${MAX_INFLIGHT}}"

export MODEL_PATH="${MODEL_PATH:-${WORKSPACE_ROOT}/models/Qwen3-8B}"
export WORKLOAD_CONFIG MAX_INFLIGHT RUN_DIR
export REQUESTS="${REQUESTS:-8192}"
export SEED="${SEED:-2026}"
export TEMPERATURE="${TEMPERATURE:-0}"
export TOP_P="${TOP_P:-1}"
export TOP_K="${TOP_K:--1}"
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export MODEL_CONTEXT_LENGTH="${MODEL_CONTEXT_LENGTH:-40960}"
export MODEL_MAX_RESPONSE_LENGTH="${MODEL_MAX_RESPONSE_LENGTH:-36864}"
export MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-${MODEL_CONTEXT_LENGTH}}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${MODEL_MAX_RESPONSE_LENGTH}}"
export START_SEARCH_SERVER=false
export PD_SKIP_SEARCH=1
export PRESERVE_SOURCE_ORDER=true
export POST_ANALYZER=none
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
export WEB_SEARCH_CONCURRENCY="${WEB_SEARCH_CONCURRENCY:-16}"
export WEB_FETCH_CONCURRENCY="${WEB_FETCH_CONCURRENCY:-32}"

case "${METHOD}" in
  colocated)
    export PD_ENV_BIN="${PD_ENV_BIN:-/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd_baseline/bin}"
    export MODEL_GPUS="${MODEL_GPUS:-0 1 2 3 4 5 6 7}"
    export MODEL_TP_SIZE=1
    export MODEL_PORTS="${MODEL_PORTS:-39200 39201 39202 39203 39204 39205 39206 39207}"
    export ROUTER_PORT="${ROUTER_PORT:-39210}"
    exec bash "${PD_DIR}/scripts/baseline/run_colocated_case.sh"
    ;;
  no_reverse|hicache)
    export PD_ENV_BIN="${PD_ENV_BIN:-/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd_baseline/bin}"
    export PREFILL_GPUS="${PREFILL_GPUS:-0 2 4 6}"
    export DECODE_GPUS="${DECODE_GPUS:-1 3 5 7}"
    export PREFILL_TP_SIZE=1
    export DECODE_TP_SIZE=1
    export PREFILL_PORTS="${PREFILL_PORTS:-39300 39320 39400 39420}"
    export PREFILL_BOOTSTRAP_PORTS="${PREFILL_BOOTSTRAP_PORTS:-40300 40320 40400 40420}"
    export DECODE_PORTS="${DECODE_PORTS:-39301 39321 39401 39421}"
    export ROUTER_PORT="${ROUTER_PORT:-39510}"
    export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-39520}"
    export MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-59651}"
    export MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-59680}"
    export MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-59603}"
    export MOONCAKE_CLIENT_PORT="${MOONCAKE_CLIENT_PORT:-59652}"
    export MOONCAKE_CLIENT_HTTP_PORT="${MOONCAKE_CLIENT_HTTP_PORT:-59690}"
    if [[ "${METHOD}" == no_reverse ]]; then
      export CASE_MODE=no_reverse
    else
      export CASE_MODE=native_mooncake
    fi
    exec bash "${PD_DIR}/scripts/baseline/run_pd_case.sh"
    ;;
  new)
    export PD_ENV_BIN="${PD_ENV_BIN:-/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin}"
    export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-${WORKSPACE_ROOT}/sglang_pd/python}"
    export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${PD_DIR}:${WORKSPACE_ROOT}/slime:${PYTHONPATH:-}"
    export CPATH="/usr/include/x86_64-linux-gnu:${CPATH:-}"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
    export PREFILL_GPUS="${PREFILL_GPUS:-0 2 4 6}"
    export DECODE_GPUS="${DECODE_GPUS:-1 3 5 7}"
    export PREFILL_TP_SIZE=1
    export DECODE_TP_SIZE=1
    export PREFILL_PORTS="${PREFILL_PORTS:-39300 39320 39400 39420}"
    export BOOTSTRAP_PORTS="${BOOTSTRAP_PORTS:-40300 40320 40400 40420}"
    export DECODE_PORTS="${DECODE_PORTS:-39301 39321 39401 39421}"
    export ROUTER_PORT="${ROUTER_PORT:-39510}"
    export ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-39520}"
    export CLOSED_LOOP=1
    export WARMUP_REQUESTS=0
    export MAX_WARMUP_SECONDS="$((WARMUP_SECONDS + 120))"
    export D2P_HOST_ARENA_GIB_PER_P=128
    export P2D_HOST_STAGING=true
    export P2D_HOST_ARENA_GIB_PER_P=64
    export FAST_TOOL_THRESHOLD_SECONDS="${FAST_TOOL_THRESHOLD_SECONDS:-2}"
    export DIRECT_WAIT_SECONDS="${DIRECT_WAIT_SECONDS:-2}"
    export P_ACCEPT_TIMEOUT_SECONDS="${P_ACCEPT_TIMEOUT_SECONDS:-600}"
    export P_QUEUE_TIMEOUT_SECONDS="${P_QUEUE_TIMEOUT_SECONDS:-3600}"
    export P_READY_TIMEOUT_SECONDS="${P_READY_TIMEOUT_SECONDS:-600}"
    export D_TARGET_KV_FRACTION=1.0
    export P_READY_BACKPRESSURE_MODE=disabled
    exec bash "${PD_DIR}/scripts/new_method/run_4p4d_numa_case.sh"
    ;;
  *)
    echo "METHOD must be colocated, new, no_reverse, or hicache" >&2
    exit 2
    ;;
esac
