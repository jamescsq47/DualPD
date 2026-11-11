#!/usr/bin/env bash
# Restored recipe from 61e3342:scripts/baseline/run_qwen35_27b_tp2_browsecomp_colocated.sh
# Input/environment paths relocated; serving/evaluation settings retained.
set -euo pipefail

# Qwen3.5-specific BrowseComp baseline.  Its workload profile uses the model's
# native tool template and bounds the research loop independently of Qwen3.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PD_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PD_ENV_BIN="${PD_ENV_BIN:-${BASELINE_ENV_BIN:?Set BASELINE_ENV_BIN}}"
export PD_DATA_ROOT="${PD_DATA_ROOT:?Set PD_DATA_ROOT}"
export MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
export MODEL_GPU_GROUPS="${MODEL_GPU_GROUPS:-0,4;1,5;2,6;3,7}"
export MODEL_TP_SIZE=2
export MODEL_ATTENTION_BACKEND="${MODEL_ATTENTION_BACKEND:-triton}"
export MODEL_SAMPLING_BACKEND="${MODEL_SAMPLING_BACKEND:-pytorch}"
export MODEL_PORTS="${MODEL_PORTS:-34900 34901 34902 34903}"
export MODEL_MEM_FRACTION_STATICS="${MODEL_MEM_FRACTION_STATICS:-0.80 0.80 0.80 0.60}"
export ROUTER_PORT="${ROUTER_PORT:-34910}"
export SEARCH_GPU="${SEARCH_GPU:-7}"
export SEARCH_PORT="${SEARCH_PORT:-8750}"
export SEARCH_START_AFTER_MODELS=true
export WORKLOAD_CONFIG="${WORKLOAD_CONFIG:-${PD_DIR}/configs/experiments/browsecomp_qwen35.yaml}"
export MAX_INFLIGHT="${MAX_INFLIGHT:-192}"
export REQUESTS="${REQUESTS:-680}"
export MATH_RATIO=0
export PRESERVE_SOURCE_ORDER=true
export SCHEDULE_FILE="${SCHEDULE_FILE:-${PD_DIR}/configs/workloads/fixed_browsecomp_source_order_n680.json}"
export WARMUP_SECONDS="${WARMUP_SECONDS:-300}"
export MEASURE_SECONDS="${MEASURE_SECONDS:-1200}"
export RUN_DIR="${RUN_DIR:-/tmp/pd-persist/qwen35-27b-tp2-browsecomp-colocated-bounded-observation-c192-w300-m1200-r4}"
export PD_INFERENCE_RETURN_LOGPROB=false
export POST_ANALYZER=none

exec bash "${PD_DIR}/runtime/launch/baseline/run_colocated_case.sh"
