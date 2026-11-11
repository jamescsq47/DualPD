#!/usr/bin/env bash
# Capacity study from the formal DRAM runs; sizes below are not per-node guesses.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
variant="${1:-new_full}"
if (( $# > 1 )); then echo "Expected one capacity variant" >&2; exit 2; fi
method=new_method
case "${variant}" in
  new_full) d2p=128; p2d=64 ;;       # Node totals: 512 / 256 GiB.
  new_half) d2p=64; p2d=32 ;;        # Node totals: 256 / 128 GiB.
  new_small) d2p=16; p2d=32 ;;       # Node totals: 64 / 128 GiB.
  hicache_full) method=native_mooncake; p=128; d=64 ;;
  hicache_reduced) method=native_mooncake; p=96; d=52 ;;
  hicache_p64d64) method=native_mooncake; p=64; d=64 ;;
  -h|--help)
    echo "Usage: $0 [new_full|new_half|new_small|hicache_full|hicache_reduced|hicache_p64d64]"
    echo "Qwen3-8B TP1 4P:4D, c512, 300+1200s; automatically record Host counters and PSS."
    echo "Set MODEL_PATH, PD_DATA_ROOT, RUN_DIR; HiCache also needs BASELINE_ENV_BIN."
    exit 0 ;;
  *) echo "Unknown DRAM variant: ${variant}" >&2; exit 2 ;;
esac
export RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/runs/ablation/dram-${variant}-$(date -u +%Y%m%dT%H%M%S)-$$}"
export MAX_INFLIGHT=512
export REQUESTS=680 WARMUP_REQUESTS=0 MATH_RATIO=0
export SEED=2026 TEMPERATURE=0 TOP_P=1 TOP_K=-1
export PRESERVE_SOURCE_ORDER=true DISPATCH_POLICY=fixed
export WORKLOAD_CONFIG='' EXPERIMENT_CONFIG=''
export SCHEDULE_FILE="${PROJECT_ROOT}/configs/workloads/fixed_browsecomp_source_order_n680.json"
export PREFILL_GPUS="0 2 4 6" DECODE_GPUS="1 3 5 7"
export PREFILL_GPU_GROUPS='0;2;4;6' DECODE_GPU_GROUPS='1;3;5;7'
export PREFILL_TP_SIZE=1 DECODE_TP_SIZE=1
export MEM_FRACTION_STATIC=0.80
export PREFILL_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.80'
export DECODE_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.60'
export MAX_CONTEXT_LENGTH=40960 MODEL_CONTEXT_LENGTH=40960
export MAX_RESPONSE_LENGTH=36864 MODEL_MAX_RESPONSE_LENGTH=36864
export WARMUP_SECONDS=300 MAX_WARMUP_SECONDS=420 MEASURE_SECONDS=1200
export PD_DRAM_TRACE_DIR="${RUN_DIR}/dram-pools" PD_DRAM_SAMPLE_SECONDS=2
export PD_HOST_OBSERVE=true
if [[ "${method}" == new_method ]]; then
  export CLOSED_LOOP=1
  export D2P_HOST_ARENA_GIB_PER_P="${d2p}" P2D_HOST_ARENA_GIB_PER_P="${p2d}"
  export SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB="${d2p}"
  export SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB="${p2d}"
  export SGLANG_AGENTIC_KV_HOST_STAGING=true SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=false
  export SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=false
  export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=false
  export SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=true
  export SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH=32 SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW=8
  export SGLANG_PD_ABLATION_RANDOM_ROUTING=false SGLANG_PD_ABLATION_P2D_PREBIND=false
  export SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC=false SGLANG_PD_ABLATION_P2D_ROUTE_SETTLED=false
  export SGLANG_PD_ABLATION_P2D_PAIRED_CREDIT=false P2D_HOST_STAGING=true
  export SGLANG_AGENTIC_KV_P2D_HOST_STAGING=true
  exec bash "${SCRIPT_DIR}/../new-method/run_qwen3_8b.sh" browsecomp
else
  export CLOSED_LOOP=true
  # Native sizes are decimal GB per rank; the store remains the baseline's 256 GB setting.
  # 48/51 GB are not valid substitutes: historical D's aligned Host pool must exceed GPU KV.
  export P_HICACHE_SIZE="${p}" D_HICACHE_SIZE="${d}"
  exec bash "${SCRIPT_DIR}/../baseline/run_qwen3_8b_browsecomp.sh" hicache
fi
