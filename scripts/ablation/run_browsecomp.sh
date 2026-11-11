#!/usr/bin/env bash
# One ablation per invocation; delegate data, window and topology to main entry.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
variant="${1:-full}"
if (( $# > 1 )); then
  echo "Expected one ablation name" >&2; exit 2
fi
export SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=false
export SGLANG_AGENTIC_KV_HOST_STAGING=true
export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=false
export SGLANG_PD_ABLATION_RANDOM_ROUTING=false
export SGLANG_PD_ABLATION_P2D_PREBIND=false
export SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC=false
export SGLANG_PD_ABLATION_P2D_ROUTE_SETTLED=false
export SGLANG_PD_ABLATION_P2D_PAIRED_CREDIT=false
export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=false
export SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE=false
export P2D_HOST_STAGING=true
export FAST_TOOL_THRESHOLD_SECONDS=1 DIRECT_WAIT_SECONDS=1
case "${variant}" in
  full|full-1s) ;;
  no_d2p|pure-recompute) export SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=true ;;
  fast_failure_recompute|fast-direct-fail-recompute-1s)
    export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true ;;
  d2p_direct_bounded|d2p-direct-only)
    # Preserve the 1s tool cutoff; unlike direct_only, do not extend it to a year.
    export SGLANG_AGENTIC_KV_HOST_STAGING=false ;;
  direct_only|direct-only-recompute-1s)
    export FAST_TOOL_THRESHOLD_SECONDS=31536000
    export DIRECT_WAIT_SECONDS=1
    export SGLANG_AGENTIC_KV_HOST_STAGING=false
    export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true ;;
  host_only|d2p-slow-only|d2p-slow-only-1s) export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=true ;;
  p2d_direct|p2d-direct-only) export P2D_HOST_STAGING=false ;;
  p2d_prebind|p2d-prebind-direct-only)
    export SGLANG_PD_ABLATION_P2D_PREBIND=true
    export P2D_HOST_STAGING=false ;;
  p2d_native_prealloc|p2d_paired_credit)
    export SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC=true
    export SGLANG_PD_ABLATION_P2D_ROUTE_SETTLED=true
    export P2D_HOST_STAGING=false
    export PD_MAX_TRANSFER_INFLIGHT=0
    export P_READY_BACKPRESSURE_MODE=disabled
    if [[ "${variant}" == p2d_paired_credit ]]; then
      export SGLANG_PD_ABLATION_P2D_PAIRED_CREDIT=true
    fi ;;
  random_routing|random-routing)
    export SGLANG_PD_ABLATION_RANDOM_ROUTING=true
    export SGLANG_PD_ABLATION_RANDOM_SEED=2026 ;;
  dram_small)
    # Four P ranks: whole-node D→P 64 GiB / P→D 128 GiB.
    export D2P_HOST_ARENA_GIB_PER_P=16
    export P2D_HOST_ARENA_GIB_PER_P=32 ;;
  -h|--help)
    echo "Usage: $0 full|no_d2p|fast_failure_recompute|d2p_direct_bounded|direct_only|host_only|p2d_direct|p2d_prebind|p2d_native_prealloc|p2d_paired_credit|random_routing|dram_small"
    exit 0 ;;
  *) echo "Unknown ablation: ${variant}" >&2; exit 2 ;;
esac
# Keep both public and engine spellings consistent, including inherited shells.
export SGLANG_AGENTIC_KV_P2D_HOST_STAGING="${P2D_HOST_STAGING}"
exec bash "${SCRIPT_DIR}/../new-method/run_qwen3_8b.sh" browsecomp
