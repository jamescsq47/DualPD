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
export P2D_HOST_STAGING=true
case "${variant}" in
  full) ;;
  no_d2p) export SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=true ;;
  direct_only)
    export FAST_TOOL_THRESHOLD_SECONDS=31536000
    export DIRECT_WAIT_SECONDS=1
    export SGLANG_AGENTIC_KV_HOST_STAGING=false
    export SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE=true ;;
  host_only) export SGLANG_AGENTIC_KV_FORCE_SLOW_PATH=true ;;
  p2d_direct) export P2D_HOST_STAGING=false ;;
  p2d_prebind)
    export SGLANG_PD_ABLATION_P2D_PREBIND=true
    export P2D_HOST_STAGING=false ;;
  p2d_native_prealloc)
    export SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC=true
    export P2D_HOST_STAGING=false ;;
  random_routing)
    export SGLANG_PD_ABLATION_RANDOM_ROUTING=true
    export SGLANG_PD_ABLATION_RANDOM_SEED=2026 ;;
  dram_small)
    # Four P ranks: whole-node D→P 64 GiB / P→D 128 GiB.
    export D2P_HOST_ARENA_GIB_PER_P=16
    export P2D_HOST_ARENA_GIB_PER_P=32 ;;
  -h|--help)
    echo "Usage: $0 full|no_d2p|direct_only|host_only|p2d_direct|p2d_prebind|p2d_native_prealloc|random_routing|dram_small"
    exit 0 ;;
  *) echo "Unknown ablation: ${variant}" >&2; exit 2 ;;
esac
exec bash "${SCRIPT_DIR}/../new_method/run_qwen3_8b.sh" browsecomp
