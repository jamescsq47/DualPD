#!/usr/bin/env bash
# D->P path and Router ablations; reuse the current main-method serving path.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-full}" in
  full|no_d2p|host_only|direct_only|d2p_direct_bounded|fast_failure_recompute|random_routing|full-1s|pure-recompute|d2p-slow-only-1s|fast-direct-fail-recompute-1s|direct-only-recompute-1s|d2p-direct-only|d2p-slow-only|random-routing) ;;
  -h|--help)
    echo "Usage: $0 [full|no_d2p|host_only|direct_only|d2p_direct_bounded|fast_failure_recompute|random_routing]"
    echo "Qwen3-8B TP1 4P:4D, c512, 300s warmup + 1200s measurement."
    exit 0 ;;
  *) echo "Unknown path ablation: $1" >&2; exit 2 ;;
esac
exec bash "${SCRIPT_DIR}/run_browsecomp.sh" "${@:-full}"
