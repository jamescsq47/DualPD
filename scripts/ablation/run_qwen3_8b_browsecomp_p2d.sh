#!/usr/bin/env bash
# Retained P->D comparisons; shared data, measurement and service lifecycle.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-p2d_direct}" in
  full|p2d_direct|p2d_prebind|p2d_native_prealloc|p2d_paired_credit) ;;
  -h|--help)
    echo "Usage: $0 [full|p2d_direct|p2d_prebind|p2d_native_prealloc|p2d_paired_credit]"
    echo "Qwen3-8B TP1 4P:4D, default c576; set MAX_INFLIGHT=512 for c512."
    exit 0 ;;
  *) echo "Unknown P->D ablation: $1" >&2; exit 2 ;;
esac
export MAX_INFLIGHT="${MAX_INFLIGHT:-576}"
exec bash "${SCRIPT_DIR}/run_browsecomp.sh" "${@:-p2d_direct}"
