#!/usr/bin/env bash
# Qwen3.8-27B TP2, single-node logical 1P:3D, SWE Verified500
set -euo pipefail
method="${1:-}"
case "${method}" in
  no_reverse|hicache|colocated) shift ;;
  *) echo "Usage: $0 no_reverse|hicache|colocated [launch|prepare|status|stop] [options]" >&2; exit 2 ;;
esac
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen38_27b_tp2_1p3d "${method}" "$@"
