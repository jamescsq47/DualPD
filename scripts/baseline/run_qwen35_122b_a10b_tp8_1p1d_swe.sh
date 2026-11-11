#!/usr/bin/env bash
# Qwen3.5-122B-A10B TP8, two-node logical 1P:1D, SWE Verified500
set -euo pipefail
method="${1:-}"
case "${method}" in
  no_reverse|hicache|colocated) shift ;;
  *) echo "Usage: $0 no_reverse|hicache|colocated [launch|prepare|status|stop] [options]" >&2; exit 2 ;;
esac
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen35_122b_a10b_tp8_1p1d "${method}" "$@"
