#!/usr/bin/env bash
# Qwen3.5-9B TP1, single-node logical 2P:6D, SWE Verified500 PR51 fenced-shell
set -euo pipefail
method="${1:-}"
case "${method}" in
  no_reverse|hicache|colocated) shift ;;
  *) echo "Usage: $0 no_reverse|hicache|colocated [launch|prepare|status|stop] [options]" >&2; exit 2 ;;
esac
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen35_9b_tp1_2p6d "${method}" "$@"
