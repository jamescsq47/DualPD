#!/usr/bin/env bash
# Qwen3.5-9B TP1, single-node logical 2P:6D, SWE Verified500 PR51 fenced-shell
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen35_9b_tp1_2p6d new_method "$@"
