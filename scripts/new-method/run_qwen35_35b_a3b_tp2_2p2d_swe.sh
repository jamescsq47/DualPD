#!/usr/bin/env bash
# Qwen3.5-35B-A3B TP2, single-node logical 2P:2D, SWE Verified500
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen35_35b_a3b_tp2_2p2d new_method "$@"
