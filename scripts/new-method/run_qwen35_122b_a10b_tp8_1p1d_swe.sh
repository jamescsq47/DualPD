#!/usr/bin/env bash
# Qwen3.5-122B-A10B TP8, two-node logical 1P:1D, SWE Verified500
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen35_122b_a10b_tp8_1p1d new_method "$@"
