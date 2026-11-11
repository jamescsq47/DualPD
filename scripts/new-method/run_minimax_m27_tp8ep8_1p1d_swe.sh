#!/usr/bin/env bash
# MiniMax-M2.7 TP8/EP8, two-node logical 1P:1D, SWE Verified500
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes minimax_m27_tp8ep8_1p1d new_method "$@"
