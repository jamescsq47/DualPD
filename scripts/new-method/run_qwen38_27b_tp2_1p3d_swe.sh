#!/usr/bin/env bash
# Qwen3.8-27B TP2, single-node logical 1P:3D, SWE Verified500
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen38_27b_tp2_1p3d new_method "$@"
