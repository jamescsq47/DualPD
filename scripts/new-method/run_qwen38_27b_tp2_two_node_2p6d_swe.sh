#!/usr/bin/env bash
# Qwen3.8-27B TP2, two-node mixed logical 2P:6D, SWE Verified500
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.recipes qwen38_27b_tp2_two_node_2p6d new_method "$@"
