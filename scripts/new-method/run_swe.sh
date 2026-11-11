#!/usr/bin/env bash
# Main-method SWE serving: pass model, TP, P:D topology and allocated job flags.
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.qwen35_slurm "$@"
