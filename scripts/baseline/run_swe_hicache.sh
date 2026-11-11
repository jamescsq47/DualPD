#!/usr/bin/env bash
set -euo pipefail
exec "${DUALPD_PYTHON:-python}" -m dualpd.runtime.launch.cluster.qwen35_native_pd_slurm "$@" --native-mooncake
