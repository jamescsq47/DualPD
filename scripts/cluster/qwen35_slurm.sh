#!/usr/bin/env bash
# Adapt the upstream v3 launcher to an existing two-or-more-node Slurm
# allocation.  probe/prepare/plan are CPU-only; service actions use srun and
# never SSH between compute nodes.
set -euo pipefail
dualpd_tools="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
dualpd_py="${DUALPD_PYTHON:-$(command -v python)}"
exec "$dualpd_py" "$dualpd_tools/qwen35_slurm.py" "$@"
