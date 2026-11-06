#!/usr/bin/env bash
set -euo pipefail

APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
TARGET=/orcd/pool/004/shangy/dualpd/containers/sglang-v0.5.22-ppd45423.sif

if [[ -s "${TARGET}" ]]; then
  echo "image already exists: ${TARGET}"
  exit 0
fi

"${APPTAINER}" pull "${TARGET}" docker://lmsysorg/sglang:latest
"${APPTAINER}" inspect "${TARGET}"
