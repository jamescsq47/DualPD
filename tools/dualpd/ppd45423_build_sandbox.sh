#!/usr/bin/env bash
set -euo pipefail

APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
TARGET=/orcd/pool/004/shangy/dualpd/containers/sglang-v0.5.22-ppd45423-rootfs

if [[ -d "${TARGET}" ]]; then
  echo "sandbox already exists: ${TARGET}"
else
  "${APPTAINER}" build --sandbox "${TARGET}" docker://lmsysorg/sglang:latest
fi

"${APPTAINER}" exec "${TARGET}" /bin/true
"${APPTAINER}" inspect "${TARGET}"
