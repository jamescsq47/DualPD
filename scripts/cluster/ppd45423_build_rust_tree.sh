#!/usr/bin/env bash
set -euo pipefail

APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
SOURCE=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/sglang
IMAGE=/orcd/pool/004/shangy/dualpd/containers/sglang-v0.5.22-ppd45423.sif
CACHE=/orcd/pool/004/shangy/dualpd/ppd45423-cache

exec timeout 30m "${APPTAINER}" exec --nv --cleanenv \
  --env PYTHONPATH="${SOURCE}/python" \
  --env SGLANG_RUST_BUILD_MODE=auto \
  --env SGLANG_CACHE_DIR="${CACHE}" \
  --env RUSTUP_HOME=/root/.rustup \
  --env CARGO_HOME=/root/.cargo \
  --env RUSTUP_TOOLCHAIN=stable \
  --bind /orcd:/orcd,/home/shangy/dualpd:/home/shangy/dualpd \
  "${IMAGE}" \
  /opt/sglang/bin/python -c \
  'from sglang.srt.mem_cache.rust_tree_core.extension import bindings; print(bindings.__file__)'
