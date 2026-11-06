#!/usr/bin/env bash
set -euo pipefail

APPTAINER=/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer
ROOT=/orcd/scratch/orcd/003/shangy/dualpd/ppd45423
SOURCE=${ROOT}/sglang
IMAGE=${1:-/orcd/pool/004/shangy/dualpd/containers/sglang-v0.5.22-ppd45423.sif}
TEST=/home/shangy/dualpd/DualPD/tools/dualpd/test_ppd45423_qwen3_8b.py

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=${SOURCE}/python
export SGLANG_TEST_PD_DISAGG_BACKEND=nixl
export SGLANG_RUST_BUILD_MODE=auto
export SGLANG_CACHE_DIR=/orcd/pool/004/shangy/dualpd/ppd45423-cache
export RUSTUP_HOME=/root/.rustup
export CARGO_HOME=/root/.cargo
export RUSTUP_TOOLCHAIN=stable

exec timeout 45m "${APPTAINER}" exec --nv --cleanenv \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  --env PYTHONPATH="${PYTHONPATH}" \
  --env SGLANG_TEST_PD_DISAGG_BACKEND="${SGLANG_TEST_PD_DISAGG_BACKEND}" \
  --env SGLANG_RUST_BUILD_MODE="${SGLANG_RUST_BUILD_MODE}" \
  --env SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR}" \
  --env RUSTUP_HOME="${RUSTUP_HOME}" \
  --env CARGO_HOME="${CARGO_HOME}" \
  --env RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN}" \
  --bind /orcd:/orcd,/home/shangy/dualpd:/home/shangy/dualpd \
  "${IMAGE}" \
  /opt/sglang/bin/python -m pytest -vv -s "${TEST}"
