#!/usr/bin/env bash
set -euo pipefail

# Transparent Python launcher used only by the Nsight Systems batch job.
# Ordinary helper programs are passed through.  SGLang model servers are
# placed under nsys and inference.py is told to delimit the steady-state
# capture with cudaProfilerStart/cudaProfilerStop.

: "${NSYS_REAL_PYTHON:?NSYS_REAL_PYTHON must point at the real Python binary}"

is_server=false
is_inference=false
role="server"
port="unknown"
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "-m" && "${argument}" == "sglang.launch_server" ]]; then
    is_server=true
  fi
  if [[ "${argument##*/}" == "inference.py" ]]; then
    is_inference=true
  fi
  if [[ "${previous}" == "--disaggregation-mode" ]]; then
    role="${argument}"
  elif [[ "${previous}" == "--port" ]]; then
    port="${argument}"
  fi
  previous="${argument}"
done

if [[ "${is_server}" == true && -n "${NSYS_OUTPUT_DIR:-}" ]]; then
  : "${NSYS_BIN:=/orcd/software/core/001/pkg/cuda/13.0.1/bin/nsys}"
  mkdir -p "${NSYS_OUTPUT_DIR}"
  gpu="${CUDA_VISIBLE_DEVICES:-unknown}"
  gpu="${gpu//,/+}"
  output="${NSYS_OUTPUT_DIR}/${role}-port${port}-gpu${gpu}"
  exec "${NSYS_BIN}" profile \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output="${output}" \
    "${NSYS_REAL_PYTHON}" "$@"
fi

if [[ "${is_inference}" == true ]]; then
  for argument in "$@"; do
    if [[ "${argument}" == "--cuda-profiler-range" ]]; then
      exec "${NSYS_REAL_PYTHON}" "$@"
    fi
  done
  exec "${NSYS_REAL_PYTHON}" "$@" --cuda-profiler-range
fi

exec "${NSYS_REAL_PYTHON}" "$@"
