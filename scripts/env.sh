#!/usr/bin/env bash
# Source this file in the environment used for the modified method.
_dualpd_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DUALPD_ROOT="${_dualpd_project}"
export DUALPD_ENGINE_ROOT="${DUALPD_ENGINE_ROOT:-${_dualpd_project}/3rdparty/sglang}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-${DUALPD_ENGINE_ROOT}/python}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${_dualpd_project}:${_dualpd_project}/scripts/cluster${PYTHONPATH:+:${PYTHONPATH}}"
unset _dualpd_project
