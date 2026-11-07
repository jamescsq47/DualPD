#!/usr/bin/env bash
# Source this file in the environment used for the modified method.
_dualpd_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DUALPD_ROOT="${_dualpd_project}"
export SGLANG_OVERLAY_ROOT="${SGLANG_OVERLAY_ROOT:-$(dirname -- "${_dualpd_project}")/sglang_pd/python}"
export PYTHONPATH="${SGLANG_OVERLAY_ROOT}:${_dualpd_project}:${_dualpd_project}/scripts/cluster${PYTHONPATH:+:${PYTHONPATH}}"
unset _dualpd_project
