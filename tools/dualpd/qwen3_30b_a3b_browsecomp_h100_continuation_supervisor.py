#!/usr/bin/env python3
"""Supervisor entry point for the c128/c384 continuation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SOURCE = Path(__file__).with_name("qwen3_30b_a3b_browsecomp_h100_supervisor.py")
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-30b-a3b-tp2-c128-c384-20261013"
)
CONTROLLER = Path(__file__).with_name("qwen3_30b_a3b_browsecomp_h100_continuation.py")

spec = importlib.util.spec_from_file_location("qwen3_30b_a3b_parent_supervisor", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
supervisor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervisor
spec.loader.exec_module(supervisor)

supervisor.CONTROL = CONTROL
supervisor.STATE = CONTROL / "state.json"
supervisor.CONTROLLER = CONTROLLER
supervisor.LOG = CONTROL / "supervisor.log"
supervisor.HEARTBEAT = CONTROL / "supervisor-heartbeat.json"
supervisor.EXPECTED_CASES = 8


if __name__ == "__main__":
    raise SystemExit(supervisor.main())
