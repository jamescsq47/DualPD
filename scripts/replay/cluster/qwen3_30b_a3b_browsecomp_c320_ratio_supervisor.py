#!/usr/bin/env python3
"""Supervise one four-case c320 P:D ratio queue."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


ratio = os.environ.get("PD_RATIO", "")
if ratio not in {"2p2d", "3p1d"}:
    raise RuntimeError("PD_RATIO must be 2p2d or 3p1d")

here = Path(__file__).resolve().parent
source = here / "qwen3_30b_a3b_browsecomp_h100_supervisor.py"
spec = importlib.util.spec_from_file_location("qwen30_ratio_supervisor_base", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {source}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    f"h100-browsecomp-qwen3-30b-a3b-tp2-c320-{ratio}-20261021"
)
base.STATE = base.CONTROL / "state.json"
base.CONTROLLER = here / "qwen3_30b_a3b_browsecomp_c320_ratio_matrix.py"
base.LOG = base.CONTROL / "supervisor.log"
base.HEARTBEAT = base.CONTROL / "supervisor-heartbeat.json"
base.EXPECTED_CASES = 4


if __name__ == "__main__":
    raise SystemExit(base.main())
