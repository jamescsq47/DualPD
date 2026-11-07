#!/usr/bin/env python3
"""Seven-day-allocation c576 P->D-only corrected rerun.

Control and result output live on pool storage because the scratch project
quota cannot currently create even controller heartbeat files.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/cluster/browsecomp_qwen3_8b_p2d_only_h100.py"
CONTROL = Path(
    "/orcd/pool/004/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r3"
)
RUN_DIR = Path(
    "/orcd/pool/004/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-only-c576-w300-m1200-20261030-r3"
)


spec = importlib.util.spec_from_file_location("p2d_only_r3_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
run = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run
spec.loader.exec_module(run)

run.CONTROL = CONTROL
run.RUN_DIR = RUN_DIR
run.lowc.CONTROL = CONTROL
run.base.CONTROL = CONTROL


if __name__ == "__main__":
    CONTROL.mkdir(parents=True, exist_ok=True)
    state_path = CONTROL / "state.json"
    existing = json.loads(state_path.read_text()) if state_path.exists() else {}
    run.render_report(existing)
    run.base.main()
