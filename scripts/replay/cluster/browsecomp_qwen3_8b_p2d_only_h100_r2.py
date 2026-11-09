#!/usr/bin/env python3
"""Corrected c576 P->D-only direction ablation.

R2 keeps the requested post-P-ready late binding, but makes Direct admission
respect D's native transfer window and leaves bounded capacity for P->D Host
recovery.  R1 is retained as an invalid diagnostic artifact.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/replay/cluster/browsecomp_qwen3_8b_p2d_only_h100.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r2"
)
RUN_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-only-c576-w300-m1200-20261030-r2"
)


spec = importlib.util.spec_from_file_location("p2d_only_r2_base", SOURCE)
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
