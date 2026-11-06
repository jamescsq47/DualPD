#!/usr/bin/env python3
"""Run the isolated Qwen3.8-27B c128 HiCache ratio=1.5 T450 cell."""

from pathlib import Path

import qwen38_27b_hicache_ratio12_c128_node1800 as run


run.RUN_ID = (
    "q38-27b-h100-1n-2p2d-hicache-mooncake-swe500-"
    "c128-ratio15-20261018-r1"
)
run.CONTROL_DIR = run.RUNS / "q38-27b-hicache-ratio15-c128-node1800-control"
run.STATE_PATH = run.CONTROL_DIR / "state.json"
run.LOCK_PATH = run.CONTROL_DIR / "controller.lock"
run.ALLOCATION_PATH = run.CONTROL_DIR / "allocation.json"
run.RESULT_PATH = run.CONTROL_DIR / "result.json"
run.MAMBA_RATIO = 1.5
run.EXPERIMENT = {
    "concurrency": 128,
    "method": "hicache",
    "run_id": run.RUN_ID,
}
run.EXPERIMENTS = [run.EXPERIMENT]


if __name__ == "__main__":
    run.configure()
    raise SystemExit(run.core.main())
