#!/usr/bin/env python3
"""Quiet ten-minute watchdog for the Qwen3.5-35B-A3B 2P:2D matrix."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

try:
    from . import qwen35_35b_a3b_matrix_watchdog as watchdog
    from . import qwen35_35b_a3b_single_node_2p2d_matrix_autopilot as matrix
except ImportError:
    import qwen35_35b_a3b_matrix_watchdog as watchdog
    import qwen35_35b_a3b_single_node_2p2d_matrix_autopilot as matrix


CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "qwen35-35b-a3b-tp2-single-node-2p2d-c64c128c192-tuned-20261016"
)


def matrix_snapshot() -> dict[str, Any]:
    matrix.configure()
    matrix.base.configure_core()
    state = watchdog.read_json(CONTROL_DIR / "state.json")
    snapshot: dict[str, Any] = {"matrix_state": state}
    if int(state["index"]) >= len(matrix.EXPERIMENTS):
        return snapshot
    experiment = matrix.EXPERIMENTS[int(state["index"])]
    directory = matrix.base.run_dir(experiment)
    rows = matrix.base.core.load_progress(directory)
    snapshot.update(
        experiment=experiment,
        records=len(rows),
        # The controller deliberately keeps the completed run as the current
        # state while it tears that runtime down and starts the next one.  A
        # sealed result is therefore the authoritative handoff marker even
        # when the workload stopped at T450 and never emitted 500 callbacks.
        result_sealed=(
            (directory / "T450_RESULT.json").is_file()
            or (directory / "RESULT.json").is_file()
        ),
        completed=sum(
            str(row.get("status")) == "Status.COMPLETED" for row in rows
        ),
        environment_errors=sum(bool(row.get("environment_error")) for row in rows),
        resolved=sum(row.get("reward") == 1 for row in rows),
        expected_ports=sorted(matrix.base.expected_runtime_ports(experiment)),
    )
    if rows:
        snapshot["last_completed_at"] = rows[-1].get("completed_at")
    return snapshot


def main() -> int:
    watchdog.CONTROL_DIR = CONTROL_DIR
    watchdog.STATE_PATH = CONTROL_DIR / "state.json"
    watchdog.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
    watchdog.WATCHDOG_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
    watchdog.WATCHDOG_LOG_PATH = CONTROL_DIR / "watchdog.jsonl"
    watchdog.WATCHDOG_LOCK_PATH = CONTROL_DIR / "watchdog.lock"
    watchdog.CONTROLLER = Path(matrix.__file__).resolve()
    watchdog.matrix_snapshot = matrix_snapshot
    return watchdog.main()


if __name__ == "__main__":
    raise SystemExit(main())
