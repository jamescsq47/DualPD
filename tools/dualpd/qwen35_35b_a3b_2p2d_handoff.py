#!/usr/bin/env python3
"""Hand node2800 from the final 1P:3D tuning run to the 2P:2D matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools/dualpd"
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
SOURCE = (
    RUNS / "q35-35b-a3b-d2p-restore-ft30-p2dl3-d2pl4-r7-20261016"
    / "state.json"
)
TARGET = (
    RUNS / "qwen35-35b-a3b-tp2-single-node-2p2d-c64c128c192-tuned-20261016"
)
CONTROLLER = TOOLS / "qwen35_35b_a3b_single_node_2p2d_matrix_autopilot.py"
WATCHDOG = TOOLS / "qwen35_35b_a3b_single_node_2p2d_watchdog.py"
LOG = TARGET / "handoff.log"


def append(message: str) -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] "
            f"{message}\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    append("waiting for r7 T450 recycle")
    while True:
        try:
            state = json.loads(SOURCE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if state.get("status") == "failed" and not state.get("replacement_job_id"):
            append(f"source failed before replacement: {state.get('error', '')}")
            return 1
        replacement = state.get("replacement_job_id")
        if state.get("status") == "complete" and replacement:
            job_id = str(replacement)
            break
        time.sleep(30)

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        status = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%N"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, check=False,
        ).stdout.strip()
        if status == "RUNNING|node2800":
            break
        time.sleep(10)
    else:
        append(f"replacement {job_id} did not become RUNNING on node2800")
        return 1

    atomic_json(
        TARGET / "allocation.json",
        {
            "job_id": job_id,
            "node": "node2800",
            "status": "running",
            "updated_epoch": time.time(),
            "source": str(SOURCE),
        },
    )
    append(f"admitting 2P:2D matrix on clean replacement {job_id}")
    controller_log = (TARGET / "controller.log").open("a")
    watchdog_log = (TARGET / "watchdog-process.log").open("a")
    controller = subprocess.Popen(
        [
            sys.executable, str(CONTROLLER), "--start",
            "--interval-seconds", "300",
        ],
        cwd=ROOT.parent,
        stdout=controller_log,
        stderr=subprocess.STDOUT,
    )
    state_path = TARGET / "state.json"
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and controller.poll() is None:
        if state_path.is_file():
            try:
                if json.loads(state_path.read_text()).get("status") in {
                    "running", "paused", "complete"
                }:
                    break
            except json.JSONDecodeError:
                pass
        time.sleep(5)
    watchdog_process = subprocess.Popen(
        [sys.executable, str(WATCHDOG), "--interval-seconds", "600"],
        cwd=ROOT.parent,
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
    )
    append(
        f"controller pid={controller.pid}; watchdog pid={watchdog_process.pid}"
    )
    return_code = controller.wait()
    append(f"controller exited rc={return_code}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
