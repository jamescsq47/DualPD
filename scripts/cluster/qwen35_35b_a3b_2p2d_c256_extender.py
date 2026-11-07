#!/usr/bin/env python3
"""Append the requested c256 methods after the original nine-run 2P:2D matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
CONTROL = (
    RUNS / "qwen35-35b-a3b-tp2-single-node-2p2d-c64c128c192-tuned-20261016"
)
STATE = CONTROL / "state.json"
LOG = CONTROL / "c256-extender.log"
CONTROLLER = ROOT / "scripts/cluster/qwen35_35b_a3b_single_node_2p2d_matrix_autopilot.py"
WATCHDOG = ROOT / "scripts/cluster/qwen35_35b_a3b_single_node_2p2d_watchdog.py"
OLD_RUN_COUNT = 9


def append(message: str) -> None:
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


def script_pids(script: Path) -> list[int]:
    needle = str(script).encode()
    result: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            argv = Path("/proc", entry, "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in argv:
            result.append(int(entry))
    return result


def main() -> int:
    append("waiting for the original nine-run matrix to complete")
    while True:
        try:
            state = json.loads(STATE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(30)
            continue
        if state.get("status") == "paused":
            append(f"matrix paused before c256 handoff: {state.get('error', '')}")
            return 1
        if (
            state.get("status") == "complete"
            and int(state.get("index", -1)) == OLD_RUN_COUNT
        ):
            break
        time.sleep(30)

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline and script_pids(CONTROLLER):
        time.sleep(5)
    if script_pids(CONTROLLER):
        append("old controller did not exit after matrix completion")
        return 1

    # The original watchdog has the old nine-item list loaded in memory.
    # Replace only that monitor process; the allocation and data plane are
    # already stopped by the controller's exact cleanup path.
    for pid in script_pids(WATCHDOG):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and script_pids(WATCHDOG):
        time.sleep(1)
    if script_pids(WATCHDOG):
        append("old watchdog did not exit")
        return 1

    sys.path.insert(0, str(ROOT / "scripts/cluster"))
    import qwen35_35b_a3b_single_node_2p2d_matrix_autopilot as matrix

    matrix.configure()
    matrix.base.configure_core()
    matrix.base.core.load_allocation()
    try:
        matrix.base.core.assert_clean_for_start()
    except BaseException as error:
        append(f"zero gate failed; rebuilding allocation: {type(error).__name__}: {error}")
        matrix.rebuild_allocation(f"c256 extension admission: {error}")

    state.update(
        status="ready",
        index=OLD_RUN_COUNT,
        progress=0,
        error="",
        extension="c256-agentic-hicache-native",
        extension_started_epoch=time.time(),
    )
    state.pop("current_run_id", None)
    atomic_json(STATE, state)
    append("starting c256 method 1/4 after a clean admission gate")

    controller_log = (CONTROL / "controller.log").open("a")
    watchdog_log = (CONTROL / "watchdog-process.log").open("a")
    controller = subprocess.Popen(
        [sys.executable, str(CONTROLLER), "--start", "--interval-seconds", "300"],
        cwd=ROOT.parent,
        stdout=controller_log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline and controller.poll() is None:
        try:
            current = json.loads(STATE.read_text())
        except json.JSONDecodeError:
            time.sleep(5)
            continue
        if current.get("status") in {"running", "paused", "complete"}:
            break
        time.sleep(5)
    watchdog_process = subprocess.Popen(
        [sys.executable, str(WATCHDOG), "--interval-seconds", "600"],
        cwd=ROOT.parent,
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
    )
    append(f"controller pid={controller.pid}; watchdog pid={watchdog_process.pid}")
    return_code = controller.wait()
    append(f"extended controller exited rc={return_code}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
