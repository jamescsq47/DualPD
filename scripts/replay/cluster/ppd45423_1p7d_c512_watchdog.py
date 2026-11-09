#!/usr/bin/env python3
"""Quiet watchdog for the PR #45423 c512 1P:7D run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/orcd/scratch/orcd/003/shangy/dualpd/ppd45423")
CONTROL = BASE / "aligned-1p7d-c512"
RUNS = BASE / "runs"
CODEX = Path("/home/shangy/.local/bin/codex")
CONTROLLER = b"/home/shangy/dualpd/DualPD/scripts/replay/cluster/ppd45423_browsecomp_1p7d_c512_controller.sh"


def count_contains(needle: bytes) -> int:
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit(): continue
        try: command = Path("/proc", entry, "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError: continue
        count += needle in command
    return count


def count_exact_argument(argument: bytes) -> int:
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit(): continue
        try: arguments = Path("/proc", entry, "cmdline").read_bytes().split(b"\0")
        except OSError: continue
        count += argument in arguments
    return count


def queue(thread: str, message: str) -> None:
    subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120, check=False,
    )


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    previous = ""
    last_count = -1
    last_progress = time.time()
    while True:
        now = time.time()
        errors: list[str] = []
        state_path = CONTROL / "state.json"
        if not state_path.exists():
            time.sleep(10); continue
        state = json.loads(state_path.read_text())
        status = str(state.get("status"))
        if status == "complete":
            queue(args.thread, "PPD #45423 c512 1P:7D COMPLETE: 300+1200 秒窗口完成且清理门禁通过，请读取正式汇总。")
            return 0
        if status == "failed": errors.append(f"controller failed: {state.get('detail')}")
        run_id = str(state.get("run_id"))
        run_dir = RUNS / run_id
        run_stage = None
        event_count = 0
        run_state = run_dir / "state.json"
        if run_state.exists(): run_stage = json.loads(run_state.read_text()).get("stage")
        events = run_dir / "affinity-events.jsonl"
        if events.exists(): event_count = sum(1 for _ in events.open())
        if event_count != last_count:
            last_count = event_count; last_progress = now
        if run_stage == "running_workload" and now - last_progress > 600:
            errors.append(f"{run_id} no request progress for {int(now-last_progress)}s")
        if run_stage in {"starting_services", "running_workload"}:
            servers = count_contains(b"/opt/sglang/bin/python -m sglang.launch_server")
            if servers != 8: errors.append(f"expected 8 model servers, found {servers}")
        if status == "running" and count_exact_argument(CONTROLLER) != 1:
            errors.append("expected one 1p7d c512 controller")
        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest()
        event = {
            "epoch": now, "controller_status": status, "run_id": run_id,
            "run_stage": run_stage, "event_count": event_count,
            "last_progress_epoch": last_progress, "errors": errors,
            "fingerprint": fingerprint,
        }
        with (CONTROL / "watchdog.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        atomic_json(CONTROL / "watchdog-state.json", event)
        if errors and fingerprint != previous:
            queue(args.thread, "PPD #45423 c512 1P:7D WATCHDOG ALERT: " + "; ".join(errors))
        previous = fingerprint
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
