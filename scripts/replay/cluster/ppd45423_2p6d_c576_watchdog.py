#!/usr/bin/env python3
"""Quiet watchdog for the c512-to-c576-2P6D transition."""

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
CONTROL = BASE / "aligned-2p6d-c576"
RUNS = BASE / "runs"
CODEX = Path("/home/shangy/.local/bin/codex")


def count_contains(needle: bytes) -> int:
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            command = Path("/proc", entry, "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        count += needle in command
    return count


def count_exact_argument(argument: bytes) -> int:
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            arguments = Path("/proc", entry, "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        count += argument in arguments
    return count


def queue(thread: str, message: str) -> None:
    subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=120, check=False,
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
        state = json.loads((CONTROL / "state.json").read_text())
        status = str(state.get("status"))
        if status == "complete":
            queue(
                args.thread,
                "PPD #45423 c576 2P:6D COMPLETE: 300s warmup + 1200s steady "
                "state 已完成且清理门禁通过。请读取正式汇总并向用户汇报。",
            )
            return 0
        if status == "failed":
            errors.append(f"controller failed: {state.get('detail')}")

        if status == "waiting":
            run_id = "ppd45423-browsecomp-local-4p4d-c512-w300-m1200-r1"
        else:
            run_id = str(state.get("run_id"))
        run_dir = RUNS / run_id
        run_stage = None
        event_count = 0
        state_path = run_dir / "state.json"
        if state_path.exists():
            run_stage = json.loads(state_path.read_text()).get("stage")
        event_path = run_dir / "affinity-events.jsonl"
        if event_path.exists():
            event_count = sum(1 for _ in event_path.open())
        if event_count != last_count:
            last_count = event_count
            last_progress = now
        if run_stage == "running_workload" and now - last_progress > 600:
            errors.append(f"{run_id} no request progress for {int(now-last_progress)}s")
        if run_stage in {"starting_services", "running_workload"}:
            servers = count_contains(b"/opt/sglang/bin/python -m sglang.launch_server")
            if servers != 8:
                errors.append(f"expected 8 model servers, found {servers}")

        controllers = count_exact_argument(
            b"/home/shangy/dualpd/DualPD/scripts/replay/cluster/ppd45423_browsecomp_2p6d_c576_controller.sh"
        )
        if status in {"waiting", "running"} and controllers != 1:
            errors.append(f"expected one 2p6d controller, found {controllers}")

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
            queue(args.thread, "PPD #45423 2P:6D WATCHDOG ALERT: " + "; ".join(errors))
        previous = fingerprint
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
