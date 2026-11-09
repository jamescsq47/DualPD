#!/usr/bin/env python3
"""Node-local quiet watchdog for the baseline-aligned PR #45423 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path("/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/aligned-matrix")
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/runs")
CODEX = Path("/home/shangy/.local/bin/codex")


def process_count(prefix: bytes) -> int:
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            command = Path("/proc", entry, "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        count += command.startswith(prefix)
    return count


def process_contains(needle: bytes) -> int:
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


def queue(thread: str, message: str) -> None:
    subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
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
    last_count = -1
    last_progress = time.time()
    previous = ""

    while True:
        now = time.time()
        errors: list[str] = []
        matrix = json.loads((CONTROL / "state.json").read_text())
        status = str(matrix.get("status"))
        concurrency = matrix.get("current_concurrency")
        run_stage = None
        event_count = 0

        # The controller advances to the next concurrency immediately after a
        # case is summarized, so poll the first requested case independently.
        # This creates exactly one wake-up in the current Codex thread.
        first_run = RUNS / "ppd45423-browsecomp-local-4p4d-c384-w300-m1200-r1"
        first_notice = CONTROL / "c384-complete-notified"
        first_state = first_run / "state.json"
        first_summary = first_run / "agentic_steady_summary.json"
        if first_state.exists() and first_summary.exists() and not first_notice.exists():
            if json.loads(first_state.read_text()).get("stage") == "complete_clean":
                queue(
                    args.thread,
                    "PPD #45423 c384 FIRST FORMAL CASE COMPLETE: 300s warmup + "
                    "1200s steady state 已完成且清理门禁通过。请立即读取 "
                    "agentic_steady_summary.json 和 affinity-summary.json，向用户汇报结果。",
                )
                first_notice.write_text(f"{now}\n")

        if status == "complete":
            queue(args.thread, "PPD #45423 BASELINE-ALIGNED MATRIX COMPLETE: c384/c512/c576 已全部完成，请核验结果与最终清理门禁。")
            return 0
        if status == "failed":
            errors.append(f"controller failed at c{concurrency}: {matrix.get('detail')}")
        if status in {"running", "transition"} and concurrency:
            run_id = f"ppd45423-browsecomp-local-4p4d-c{concurrency}-w300-m1200-r1"
            run_dir = RUNS / run_id
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
                errors.append(f"c{concurrency} no request progress for {int(now-last_progress)}s")
            servers = process_count(b"/opt/sglang/bin/python -m sglang.launch_server")
            if run_stage in {"starting_services", "running_workload"} and servers != 8:
                errors.append(f"c{concurrency} expected 8 model servers, found {servers}")
        controller_count = process_contains(
            b"/home/shangy/dualpd/DualPD/scripts/replay/cluster/ppd45423_browsecomp_aligned_matrix.sh"
        )
        if status in {"running", "transition"} and controller_count != 1:
            errors.append(f"expected one matrix controller, found {controller_count}")

        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest()
        event = {
            "epoch": now, "matrix_status": status, "concurrency": concurrency,
            "run_stage": run_stage, "event_count": event_count,
            "last_progress_epoch": last_progress, "errors": errors,
            "fingerprint": fingerprint,
        }
        with (CONTROL / "watchdog.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        atomic_json(CONTROL / "watchdog-state.json", event)
        if errors and fingerprint != previous:
            queue(args.thread, "PPD #45423 BASELINE-ALIGNED WATCHDOG ALERT: " + "; ".join(errors))
        previous = fingerprint
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
