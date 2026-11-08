"""Keep the user-requested single-node allocation available for serial tests.

This process does not launch or terminate GPU services. It only records a live
allocation heartbeat. Experiment launchers must separately enforce their own
audit, resource-ownership and post-run cleanup gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise SystemExit("must run inside the dedicated Slurm allocation")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stopped = False
    stop_signal = None

    def stop(signum, _frame):
        nonlocal stopped, stop_signal
        stopped = True
        stop_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    info = {
        "job_id": job_id, "node": socket.gethostname(), "pid": os.getpid(),
        "started_at": started, "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "job_end_time": os.environ.get("SLURM_JOB_END_TIME"),
        "purpose": "seven-day user-requested single-node serial experiment allocation",
    }
    try:
        while not stopped:
            atomic_json(args.state_dir / "allocation.json", {
                **info, "status": "running", "heartbeat_at": time.time(),
            })
            for _ in range(30):
                if stopped:
                    break
                time.sleep(1)
    finally:
        atomic_json(args.state_dir / "allocation.json", {
            **info, "status": "stopped", "heartbeat_at": time.time(),
            "stop_signal": stop_signal,
        })


if __name__ == "__main__":
    main()
