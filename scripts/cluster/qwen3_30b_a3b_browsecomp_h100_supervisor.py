#!/usr/bin/env python3
"""Supervise the Qwen3-30B-A3B matrix and retry startup port collisions only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-30b-a3b-tp2-12-matrix-20261012"
)
STATE = CONTROL / "state.json"
CONTROLLER = Path(
    "/home/shangy/dualpd/DualPD/scripts/cluster/"
    "qwen3_30b_a3b_browsecomp_h100_matrix.py"
)
PYTHON = Path("/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python")
LOG = CONTROL / "supervisor.log"
HEARTBEAT = CONTROL / "supervisor-heartbeat.json"
MAX_PORT_RETRIES = 10
EXPECTED_CASES = 12


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def log(message: str) -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(f"{time.time():.3f} {message}\n")
        handle.flush()


def states() -> dict[str, object]:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=30,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collision_in(path: Path) -> bool:
    candidates = [path] if path.is_file() else [path / "launcher.log", *(path / "logs").glob("*.log")]
    markers = ("address already in use", "errno 98", "configured ports remained occupied")
    return any(
        candidate.exists()
        and any(marker in candidate.read_text(errors="replace").lower() for marker in markers)
        for candidate in candidates
    )


def latest_failed(current: dict[str, object]) -> tuple[str, dict[str, object]] | None:
    failed = [
        (key, item)
        for key, item in current.items()
        if isinstance(item, dict) and item.get("status") == "failed"
    ]
    return max(failed, key=lambda pair: float(pair[1].get("ended", 0))) if failed else None


def wait_gpu_drain() -> None:
    for _ in range(30):
        occupied = gpu_processes()
        if not occupied:
            return
        log(f"waiting for GPU processes to drain: {occupied}")
        time.sleep(10)
    raise RuntimeError("GPU processes did not drain after 300 seconds")


def archive_failed(directory: Path, attempt: int) -> None:
    if not directory.exists():
        return
    destination = directory.with_name(f"{directory.name}-startup-port-retry-{attempt}")
    if destination.exists():
        raise RuntimeError(f"retry archive already exists: {destination}")
    shutil.move(str(directory), str(destination))
    log(f"archived failed startup as {destination}")


def main() -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    log(f"supervisor started pid={os.getpid()}")
    child: subprocess.Popen[bytes] | None = None

    def stop(_signum: int, _frame: object) -> None:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for attempt in range(MAX_PORT_RETRIES + 1):
        controller_log = CONTROL / f"controller-attempt-{attempt}.log"
        log(f"starting controller attempt={attempt}")
        with controller_log.open("a") as output:
            child = subprocess.Popen(
                [str(PYTHON), str(CONTROLLER)],
                cwd="/home/shangy/dualpd",
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while child.poll() is None:
                current = states()
                write_json(
                    HEARTBEAT,
                    {
                        "ts": time.time(),
                        "attempt": attempt,
                        "controller_pid": child.pid,
                        "complete": sum(
                            isinstance(item, dict) and item.get("status") == "complete"
                            for item in current.values()
                        ),
                    },
                )
                time.sleep(60)
        current = states()
        complete = sum(
            isinstance(item, dict) and item.get("status") == "complete"
            for item in current.values()
        )
        if child.returncode == 0 and complete == EXPECTED_CASES:
            log("matrix complete")
            write_json(
                HEARTBEAT,
                {"ts": time.time(), "status": "complete", "complete": EXPECTED_CASES},
            )
            return 0

        failed = latest_failed(current)
        failed_directory = Path(failed[1]["run_dir"]) if failed and failed[1].get("run_dir") else None
        collision = collision_in(controller_log) or (
            failed_directory is not None and collision_in(failed_directory)
        )
        if not collision:
            log(
                f"controller exited rc={child.returncode} complete={complete}; "
                f"non-port failure={failed}; stopping fail-closed"
            )
            return child.returncode or 1
        if attempt >= MAX_PORT_RETRIES:
            log("startup port retry limit reached")
            return 1
        wait_gpu_drain()
        if failed_directory is not None:
            archive_failed(failed_directory, attempt + 1)
        log("recoverable startup port collision detected; retrying")
        time.sleep(15)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
