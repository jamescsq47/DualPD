#!/usr/bin/env python3
"""Quiet watchdog for the seven-day PR #45423 H100 allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path("/orcd/scratch/orcd/003/shangy/dualpd/ppd45423")
STATE = CONTROL / "allocation-watchdog-state.json"
LOG = CONTROL / "logs" / "allocation-watchdog.jsonl"
CODEX = Path("/home/shangy/.local/bin/codex")


def query(job_id: str) -> tuple[str, str, str]:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T|%N|%L"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    lines = result.stdout.strip().splitlines()
    if not lines:
        return "MISSING", "", ""
    fields = lines[0].split("|", 2)
    return tuple((fields + ["", ""])[:3])  # type: ignore[return-value]


def queue(thread: str, message: str) -> None:
    subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )


def write_event(event: dict[str, object]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    temporary.replace(STATE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    previous = ""

    while True:
        status, node, remaining = query(args.job_id)
        errors = [] if status == "RUNNING" else [f"allocation={status}"]
        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest()
        event = {
            "epoch": time.time(),
            "job_id": args.job_id,
            "status": status,
            "node": node,
            "remaining": remaining,
            "errors": errors,
            "fingerprint": fingerprint,
        }
        write_event(event)
        if errors and fingerprint != previous:
            queue(
                args.thread,
                "PPD #45423 LAB ALERT: "
                f"7-day allocation {args.job_id} is {status}; node={node or 'none'}, "
                f"remaining={remaining or 'unknown'}. Please inspect and renew it.",
            )
        previous = fingerprint
        time.sleep(max(args.interval_seconds, 60))


if __name__ == "__main__":
    raise SystemExit(main())
