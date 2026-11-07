#!/usr/bin/env python3
"""Kill only runaway SWE tool commands caused by materializing prop_cycler.

This is a narrow, run-scoped safety guard for a known pathological generated
test.  It only considers descendants of the requested inference workload and
only kills a process group when the command contains both ``prop_cycler`` and
``list(`` and its resident set exceeds the configured limit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def proc_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return None


def proc_rss_kib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return 0


def is_descendant(pid: int, roots: set[int]) -> bool:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        parent = proc_ppid(pid)
        if parent is None:
            return False
        pid = parent
    return False


def matching_command_in_ancestry(pid: int, roots: set[int]) -> str | None:
    """Return the pathological command from pid or one of its ancestors."""
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        command = proc_cmdline(pid)
        if "prop_cycler" in command and "list(" in command:
            return command
        if pid in roots:
            return None
        parent = proc_ppid(pid)
        if parent is None:
            return None
        pid = parent
    return None


def append_event(path: Path, **event: object) -> None:
    event = {"epoch": time.time(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rss-limit-gib", type=float, default=8.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--missing-workload-grace-seconds", type=float, default=120.0)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.run_dir / "prop-cycler-guard.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0

    log_path = args.run_dir / "prop-cycler-guard.jsonl"
    rss_limit_kib = int(args.rss_limit_gib * 1024 * 1024)
    workload_missing_since: float | None = None
    append_event(
        log_path,
        kind="started",
        run_id=args.run_id,
        rss_limit_kib=rss_limit_kib,
    )

    while not (args.run_dir / "T450_RESULT.json").exists():
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
        roots = {
            pid
            for pid in pids
            if args.run_id in proc_cmdline(pid) and "dualpd.inference.py" in proc_cmdline(pid)
        }
        now = time.monotonic()
        if roots:
            workload_missing_since = None
        elif workload_missing_since is None:
            workload_missing_since = now
        elif now - workload_missing_since >= args.missing_workload_grace_seconds:
            append_event(log_path, kind="stopped", reason="workload_missing")
            return 0

        for pid in pids:
            if not is_descendant(pid, roots):
                continue
            rss_kib = proc_rss_kib(pid)
            if rss_kib < rss_limit_kib:
                continue
            command = matching_command_in_ancestry(pid, roots)
            if command is None:
                continue
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            append_event(
                log_path,
                kind="kill",
                pid=pid,
                pgid=pgid,
                rss_kib=rss_kib,
                command=command[:4096],
            )
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            time.sleep(1.0)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(args.interval_seconds)

    append_event(log_path, kind="stopped", reason="result_sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
