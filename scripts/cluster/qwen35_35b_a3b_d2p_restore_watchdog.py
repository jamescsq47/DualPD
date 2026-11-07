#!/usr/bin/env python3
"""Quietly wake the owning Codex thread if the D-to-P tuning run stops."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "q35-35b-a3b-d2p-restore-ft30-p2dl3-d2pl4-r7-20261016"
)
STATE = CONTROL / "state.json"
WATCH_STATE = CONTROL / "watchdog-state.json"
CODEX = Path("/home/shangy/.local/bin/codex")
DEFAULT_THREAD = "01a13b01-e01f-7f83-adde-b03053ded355"
CONTROLLER_NEEDLE = "qwen35_35b_a3b_d2p_restore_tuning.py"


def save(value: dict) -> None:
    temporary = WATCH_STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, WATCH_STATE)


def controller_count() -> int:
    count = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            command = Path("/proc", entry, "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if CONTROLLER_NEEDLE.encode() in command:
            count += 1
    return count


def queue(thread: str, message: str) -> None:
    subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd="/home/shangy/dualpd",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD)
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    last_event = ""
    missing_since = time.time()
    while True:
        try:
            state = json.loads(STATE.read_text())
            missing_since = time.time()
            status = str(state.get("status"))
            event = ""
            message = ""
            if status == "complete":
                event = "complete"
                message = (
                    "D2P RESTORE TUNING COMPLETE: node2800 c128试验已到T450并清理完成。"
                    "请读取t450-result.json，比较HiCache 1861.40 tok/s并继续修复或记录。"
                )
            elif status == "failed":
                event = "failed:" + str(state.get("error"))
                message = (
                    "D2P RESTORE TUNING ALERT: node2800试验失败："
                    + str(state.get("error"))
                    + "。请立即检查、修复并恢复。"
                )
            elif status == "running":
                age = time.time() - float(state.get("last_poll_epoch") or 0)
                progress_age = time.time() - float(
                    state.get("last_progress_epoch") or 0
                )
                controllers = controller_count()
                if controllers != 1 or age > 300 or progress_age > 1200:
                    event = (
                        f"controller:{controllers}:age:{int(age)}:"
                        f"progress_age:{int(progress_age)}"
                    )
                    message = (
                        "D2P RESTORE TUNING ALERT: 控制器异常，"
                        f"count={controllers}, last_poll_age={int(age)}s, "
                        f"last_progress_age={int(progress_age)}s。"
                        "请立即检查、修复并恢复。"
                    )
            if event and event != last_event:
                queue(args.thread_id, message)
                last_event = event
            save(
                {
                    "checked_at_epoch": time.time(),
                    "matrix_status": status,
                    "progress": state.get("progress"),
                    "last_event": last_event,
                }
            )
            if status == "complete":
                return 0
        except FileNotFoundError:
            # The controller and watchdog are launched back-to-back.  The state
            # file can legitimately be absent until the controller has acquired
            # its allocation and published the initial state.  Treat that as a
            # startup condition, and only alert if both the controller is gone
            # and the state has remained absent for five minutes.
            controllers = controller_count()
            missing_age = time.time() - missing_since
            event = ""
            if controllers != 1 and missing_age > 300:
                event = f"state-missing:controller:{controllers}"
                if event != last_event:
                    queue(
                        args.thread_id,
                        "D2P RESTORE TUNING ALERT: state.json持续缺失且"
                        f"控制器数量为{controllers}。请立即检查、修复并恢复。",
                    )
                    last_event = event
            save(
                {
                    "checked_at_epoch": time.time(),
                    "matrix_status": "state-pending",
                    "progress": None,
                    "last_event": last_event,
                }
            )
        except BaseException as error:
            event = f"watchdog:{type(error).__name__}:{error}"
            if event != last_event:
                queue(args.thread_id, "D2P RESTORE WATCHDOG自身异常：" + event)
                last_event = event
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
