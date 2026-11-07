#!/usr/bin/env python3
"""Silent watchdog for the isolated c128 HiCache ratio=1.5 run."""

from pathlib import Path

import qwen38_27b_hicache_ratio12_c128_node1800_watchdog as configured


watchdog = configured.watchdog
CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "q38-27b-hicache-ratio15-c128-node1800-control"
)
watchdog.CONTROL_DIR = CONTROL_DIR
watchdog.STATE_PATH = CONTROL_DIR / "state.json"
watchdog.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
watchdog.WATCH_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
watchdog.LOCK_PATH = CONTROL_DIR / "watchdog.lock"
watchdog.LOG_PATH = Path(
    "/home/shangy/dualpd/"
    "qwen38_27b_hicache_ratio15_c128_node1800_watchdog.log"
)
watchdog.CONTROLLER_PATTERN = "qwen38_27b_hicache_ratio15_c128_node1800.py"


def queue_alert(thread_id, key, message):
    text = (
        f"[Qwen3.8-27B c128 HiCache ratio15 watchdog:{key}] {message} "
        "请立即检查node1800/job26049949并恢复该独立实验；不要修改baseline源码，"
        "不要影响node2701或node2700主矩阵。正常时无需向用户输出。"
    )
    result = watchdog.run(
        ["codex", "queue", "--thread", thread_id, "--message", text], timeout=60
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


watchdog.queue_alert = queue_alert


if __name__ == "__main__":
    raise SystemExit(watchdog.main())
