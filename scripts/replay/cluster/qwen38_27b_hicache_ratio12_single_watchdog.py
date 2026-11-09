#!/usr/bin/env python3
"""Silent watchdog for the isolated node2701 HiCache ratio=1.2 run."""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from . import qwen38_27b_single_node_1p3d_watchdog as watchdog
else:
    import qwen38_27b_single_node_1p3d_watchdog as watchdog


ROOT = Path(__file__).resolve().parents[3]
CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "q38-27b-hicache-ratio12-c192-node2701-20261018-r1"
)
watchdog.CONTROL_DIR = CONTROL_DIR
watchdog.STATE_PATH = CONTROL_DIR / "state.json"
watchdog.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
watchdog.WATCH_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
watchdog.LOCK_PATH = CONTROL_DIR / "watchdog.lock"
watchdog.LOG_PATH = ROOT / "q38_27b_hicache_ratio12_c192_node2701_watchdog.log"
watchdog.CONTROLLER_PATTERN = "qwen38_27b_hicache_ratio12_single_run.py"

_base_node_snapshot = watchdog.node_snapshot
_base_anomaly_key = watchdog.anomaly_key
_base_queue_alert = watchdog.queue_alert
_base_sample = watchdog.sample


def node_snapshot(node: str):
    snapshot, error = _base_node_snapshot(node)
    if snapshot:
        return snapshot, ""
    remote = (
        "gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
        "2>/dev/null | sed '/^$/d' | wc -l); "
        "ports=$(ss -H -ltn | awk '$4 ~ /:(242[0-9][0-9]|243[0-9][0-9]|"
        "244[0-9][0-9]|251[0-9][0-9])$/ {n++} END {print n+0}'); "
        "workload=$(pgrep -af 'dualpd.inference.py' | grep -v 'pgrep -af' | wc -l); "
        "helpers=$(ps -u $(id -u) -o comm= | "
        "awk '$1 ~ /^(apptainer|appinit|starter|starter-suid|squashfuse_ll|fuse2fs|faked-sysv)$/ "
        "{n++} END {print n+0}'); "
        "mem=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo); "
        "threads=$(ps -u $(id -u) -L --no-headers 2>/dev/null | wc -l); "
        "printf '%s %s %s %s %s %s\\n' \"$gpu\" \"$ports\" \"$workload\" "
        "\"$helpers\" \"$mem\" \"$threads\""
    )
    result = watchdog.run(
        [
            "srun", "--overlap", "--exact", "--jobid", "26049949",
            "-N1", "-n1", "-c1", "-w", node, "bash", "-lc", remote,
        ],
        timeout=90,
    )
    fields = result.stdout.strip().split()
    if result.returncode != 0 or len(fields) != 6 or not all(x.isdigit() for x in fields):
        detail = result.stderr.strip() or repr(result.stdout)
        return {}, f"ssh probe failed ({error}); srun fallback failed ({detail})"
    names = ("gpu", "ports", "workload", "helpers", "mem_available_kib", "threads")
    return dict(zip(names, map(int, fields))), ""


def anomaly_key(state, allocation_state, controllers, snapshot, probe_error, counters):
    status = str(state.get("status", ""))
    if status == "complete":
        return None
    adjusted = dict(snapshot)
    available = int(adjusted.get("mem_available_kib", 0))
    if 128 * 1024 * 1024 <= available < 256 * 1024 * 1024:
        adjusted["mem_available_kib"] = 256 * 1024 * 1024
    adjusted_state = dict(state)
    if int(adjusted_state.get("progress", 0)) >= 450:
        adjusted_state["progress"] = 500
    issue = _base_anomaly_key(
        adjusted_state, allocation_state, controllers, adjusted,
        probe_error, counters,
    )
    if issue is None:
        return None
    key, message = issue
    return key, message.replace("/500", "/450").replace(
        "已有500条terminal记录", "已有450条terminal记录"
    )


def queue_alert(thread_id: str, key: str, message: str):
    text = (
        f"[Qwen3.8-27B c192 HiCache ratio1.2 node2701 watchdog:{key}] "
        f"{message} 请检查job 26049949/node2701；不要修改baseline实现，也不要影响node1800。"
    )
    result = watchdog.run(
        ["codex", "queue", "--thread", thread_id, "--message", text],
        timeout=60,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def sample(thread_id: str, dry_run: bool) -> bool:
    if watchdog.STATE_PATH.is_file():
        try:
            if json.loads(watchdog.STATE_PATH.read_text()).get("status") == "complete":
                watchdog.append_log({"state": "complete", "issue": None, "watchdog_exit": True})
                return True
        except (OSError, json.JSONDecodeError):
            pass
    return _base_sample(thread_id, dry_run)


watchdog.node_snapshot = node_snapshot
watchdog.anomaly_key = anomaly_key
watchdog.queue_alert = queue_alert
watchdog.sample = sample


if __name__ == "__main__":
    raise SystemExit(watchdog.main())
