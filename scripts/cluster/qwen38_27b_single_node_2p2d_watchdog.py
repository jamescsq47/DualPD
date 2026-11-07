#!/usr/bin/env python3
"""Silent watchdog binding for the Qwen3.8-27B 2P:2D matrix."""

from pathlib import Path

import qwen38_27b_single_node_1p3d_watchdog as watchdog


ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "qwen38-27b-tp2-single-node-2p2d-c96c128c192-20261016-r2"
)

watchdog.CONTROL_DIR = CONTROL_DIR
watchdog.STATE_PATH = CONTROL_DIR / "state.json"
watchdog.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
watchdog.WATCH_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
watchdog.LOCK_PATH = CONTROL_DIR / "watchdog.lock"
watchdog.LOG_PATH = ROOT / "qwen38_27b_single_node_2p2d_watchdog.log"
watchdog.CONTROLLER_PATTERN = "qwen38_27b_single_node_2p2d_matrix_autopilot.py"

_base_node_snapshot = watchdog.node_snapshot


def node_snapshot(node: str) -> tuple[dict[str, int], str]:
    """Fall back to the live Slurm allocation when node SSH is saturated."""
    try:
        snapshot, error = _base_node_snapshot(node)
    except Exception as exc:
        snapshot, error = {}, f"{type(exc).__name__}: {exc}"
    if snapshot:
        return snapshot, ""

    remote = (
        "gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
        "2>/dev/null | sed '/^$/d' | wc -l); "
        "ports=$(ss -H -ltn | awk '$4 ~ /:(239[0-9][0-9]|240[0-9][0-9]|242[0-9][0-9]|"
        "243[0-9][0-9]|244[0-9][0-9]|245[0-9][0-9]|251[0-9][0-9]|619[0-9][0-9]|"
        "620[0-9][0-9])$/ {n++} END {print n+0}'); "
        "workload=$(pgrep -af 'dualpd.inference.py' | grep -v 'pgrep -af' | wc -l); "
        "helpers=$(ps -u $(id -u) -o comm= | "
        "awk '$1 ~ /^(apptainer|appinit|starter|starter-suid|squashfuse_ll|fuse2fs|faked-sysv)$/ "
        "{n++} END {print n+0}'); "
        "mem=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo); "
        "threads=$(ps -u $(id -u) -L --no-headers 2>/dev/null | wc -l); "
        "printf '%s %s %s %s %s %s\\n' \"$gpu\" \"$ports\" \"$workload\" "
        "\"$helpers\" \"$mem\" \"$threads\""
    )
    try:
        result = watchdog.run(
            [
                "srun", "--overlap", "--jobid", "25642154", "-N1", "-n1",
                "-w", node, "bash", "-lc", remote,
            ],
            timeout=90,
        )
    except Exception as exc:
        return {}, f"ssh probe failed ({error}); srun fallback failed ({type(exc).__name__}: {exc})"
    fields = result.stdout.strip().split()
    if result.returncode != 0 or len(fields) != 6 or not all(value.isdigit() for value in fields):
        detail = result.stderr.strip() or repr(result.stdout)
        return {}, f"ssh probe failed ({error}); srun fallback failed ({detail})"
    names = ("gpu", "ports", "workload", "helpers", "mem_available_kib", "threads")
    return dict(zip(names, map(int, fields))), ""


watchdog.node_snapshot = node_snapshot

_base_anomaly_key = watchdog.anomaly_key
TARGET_TERMINALS = 450


def anomaly_key(state, allocation_state, controllers, snapshot, probe_error, counters):
    """Apply this matrix's memory margin and T450 completion threshold."""
    adjusted = dict(snapshot)
    available = int(adjusted.get("mem_available_kib", 0))
    if 128 * 1024 * 1024 <= available < 256 * 1024 * 1024:
        adjusted["mem_available_kib"] = 256 * 1024 * 1024
    adjusted_state = dict(state)
    status = str(adjusted_state.get("status", ""))
    if status == "complete":
        return "complete", "9组T450矩阵状态已完成。"
    if int(adjusted_state.get("progress", 0)) >= TARGET_TERMINALS:
        # The shared watchdog uses 500 as its completion-transition sentinel.
        adjusted_state["progress"] = 500
    issue = _base_anomaly_key(
        adjusted_state,
        allocation_state,
        controllers,
        adjusted,
        probe_error,
        counters,
    )
    if issue is None:
        return None
    key, message = issue
    message = (
        message.replace("已有500条terminal记录", "已有450条terminal记录")
        .replace("/500", "/450")
        .replace("12组", "9组T450")
    )
    return key, message


watchdog.anomaly_key = anomaly_key


if __name__ == "__main__":
    raise SystemExit(watchdog.main())
