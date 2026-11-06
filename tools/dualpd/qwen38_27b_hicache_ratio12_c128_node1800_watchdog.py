#!/usr/bin/env python3
"""Silent watchdog for the isolated c128 HiCache ratio=1.2 run."""

from pathlib import Path

import qwen38_27b_single_node_1p3d_watchdog as watchdog


ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "q38-27b-hicache-ratio12-c128-node1800-r2-control"
)

watchdog.CONTROL_DIR = CONTROL_DIR
watchdog.STATE_PATH = CONTROL_DIR / "state.json"
watchdog.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
watchdog.WATCH_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
watchdog.LOCK_PATH = CONTROL_DIR / "watchdog.lock"
watchdog.LOG_PATH = ROOT / "qwen38_27b_hicache_ratio12_c128_node1800_watchdog.log"
watchdog.CONTROLLER_PATTERN = "qwen38_27b_hicache_ratio12_c128_node1800.py"


def node_snapshot(node):
    remote = (
        "gpu=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
        "2>/dev/null | sed '/^$/d' | wc -l); "
        "ports=$(ss -H -ltn | awk '$4 ~ /:(242[0-9][0-9]|243[0-9][0-9]|"
        "244[0-9][0-9]|245[0-9][0-9]|251[0-9][0-9])$/ {n++} END {print n+0}'); "
        "workload=$(pgrep -af 'dualpd.inference.py' | grep -v 'pgrep -af' | wc -l); "
        "helpers=$(ps -u $(id -u) -o comm= | awk "
        "'$1 ~ /^(apptainer|appinit|starter|starter-suid|squashfuse_ll|fuse2fs|faked-sysv)$/ "
        "{n++} END {print n+0}'); "
        "mem=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo); "
        "threads=$(ps -u $(id -u) -L --no-headers 2>/dev/null | wc -l); "
        "printf '%s %s %s %s %s %s\\n' \"$gpu\" \"$ports\" \"$workload\" "
        "\"$helpers\" \"$mem\" \"$threads\""
    )
    result = watchdog.run(
        [
            "srun", "--overlap", "--jobid", "26049949", "-N1", "-n1",
            "-w", node, "bash", "-lc", remote,
        ],
        timeout=90,
    )
    fields = result.stdout.strip().split()
    if result.returncode != 0 or len(fields) != 6 or not all(v.isdigit() for v in fields):
        return {}, result.stderr.strip() or repr(result.stdout)
    names = ("gpu", "ports", "workload", "helpers", "mem_available_kib", "threads")
    return dict(zip(names, map(int, fields))), ""


watchdog.node_snapshot = node_snapshot
_base_anomaly_key = watchdog.anomaly_key


def anomaly_key(state, allocation_state, controllers, snapshot, probe_error, counters):
    adjusted = dict(state)
    if adjusted.get("status") == "complete":
        return "complete", "c128 HiCache ratio=1.2 T450实验已完成，请封存结果。"
    if int(adjusted.get("progress", 0)) >= 450:
        adjusted["progress"] = 500
    issue = _base_anomaly_key(
        adjusted, allocation_state, controllers, snapshot, probe_error, counters
    )
    if issue is None:
        return None
    key, message = issue
    return key, message.replace("已有500条terminal记录", "已有450条terminal记录").replace("/500", "/450")


def queue_alert(thread_id, key, message):
    text = (
        f"[Qwen3.8-27B c128 HiCache ratio12 watchdog:{key}] {message} "
        "请立即检查node1800/job26049949并恢复该独立实验；不要修改baseline源码，"
        "不要影响node2701或主矩阵。正常时无需向用户输出。"
    )
    result = watchdog.run(
        ["codex", "queue", "--thread", thread_id, "--message", text], timeout=60
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


watchdog.anomaly_key = anomaly_key
watchdog.queue_alert = queue_alert


if __name__ == "__main__":
    raise SystemExit(watchdog.main())
