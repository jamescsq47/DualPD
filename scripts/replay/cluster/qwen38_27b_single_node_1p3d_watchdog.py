#!/usr/bin/env python3
"""Silent watchdog for the Qwen3.8-27B single-node matrix.

The matrix controller remains the owner of experiments and cleanup.  This
watchdog only samples durable state and node health.  It queues a message to
the owning Codex thread on a persistent anomaly or when the matrix completes;
normal samples are written locally and never sent to the conversation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "qwen38-27b-tp2-single-node-1p3d-c64c128c192-growth128k-20261014"
)
STATE_PATH = CONTROL_DIR / "state.json"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
WATCH_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
LOCK_PATH = CONTROL_DIR / "watchdog.lock"
LOG_PATH = ROOT / "qwen38_27b_single_node_1p3d_watchdog.log"
CONTROLLER_PATTERN = "qwen38_27b_single_node_1p3d_matrix_autopilot.py"

DEFAULT_INTERVAL_SECONDS = 300
PERSISTENT_FAILURE_SAMPLES = 2
NO_PROGRESS_WARNING_SECONDS = 4200
COMPLETION_TRANSITION_WARNING_SECONDS = 1800
ALERT_RETRY_SECONDS = 900


def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def append_log(event: dict[str, Any]) -> None:
    event = {"timestamp": now_utc(), **event}
    with LOG_PATH.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(command: list[str], timeout: int = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def controller_pids() -> list[int]:
    result = run(["pgrep", "-f", CONTROLLER_PATTERN], timeout=10)
    if result.returncode not in (0, 1):
        return []
    own_pid = os.getpid()
    return [
        int(value)
        for value in result.stdout.split()
        if value.isdigit() and int(value) != own_pid
    ]


def progress_count(run_id: str) -> int:
    path = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs") / run_id / "episode_progress.jsonl"
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def allocation_status(job_id: str) -> tuple[str, str]:
    result = run(["squeue", "-h", "-j", job_id, "-o", "%T|%N"], timeout=30)
    if result.returncode != 0:
        return "probe_error", result.stderr.strip()
    output = result.stdout.strip()
    if not output:
        return "missing", "allocation absent from squeue"
    state, _, node = output.partition("|")
    return state, node


def node_snapshot(node: str) -> tuple[dict[str, int], str]:
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
    result = run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, remote],
        timeout=45,
    )
    if result.returncode != 0:
        return {}, result.stderr.strip() or f"ssh exit {result.returncode}"
    fields = result.stdout.strip().split()
    if len(fields) != 6 or not all(value.isdigit() for value in fields):
        return {}, f"unexpected node probe output: {result.stdout!r}"
    names = ("gpu", "ports", "workload", "helpers", "mem_available_kib", "threads")
    return dict(zip(names, map(int, fields))), ""


def anomaly_key(
    state: dict[str, Any], allocation_state: str, controllers: list[int],
    snapshot: dict[str, int], probe_error: str, counters: dict[str, int],
) -> tuple[str, str] | None:
    status = str(state.get("status", "missing"))
    if status == "complete":
        return "complete", "12组矩阵状态已变为complete，请进行最终结果与文档审计。"
    if status == "paused" or state.get("error"):
        return "controller_paused", f"控制器暂停：{state.get('error') or 'unknown error'}"
    if allocation_state != "RUNNING":
        return "allocation_not_running", f"allocation状态异常：{allocation_state}"
    if not controllers:
        return "controller_missing", "矩阵控制器进程不存在。"
    if probe_error:
        if counters.get("node_probe", 0) >= PERSISTENT_FAILURE_SAMPLES:
            return "node_probe_failed", f"连续节点探测失败：{probe_error}"
        return None
    if status != "running":
        return "unexpected_status", f"矩阵状态不是running：{status}"

    progress = int(state.get("progress", 0))
    # At 500 the controller may be deliberately cleaning before launching the
    # next run, so service-count alerts are suppressed during that transition.
    if progress < 500:
        if snapshot.get("gpu") != 8 and counters.get("gpu", 0) >= PERSISTENT_FAILURE_SAMPLES:
            return "gpu_topology", f"连续检查GPU worker数为{snapshot.get('gpu')}，期望8。"
        if snapshot.get("ports", 0) == 0 and counters.get("ports", 0) >= PERSISTENT_FAILURE_SAMPLES:
            return "ports_missing", "连续检查未发现矩阵服务端口。"
        if snapshot.get("workload", 0) == 0 and counters.get("workload", 0) >= PERSISTENT_FAILURE_SAMPLES:
            return "workload_missing", "连续检查未发现workload进程。"
        last_progress = float(state.get("last_progress_epoch", time.time()))
        if time.time() - last_progress > NO_PROGRESS_WARNING_SECONDS:
            return (
                "progress_stalled",
                f"已有{int(time.time() - last_progress)}秒无新terminal episode，"
                f"当前{progress}/500。",
            )
    elif time.time() - float(state.get("last_progress_epoch", time.time())) > COMPLETION_TRANSITION_WARNING_SECONDS:
        return (
            "completion_transition_stalled",
            "当前run已有500条terminal记录，但控制器超过30分钟仍未完成清理和切组。",
        )
    if snapshot.get("mem_available_kib", 0) < 256 * 1024 * 1024:
        return "low_memory", f"MemAvailable仅{snapshot.get('mem_available_kib')} KiB。"
    if snapshot.get("threads", 0) > 100_000:
        return "thread_runaway", f"用户线程数达到{snapshot.get('threads')}。"
    return None


def queue_alert(thread_id: str, key: str, message: str) -> tuple[bool, str]:
    text = (
        f"[Qwen3.8-27B矩阵watchdog:{key}] {message} "
        "请立即检查node2700/当前allocation，修复后确保实验恢复运行；"
        "正常时无需向用户输出周期性状态。"
    )
    result = run(
        ["codex", "queue", "--thread", thread_id, "--message", text],
        timeout=60,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def sample(thread_id: str, dry_run: bool) -> bool:
    watch = load_json(WATCH_STATE_PATH) if WATCH_STATE_PATH.is_file() else {}
    counters = dict(watch.get("failure_counters", {}))
    try:
        state = load_json(STATE_PATH)
        allocation = load_json(ALLOCATION_PATH)
    except Exception as error:
        state = {"status": "missing", "error": f"state read failed: {error}"}
        allocation = {"job_id": "25642154", "node": "node2700"}

    job_id = str(allocation.get("job_id", "25642154"))
    node = str(allocation.get("node", "node2700"))
    alloc_state, alloc_detail = allocation_status(job_id)
    controllers = controller_pids()
    snapshot, probe_error = node_snapshot(node)
    current_run = str(state.get("current_run_id", ""))
    actual_progress = progress_count(current_run) if current_run else 0
    if actual_progress > int(state.get("progress", 0)):
        state["progress"] = actual_progress

    checks = {
        "node_probe": bool(probe_error),
        "gpu": bool(snapshot) and snapshot.get("gpu") != 8,
        "ports": bool(snapshot) and snapshot.get("ports", 0) == 0,
        "workload": bool(snapshot) and snapshot.get("workload", 0) == 0,
    }
    for name, failed in checks.items():
        counters[name] = counters.get(name, 0) + 1 if failed else 0

    issue = anomaly_key(
        state, alloc_state, controllers, snapshot,
        probe_error or (alloc_detail if alloc_state == "probe_error" else ""), counters,
    )
    event = {
        "state": state.get("status"),
        "run_id": current_run,
        "progress": state.get("progress", 0),
        "allocation": alloc_state,
        "controller_pids": controllers,
        "node": snapshot,
        "probe_error": probe_error,
        "issue": issue[0] if issue else None,
    }
    append_log(event)

    watch.update(
        last_sample_epoch=time.time(),
        last_event=event,
        failure_counters=counters,
    )
    should_exit = False
    if issue:
        key, message = issue
        last_key = watch.get("last_alert_key")
        last_epoch = float(watch.get("last_alert_epoch", 0))
        should_send = key != last_key or time.time() - last_epoch >= ALERT_RETRY_SECONDS
        sent = last_key == key and bool(last_epoch)
        if should_send:
            if dry_run:
                sent, detail = False, "dry-run"
            else:
                sent, detail = queue_alert(thread_id, key, message)
            append_log({"alert": key, "sent": sent, "detail": detail, "message": message})
            if sent:
                watch["last_alert_key"] = key
                watch["last_alert_epoch"] = time.time()
        should_exit = key == "complete" and sent
    else:
        watch["last_alert_key"] = None
    atomic_json(WATCH_STATE_PATH, watch)
    return should_exit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another matrix watchdog holds the lock")

    while True:
        try:
            finished = sample(args.thread_id, args.dry_run)
        except Exception as error:
            append_log({"watchdog_error": f"{type(error).__name__}: {error}"})
            finished = False
        if args.once or finished:
            return 0
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
