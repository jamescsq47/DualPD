#!/usr/bin/env python3
"""Quiet watchdog for the Qwen3.5-35B-A3B node2800 matrix.

Healthy checks are written only to local JSONL/state files.  A new failure (or
matrix completion) is queued to the owning Codex thread, which wakes the agent
without spending a turn on routine heartbeat reports.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTROL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "qwen35-35b-a3b-tp2-single-node-1p3d-c64c128c192-growth128k-20261014"
)
STATE_PATH = CONTROL_DIR / "state.json"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
WATCHDOG_STATE_PATH = CONTROL_DIR / "watchdog-state.json"
WATCHDOG_LOG_PATH = CONTROL_DIR / "watchdog.jsonl"
WATCHDOG_LOCK_PATH = CONTROL_DIR / "watchdog.lock"
CONTROLLER = (
    ROOT / "scripts/cluster/qwen35_35b_a3b_single_node_1p3d_matrix_autopilot.py"
)
CODEX = Path("/home/shangy/.local/bin/codex")
DEFAULT_THREAD_ID = "01a13b01-e01f-7f83-adde-b03053ded355"
# One SWE verifier is allowed to run for 2400 seconds, and a terminal record is
# emitted only after verification.  Match the matrix controller's 90-minute
# stall policy so a legitimate long final verifier does not wake the agent.
NO_PROGRESS_TIMEOUT_SECONDS = 5400


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_event(event: dict[str, Any]) -> None:
    event = {"epoch": time.time(), **event}
    with WATCHDOG_LOG_PATH.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def controller_pids() -> list[int]:
    needle = str(CONTROLLER).encode()
    found: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            command = Path("/proc", entry, "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command:
            found.append(int(entry))
    return found


def allocation_status(job_id: str) -> tuple[str, str]:
    completed = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T|%N"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    line = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not line:
        return "MISSING", completed.stdout.strip()
    status, _, node = line[-1].partition("|")
    return status, node


def node_probe(
    job_id: str, node: str, required_ports: set[int], run_id: str
) -> dict[str, Any]:
    probe = r'''import json,os,subprocess,sys
required=set(map(int,sys.argv[1].split(','))) if sys.argv[1] else set()
needle=sys.argv[2].encode(); uid=os.getuid(); ports=set(); pids=threads=workloads=0
gpus=[x for x in subprocess.check_output(
 ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
 text=True).splitlines() if x.strip()]
for line in subprocess.check_output(['ss','-H','-ltn'],text=True).splitlines():
 try:ports.add(int(line.split()[3].rsplit(':',1)[1]))
 except (IndexError,ValueError):pass
for entry in os.listdir('/proc'):
 if not entry.isdigit():continue
 root='/proc/'+entry
 try:
  if os.stat(root).st_uid!=uid:continue
  pids+=1;threads+=len(os.listdir(root+'/task'))
  cmd=open(root+'/cmdline','rb').read()
  if needle in cmd and (b'dualpd.inference.py' in cmd or b'/workload' in cmd):workloads+=1
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
mem={}
for line in open('/proc/meminfo'):
 key,value,*_=line.replace(':','').split()
 if key in {'MemAvailable','CommitLimit','Committed_AS'}:mem[key]=int(value)
print(json.dumps({
 'gpu_processes':len(gpus),'missing_ports':sorted(required-ports),
 'user_pids':pids,'user_threads':threads,'workload_processes':workloads,
 'mem_available_kib':mem['MemAvailable'],
 'commit_ratio':mem['Committed_AS']/mem['CommitLimit']}))'''
    completed = subprocess.run(
        [
            "srun", "--jobid", job_id, "--overlap", "--exact", "--nodes=1",
            "--ntasks=1", "--cpus-per-task=1", "-w", node,
            "python3", "-c", probe,
            ",".join(map(str, sorted(required_ports))), run_id,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"node probe exited {completed.returncode}: {completed.stdout[-2000:]}"
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"node probe returned no JSON: {completed.stdout[-2000:]}")


def queue_message(thread_id: str, message: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(CODEX), "queue", "--thread", thread_id, "--message", message],
        cwd=ROOT.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "output": completed.stdout[-4000:],
    }


def matrix_snapshot() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts/cluster"))
    import qwen35_35b_a3b_single_node_1p3d_matrix_autopilot as matrix

    matrix.configure()
    state = read_json(STATE_PATH)
    snapshot: dict[str, Any] = {"matrix_state": state}
    if int(state["index"]) >= len(matrix.EXPERIMENTS):
        return snapshot
    experiment = matrix.EXPERIMENTS[int(state["index"])]
    rows = matrix.base.core.load_progress(matrix.base.run_dir(experiment))
    snapshot.update(
        experiment=experiment,
        records=len(rows),
        completed=sum(
            str(row.get("status")) == "Status.COMPLETED" for row in rows
        ),
        environment_errors=sum(
            bool(row.get("environment_error")) for row in rows
        ),
        resolved=sum(row.get("reward") == 1 for row in rows),
        expected_ports=sorted(matrix.base.expected_runtime_ports(experiment)),
    )
    if rows:
        snapshot["last_completed_at"] = rows[-1].get("completed_at")
    return snapshot


def check(previous: dict[str, Any]) -> tuple[dict[str, Any], list[str], bool]:
    now = time.time()
    snapshot = matrix_snapshot()
    state = snapshot["matrix_state"]
    allocation = read_json(ALLOCATION_PATH)
    job_id = str(allocation["job_id"])
    node = str(allocation["node"])
    slurm_status, slurm_node = allocation_status(job_id)
    controllers = controller_pids()
    snapshot.update(
        checked_at_epoch=now,
        job_id=job_id,
        node=node,
        slurm_status=slurm_status,
        slurm_node=slurm_node,
        controller_pids=controllers,
    )
    errors: list[str] = []
    complete = state.get("status") == "complete"
    if complete:
        return snapshot, errors, True
    if slurm_status != "RUNNING" or slurm_node != node:
        errors.append(
            f"allocation {job_id} is {slurm_status} on {slurm_node or 'no node'}"
        )
    if len(controllers) != 1:
        errors.append(f"expected one controller, found {len(controllers)}")
    if state.get("status") == "paused":
        errors.append(f"controller paused: {state.get('error', '')}")
    elif state.get("status") not in {"running", "ready"}:
        errors.append(f"unexpected matrix state {state.get('status')!r}")

    if "experiment" in snapshot and slurm_status == "RUNNING":
        try:
            probe = node_probe(
                job_id,
                node,
                set(snapshot["expected_ports"]),
                snapshot["experiment"]["run_id"],
            )
            snapshot["node_probe"] = probe
            records = int(snapshot["records"])
            # During the short cleanup/start handoff after 500 callbacks, the
            # old state can legitimately have zero GPUs and ports.  The live
            # controller is the authority during this bounded transition.
            in_handoff = (
                (records >= 500 or bool(snapshot.get("result_sealed")))
                and bool(controllers)
            )
            if not in_handoff:
                if probe["gpu_processes"] != 8:
                    errors.append(
                        f"expected 8 GPU processes, found {probe['gpu_processes']}"
                    )
                if probe["missing_ports"]:
                    errors.append(f"missing ports {probe['missing_ports']}")
                if probe["workload_processes"] < 1:
                    errors.append("workload process is missing")
            if probe["user_threads"] >= 100_000:
                errors.append(
                    f"thread gate reached: {probe['user_threads']} >= 100000"
                )
            if probe["mem_available_kib"] <= 256 * 1024 * 1024:
                errors.append(
                    "memory gate reached: "
                    f"{probe['mem_available_kib']} KiB <= 256 GiB"
                )
            last_progress = float(
                snapshot.get("last_completed_at")
                or state.get("last_progress_epoch")
                or state.get("last_started_epoch")
                or now
            )
            if (
                not in_handoff
                and now - last_progress > NO_PROGRESS_TIMEOUT_SECONDS
            ):
                errors.append(
                    f"no terminal progress for {int(now - last_progress)} seconds"
                )
        except BaseException as error:
            errors.append(f"node probe failed: {type(error).__name__}: {error}")
    return snapshot, errors, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = WATCHDOG_LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another matrix watchdog holds the lock")

    previous = (
        read_json(WATCHDOG_STATE_PATH) if WATCHDOG_STATE_PATH.is_file() else {}
    )
    while True:
        try:
            snapshot, errors, complete = check(previous)
        except BaseException as error:
            snapshot = {"checked_at_epoch": time.time()}
            errors = [f"watchdog check failed: {type(error).__name__}: {error}"]
            complete = False

        fingerprint = hashlib.sha256(
            "\n".join(errors).encode()
        ).hexdigest() if errors else ""
        event = {
            "kind": "complete" if complete else "alert" if errors else "healthy",
            "snapshot": snapshot,
            "errors": errors,
        }
        if complete and not previous.get("completion_queued"):
            event["queue"] = queue_message(
                args.thread_id,
                "WATCHDOG COMPLETE: Qwen3.5-35B-A3B 13组矩阵状态为 complete。"
                "请执行最终结果和 Markdown 完整性审计。",
            )
            previous["completion_queued"] = True
        elif errors and fingerprint != previous.get("active_alert_fingerprint"):
            checked_epoch = float(snapshot.get("checked_at_epoch", time.time()))
            checked_utc = time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime(checked_epoch)
            )
            experiment = snapshot.get("experiment") or {}
            run_id = str(experiment.get("run_id") or "unknown")
            message = (
                f"WATCHDOG ALERT [{checked_utc}; run={run_id}]: "
                "node2800 的 Qwen3.5-35B-A3B 矩阵检测到异常："
                + "; ".join(errors)
                + "。请立即检查、修复并恢复实验；恢复后确认 GPU、端口、"
                "控制器及 terminal 增长均正常。"
            )
            event["queue"] = queue_message(args.thread_id, message)
            previous["active_alert_fingerprint"] = fingerprint
        elif not errors:
            previous["active_alert_fingerprint"] = ""

        previous.update(
            last_check_epoch=snapshot.get("checked_at_epoch", time.time()),
            last_kind=event["kind"],
            last_snapshot=snapshot,
            last_errors=errors,
        )
        atomic_json(WATCHDOG_STATE_PATH, previous)
        append_event(event)
        if complete or args.once:
            return 0 if not errors else 1
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
