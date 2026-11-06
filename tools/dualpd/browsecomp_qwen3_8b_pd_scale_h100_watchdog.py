#!/usr/bin/env python3
"""Quiet 10-minute watchdog for the Qwen3-8B P:D scaling matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-pd-scale-12-matrix-20261021"
)
STATE = CONTROL / "state.json"
HEARTBEAT = CONTROL / "heartbeat.json"
WATCH_STATE = CONTROL / "watchdog-state.json"
WATCH_LOG = CONTROL / "watchdog.jsonl"
CODEX = Path("/home/shangy/.local/bin/codex")
EXPECTED = 12


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def append_event(value: dict[str, Any]) -> None:
    with WATCH_LOG.open("a") as stream:
        stream.write(json.dumps({"epoch": time.time(), **value}, sort_keys=True) + "\n")


def completed_count(state: dict[str, Any]) -> int:
    return sum(isinstance(x, dict) and x.get("status") == "complete" for x in state.values())


def allocation(job_id: str) -> tuple[str, str]:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T|%N"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30, check=False,
    )
    line = result.stdout.strip().splitlines()
    if not line:
        return "MISSING", ""
    fields = line[0].split("|", 1)
    return fields[0], fields[1] if len(fields) > 1 else ""


def node_probe(job_id: str, node: str) -> dict[str, Any]:
    program = r'''import json,os,subprocess
gpu=[x for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()]
ports=set()
for line in subprocess.check_output(['ss','-H','-ltn'],text=True).splitlines():
 try: ports.add(int(line.split()[3].rsplit(':',1)[1]))
 except (IndexError,ValueError): pass
helpers=[]; workloads=closed_loop=0
for entry in os.listdir('/proc'):
 if not entry.isdigit(): continue
 try:
  comm=open('/proc/'+entry+'/comm').read().strip()
  if comm in {'apptainer','squashfuse_ll','fuse-overlayfs'}: helpers.append(comm)
  cmd=open('/proc/'+entry+'/cmdline','rb').read().split(b'\0')
  if any(x == b'dualpd.inference.py' or x.endswith(b'/inference.py') for x in cmd):
   workloads+=1
   if b'--closed-loop' in cmd: closed_loop+=1
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
mem={}
for line in open('/proc/meminfo'):
 key,value,*_=line.replace(':','').split()
 if key in {'MemAvailable','Committed_AS','CommitLimit'}: mem[key]=int(value)
print(json.dumps({'gpu_processes':len(gpu),'ports':sorted(ports),'helpers':len(helpers),'workloads':workloads,'closed_loop_workloads':closed_loop,'mem_available_kib':mem.get('MemAvailable',0),'commit_ratio':mem.get('Committed_AS',0)/max(mem.get('CommitLimit',1),1)}))'''
    result = subprocess.run(
        ["srun", f"--jobid={job_id}", "--overlap", "--exact", "--nodes=1",
         "--ntasks=1", "--cpus-per-task=1", "--mem=1G", "-w", node,
         "python3", "-c", program],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"node probe rc={result.returncode}: {result.stdout[-1200:]}")
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"node probe returned no JSON: {result.stdout[-1200:]}")


def queue(thread: str, message: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False,
    )
    return {"returncode": result.returncode, "output": result.stdout[-1200:]}


def current_item(state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    active = [
        (key, item) for key, item in state.items()
        if isinstance(item, dict) and item.get("status") in {"running", "failed"}
    ]
    return max(
        active,
        key=lambda pair: float(pair[1].get("started", pair[1].get("ended", 0))),
    ) if active else None


def expected_gpu_processes(key: str) -> int:
    if "-4p4d-" in key:
        return 8
    if "-2p2d-" in key:
        return 4
    if "-1p1d-" in key:
        return 2
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--thread",
        default=os.environ.get("CODEX_THREAD_ID"),
        help="Codex thread that receives alerts (defaults to CODEX_THREAD_ID)",
    )
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    if not args.thread:
        parser.error("--thread is required when CODEX_THREAD_ID is unset")
    CONTROL.mkdir(parents=True, exist_ok=True)
    previous = read_json(WATCH_STATE)
    append_event({"event": "watchdog_started", "pid": os.getpid(), "job_id": args.job_id})
    while True:
        now = time.time()
        state = read_json(STATE)
        complete = completed_count(state)
        if complete == EXPECTED:
            event = {"event": "complete", "complete": complete, "job_id": args.job_id}
            append_event(event)
            atomic_json(WATCH_STATE, {"ts": now, **event})
            queue(
                args.thread,
                "BrowseComp Qwen3-8B H100 P:D缩放消融已完成12/12。请核验独立MD与最终GPU、端口、Apptainer/FUSE清理门禁。",
            )
            return 0

        errors: list[str] = []
        status, node = allocation(args.job_id)
        item = current_item(state)
        hb = read_json(HEARTBEAT)
        hb_age = now - float(hb.get("ts", 0)) if hb else float("inf")
        probe: dict[str, Any] = {}

        if status == "PENDING":
            # Pending is not an experiment failure; record quietly until a node
            # is assigned, then enforce all runtime liveness checks.
            pass
        elif status != "RUNNING":
            errors.append(f"allocation={status}")
        else:
            if hb_age > 300:
                errors.append(f"controller heartbeat age={hb_age:.0f}s")
            if item and item[1].get("status") == "failed":
                errors.append(f"case failed: {item[0]}: {item[1].get('error', '')}")
            try:
                probe = node_probe(args.job_id, node)
                if int(probe.get("workloads", 0)) and (
                    int(probe.get("closed_loop_workloads", 0)) != int(probe.get("workloads", 0))
                ):
                    errors.append("an inference workload is missing --closed-loop")
                if item and item[1].get("status") == "running":
                    key, detail = item
                    started = float(detail.get("started", now))
                    run_directory = Path(str(detail.get("run_dir", "")))
                    boundary_path = run_directory / "closed_loop_boundaries.json"
                    boundary = read_json(boundary_path)
                    measurement_complete = boundary.get("status") == "steady_churn"
                    if now - started > 900 and not measurement_complete:
                        minimum = expected_gpu_processes(key)
                        if int(probe.get("gpu_processes", 0)) < minimum:
                            errors.append(
                                f"only {probe.get('gpu_processes', 0)} GPU processes; expected >= {minimum}"
                            )
                        live_files = [
                            run_directory / "engine_metrics.jsonl",
                            run_directory / "requests.jsonl",
                            run_directory / "closed_loop_events.jsonl",
                            run_directory / "dualpd.inference.log",
                            *(run_directory / "logs").glob("*.log"),
                        ]
                        mtimes = [p.stat().st_mtime for p in live_files if p.exists()]
                        if not mtimes:
                            errors.append("runtime telemetry was not created after startup grace")
                        elif now - max(mtimes) > 900:
                            errors.append(f"no runtime telemetry growth for {now-max(mtimes):.0f}s")
                    if measurement_complete and now - boundary_path.stat().st_mtime > 900:
                        errors.append("completed measurement stuck in postprocessing for >900s")
                if int(probe.get("helpers", 0)) and int(probe.get("gpu_processes", 0)) == 0:
                    errors.append(f"orphan Apptainer/FUSE helpers={probe['helpers']}")
                if int(probe.get("mem_available_kib", 0)) < 256 * 1024 * 1024:
                    errors.append(f"MemAvailable={probe['mem_available_kib']} KiB")
            except Exception as exc:
                probe = {"error": repr(exc)}
                errors.append(str(exc))

        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest() if errors else "healthy"
        event = {
            "event": "alert" if errors else ("pending" if status == "PENDING" else "healthy"),
            "complete": complete, "errors": errors, "heartbeat_age": hb_age,
            "allocation": status, "node": node, "probe": probe,
        }
        append_event(event)
        if errors and fingerprint != previous.get("fingerprint"):
            event["queue"] = queue(
                args.thread,
                f"WATCHDOG ALERT: {node or 'unassigned'} 的 BrowseComp Qwen3-8B P:D缩放矩阵异常："
                + "; ".join(errors)
                + "。请立即检查、修复并从当前组恢复；不要跳过未完成的300+1200窗口。",
            )
        previous = {"ts": now, "fingerprint": fingerprint, **event}
        atomic_json(WATCH_STATE, previous)
        time.sleep(max(args.interval_seconds, 60))


if __name__ == "__main__":
    raise SystemExit(main())
