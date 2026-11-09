#!/usr/bin/env python3
"""Quiet watchdog for the node2800 Qwen3-8B BrowseComp low-c matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-lowc-12-matrix-20261018"
)
STATE = CONTROL / "state.json"
HEARTBEAT = CONTROL / "heartbeat.json"
WATCH_STATE = CONTROL / "watchdog-state.json"
WATCH_LOG = CONTROL / "watchdog.jsonl"
CONTROLLER = ROOT / "scripts/replay/cluster/browsecomp_qwen3_8b_lowc_h100_matrix.py"
CODEX = Path("/home/shangy/.local/bin/codex")
JOB_ID = "26078099"
NODE = "node2800"
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


def controller_processes() -> list[int]:
    needle = str(CONTROLLER).encode()
    found: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            command = Path("/proc", entry, "cmdline").read_bytes()
            fields = Path("/proc", entry, "stat").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if needle in command:
            found[int(entry)] = int(fields[3])
    # srun forks a helper with the same argv.  Count only roots of matching
    # process trees so one healthy Slurm step is not reported as two
    # controllers.
    return sorted(pid for pid, ppid in found.items() if ppid not in found)


def allocation() -> str:
    result = subprocess.run(
        ["squeue", "-h", "-j", JOB_ID, "-o", "%T|%N"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30, check=False,
    )
    return result.stdout.strip()


def node_probe() -> dict[str, Any]:
    program = r'''import json,os,subprocess,time
gpu=[x for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()]
ports=set()
for line in subprocess.check_output(['ss','-H','-ltn'],text=True).splitlines():
 try: ports.add(int(line.split()[3].rsplit(':',1)[1]))
 except (IndexError,ValueError): pass
helpers=[]; workloads=closed_loop_workloads=0
for entry in os.listdir('/proc'):
 if not entry.isdigit(): continue
 try:
  comm=open('/proc/'+entry+'/comm').read().strip()
  if comm in {'apptainer','squashfuse_ll','fuse-overlayfs'}: helpers.append(comm)
  cmd=open('/proc/'+entry+'/cmdline','rb').read().split(b'\0')
  if any(part == b'dualpd.inference.py' or part.endswith(b'/inference.py') for part in cmd):
   workloads+=1
   if b'--closed-loop' in cmd: closed_loop_workloads+=1
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
mem={}
for line in open('/proc/meminfo'):
 key,value,*_=line.replace(':','').split()
 if key in {'MemAvailable','Committed_AS','CommitLimit'}: mem[key]=int(value)
print(json.dumps({'gpu_processes':len(gpu),'ports':sorted(ports),'helpers':len(helpers),'workloads':workloads,'closed_loop_workloads':closed_loop_workloads,'mem_available_kib':mem.get('MemAvailable',0),'commit_ratio':mem.get('Committed_AS',0)/max(mem.get('CommitLimit',1),1)}))'''
    result = subprocess.run(
        ["srun", f"--jobid={JOB_ID}", "--overlap", "--exact", "--nodes=1",
         "--ntasks=1", "--cpus-per-task=1", "--mem=1G", "-w", NODE,
         "python3", "-c", program],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"node probe rc={result.returncode}: {result.stdout[-1500:]}")
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    raise RuntimeError(f"node probe returned no JSON: {result.stdout[-1500:]}")


def queue(thread: str, message: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False,
    )
    return {"returncode": result.returncode, "output": result.stdout[-2000:]}


def current_item(state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    active = [(k, v) for k, v in state.items() if isinstance(v, dict) and v.get("status") in {"running", "failed"}]
    return max(active, key=lambda pair: float(pair[1].get("started", pair[1].get("ended", 0)))) if active else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="01a13b01-e01f-7f83-adde-b03053ded355")
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    previous = read_json(WATCH_STATE)
    append_event({"event": "watchdog_started", "pid": os.getpid()})
    while True:
        now = time.time()
        state = read_json(STATE)
        complete = completed_count(state)
        if complete == EXPECTED:
            event = {"event": "complete", "complete": complete}
            append_event(event)
            atomic_json(WATCH_STATE, {"ts": now, **event})
            queue(args.thread, "BrowseComp Qwen3-8B H100低并发矩阵已完成12/12。请核验结果文档与node2800最终清理门禁。")
            return 0
        errors = []
        alloc = allocation()
        if alloc != f"RUNNING|{NODE}":
            errors.append(f"allocation={alloc or 'missing'}")
        controllers = controller_processes()
        if len(controllers) != 1:
            errors.append(f"expected one controller command, found {len(controllers)}")
        hb = read_json(HEARTBEAT)
        hb_age = now - float(hb.get("ts", 0)) if hb else float("inf")
        if hb_age > 300:
            errors.append(f"controller heartbeat age={hb_age:.0f}s")
        item = current_item(state)
        if item and item[1].get("status") == "failed":
            errors.append(f"case failed: {item[0]}: {item[1].get('error', '')}")
        try:
            probe = node_probe()
            started = float(item[1].get("started", now)) if item else now
            if int(probe.get("workloads", 0)) and (
                int(probe.get("closed_loop_workloads", 0))
                != int(probe.get("workloads", 0))
            ):
                errors.append(
                    "an inference workload is missing the --closed-loop flag"
                )
            if item and item[1].get("status") == "running" and now - started > 900:
                run_dir = Path(str(item[1].get("run_dir", "")))
                boundary_path = run_dir / "closed_loop_boundaries.json"
                boundary = read_json(boundary_path)
                measurement_complete = boundary.get("status") == "steady_churn"
                if measurement_complete:
                    # The launcher has intentionally released every GPU.  The
                    # controller still needs a short CPU-only analysis/render
                    # phase before changing state from running to complete.
                    boundary_age = now - boundary_path.stat().st_mtime
                    if boundary_age > 900:
                        errors.append(
                            f"completed measurement stuck in postprocessing for {boundary_age:.0f}s"
                        )
                else:
                    if int(probe["gpu_processes"]) < 8:
                        errors.append(f"only {probe['gpu_processes']} GPU processes after startup grace")
                    # This BrowseComp harness buffers per-request/event outputs until
                    # closed-loop boundaries.  During a healthy warmup/measurement
                    # window, engine_metrics.jsonl and the engine logs are the
                    # continuously-written runtime telemetry.
                    live_files = [
                        run_dir / "engine_metrics.jsonl",
                        run_dir / "requests.jsonl",
                        run_dir / "closed_loop_events.jsonl",
                        run_dir / "dualpd.inference.log",
                        *(run_dir / "logs").glob("*.log"),
                    ]
                    mtimes = [path.stat().st_mtime for path in live_files if path.exists()]
                    if not mtimes:
                        errors.append("runtime telemetry was not created after startup grace")
                    elif now - max(mtimes) > 900:
                        errors.append(f"no runtime telemetry growth for {now-max(mtimes):.0f}s")
            if int(probe.get("helpers", 0)) and int(probe.get("gpu_processes", 0)) == 0:
                errors.append(f"orphan helpers={probe['helpers']}")
            if int(probe.get("mem_available_kib", 0)) < 256 * 1024 * 1024:
                errors.append(f"MemAvailable={probe['mem_available_kib']} KiB")
        except Exception as exc:
            probe = {"error": repr(exc)}
            errors.append(str(exc))
        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest() if errors else "healthy"
        event = {
            "event": "alert" if errors else "healthy", "complete": complete,
            "errors": errors, "heartbeat_age": hb_age, "controllers": controllers,
            "allocation": alloc, "probe": probe,
        }
        append_event(event)
        if errors and fingerprint != previous.get("fingerprint"):
            message = (
                "WATCHDOG ALERT: node2800的BrowseComp Qwen3-8B低并发矩阵异常："
                + "; ".join(errors)
                + "。请立即检查、修复并从当前组恢复；不要跳过未完成的300+1200窗口。"
            )
            event["queue"] = queue(args.thread, message)
        previous = {"ts": now, "fingerprint": fingerprint, **event}
        atomic_json(WATCH_STATE, previous)
        time.sleep(max(args.interval_seconds, 60))


if __name__ == "__main__":
    raise SystemExit(main())
