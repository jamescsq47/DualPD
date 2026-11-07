#!/usr/bin/env python3
"""Quiet watchdog for the two-case Qwen3-8B c512 Nsight run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


REPO = Path("/home/shangy/dualpd")
ROOT = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "nsys-browsecomp-qwen3-8b-h100-4p4d-c512-w300-m100-20261025"
)
STATE = ROOT / "state.json"
WATCH_STATE = ROOT / "watchdog-state.json"
WATCH_LOG = ROOT / "watchdog.jsonl"
LOCK = ROOT / "watchdog.lock"
CODEX = Path("/home/shangy/.local/bin/codex")
EXPECTED_CASES = ("new-method", "hicache")
EXPECTED_PORTS = {
    27300, 27320, 27400, 27420, 27301, 27321, 27401, 27421, 27510, 8750,
}


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


def allocation(job_id: str) -> tuple[str, str]:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T|%N"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30, check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if lines:
        fields = lines[0].split("|", 1)
        return fields[0], fields[1] if len(fields) > 1 else ""
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", job_id, "--format=State,NodeList"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30, check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|", 2)
        if fields and fields[0]:
            return fields[0].split("+", 1)[0], fields[1] if len(fields) > 1 else ""
    return "MISSING", ""


def node_probe(job_id: str, node: str) -> dict[str, Any]:
    program = r'''import json,os,subprocess
gpu={x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()}
ports=set()
for line in subprocess.check_output(['ss','-H','-ltn'],text=True).splitlines():
 try: ports.add(int(line.split()[3].rsplit(':',1)[1]))
 except (IndexError,ValueError): pass
counts={'controller':0,'nsys':0,'workload':0}
for entry in os.listdir('/proc'):
 if not entry.isdigit(): continue
 if int(entry) == os.getpid(): continue
 try:
  cmd=open('/proc/'+entry+'/cmdline','rb').read().replace(b'\0',b' ').decode(errors='replace')
  counts['controller'] += 'browsecomp_qwen3_8b_c512_nsys.py' in cmd
  counts['nsys'] += 'nsys profile' in cmd
  counts['workload'] += 'dualpd.inference.py' in cmd and '--closed-loop' in cmd
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
mem={}
for line in open('/proc/meminfo'):
 key,value,*_=line.replace(':','').split()
 if key in {'MemAvailable','Committed_AS','CommitLimit'}: mem[key]=int(value)
print(json.dumps({'gpu_processes':len(gpu),'ports':sorted(ports),'controller':counts['controller'],'nsys':counts['nsys'],'workload':counts['workload'],'mem_available_kib':mem.get('MemAvailable',0),'commit_ratio':mem.get('Committed_AS',0)/max(mem.get('CommitLimit',1),1)}))'''
    result = subprocess.run(
        [
            "srun", f"--jobid={job_id}", "--overlap", "--exact", "--nodes=1",
            "--ntasks=1", "--cpus-per-task=1", "--mem=1G", "-w", node,
            "python3", "-c", program,
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"node probe rc={result.returncode}: {result.stdout[-1000:]}")
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"node probe returned no JSON: {result.stdout[-1000:]}")


def queue(thread: str, message: str) -> dict[str, Any]:
    result = subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False,
    )
    return {"returncode": result.returncode, "output": result.stdout[-1000:]}


def running_case(state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    active = [
        (name, item) for name, item in state.items()
        if name in EXPECTED_CASES and isinstance(item, dict) and item.get("status") == "running"
    ]
    return max(active, key=lambda pair: float(pair[1].get("started_at", 0))) if active else None


def live_age(run_dir: Path) -> float:
    candidates = [
        run_dir / "controller.log", run_dir / "dualpd.inference.log",
        run_dir / "closed_loop_events.jsonl", run_dir / "engine_metrics.jsonl",
        *(run_dir / "logs").glob("*.log"),
        *(run_dir / "nsys").glob("*"),
        *run_dir.parent.glob(f"{run_dir.name}-edge-trace*"),
    ]
    mtimes = [path.stat().st_mtime for path in candidates if path.is_file()]
    return time.time() - max(mtimes) if mtimes else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--thread", default=os.environ.get("CODEX_THREAD_ID"))
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    if not args.thread:
        parser.error("--thread or CODEX_THREAD_ID is required")

    ROOT.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another nsys watchdog holds the lock")
    previous = read_json(WATCH_STATE)
    append_event({"event": "watchdog_started", "pid": os.getpid(), "job_id": args.job_id})

    while True:
        now = time.time()
        state = read_json(STATE)
        complete = sum(state.get(name, {}).get("status") == "complete" for name in EXPECTED_CASES)
        if complete == len(EXPECTED_CASES):
            missing = [
                str(ROOT / f"{name}-edge-trace.json")
                for name in EXPECTED_CASES
                if not (ROOT / f"{name}-edge-trace.json").is_file()
            ]
            if not missing:
                event = {"event": "complete", "complete": complete, "job_id": args.job_id}
                append_event(event)
                atomic_json(WATCH_STATE, {"ts": now, **event})
                queue(
                    args.thread,
                    "NSYS WATCHDOG COMPLETE: Qwen3-8B BrowseComp c512 的新方法和HiCache均已完成，"
                    "Edge Trace JSON已生成。请核验trace、稳态边界及最终清理门禁。",
                )
                return 0

        status, node = allocation(args.job_id)
        errors: list[str] = []
        probe: dict[str, Any] = {}
        current = running_case(state)
        if any(state.get(name, {}).get("status") == "failed" for name in EXPECTED_CASES):
            failed = [name for name in EXPECTED_CASES if state.get(name, {}).get("status") == "failed"]
            errors.append("case failed: " + ", ".join(failed))
        if status == "PENDING":
            pass
        elif status != "RUNNING":
            errors.append(f"allocation={status}")
        else:
            try:
                probe = node_probe(args.job_id, node)
                if int(probe.get("controller", 0)) != 1:
                    errors.append(f"expected one controller, found {probe.get('controller', 0)}")
                if int(probe.get("mem_available_kib", 0)) < 256 * 1024 * 1024:
                    errors.append(f"MemAvailable={probe.get('mem_available_kib')} KiB")
                if current:
                    name, item = current
                    elapsed = now - float(item.get("started_at", now))
                    run_dir = Path(str(item.get("run_dir", "")))
                    age = live_age(run_dir)
                    if age > 900:
                        errors.append(f"no runtime telemetry growth for {age:.0f}s")
                    # Loading and Host-arena registration take about 20 minutes.
                    # Enforce the complete topology after a 25-minute startup grace.
                    if elapsed > 1500 and not (run_dir / "closed_loop_boundaries.json").exists():
                        if int(probe.get("gpu_processes", 0)) not in {8, 9}:
                            errors.append(
                                f"expected 8 model GPU processes plus optional search, "
                                f"found {probe.get('gpu_processes', 0)}"
                            )
                        if int(probe.get("nsys", 0)) != 8:
                            errors.append(f"expected 8 nsys parents, found {probe.get('nsys', 0)}")
                        if int(probe.get("workload", 0)) != 1:
                            errors.append(f"expected one closed-loop workload, found {probe.get('workload', 0)}")
                        missing_ports = sorted(EXPECTED_PORTS.difference(probe.get("ports", [])))
                        if missing_ports:
                            errors.append(f"missing ports {missing_ports}")
                    if (run_dir / "closed_loop_boundaries.json").exists() and age > 300:
                        errors.append("measurement complete but postprocessing is stalled")
            except Exception as error:
                probe = {"error": repr(error)}
                errors.append(f"node probe failed: {type(error).__name__}: {error}")

        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest() if errors else "healthy"
        event = {
            "event": "alert" if errors else ("pending" if status == "PENDING" else "healthy"),
            "job_id": args.job_id, "allocation": status, "node": node,
            "complete": complete, "errors": errors, "probe": probe,
        }
        append_event(event)
        last_alert = float(previous.get("last_alert_epoch", 0))
        if errors and (fingerprint != previous.get("fingerprint") or now - last_alert >= 600):
            event["queue"] = queue(
                args.thread,
                f"NSYS WATCHDOG ALERT: {node or 'unassigned'} 的 Qwen3-8B c512 nsys 实验异常："
                + "; ".join(errors)
                + "。请立即检查并修复；恢复后必须确认控制器、8个GPU服务、端口、"
                  "300秒预热及100秒trace采集都正常。",
            )
            last_alert = now
        previous = {"ts": now, "fingerprint": fingerprint, "last_alert_epoch": last_alert, **event}
        atomic_json(WATCH_STATE, previous)
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
