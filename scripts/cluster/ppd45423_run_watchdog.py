#!/usr/bin/env python3
"""Quiet watchdog for one isolated PR #45423 benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


CODEX = Path("/home/shangy/.local/bin/codex")
ROOT = Path(__file__).resolve().parents[2]
ACTIVE_STAGES = {"starting_models", "starting_services", "running_workload", "summarizing"}


def ssh(node: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ssh exited {result.returncode}: {result.stdout[-1000:]}")
    return result.stdout


def queue(thread: str, message: str) -> None:
    subprocess.run(
        [str(CODEX), "queue", "--thread", thread, "--message", message],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--interval-seconds", type=int, default=120)
    args = parser.parse_args()
    watch_state = args.run_dir / "watchdog-state.json"
    watch_log = args.run_dir / "watchdog.jsonl"
    previous_fingerprint = ""
    last_event_count = -1
    last_progress_epoch = time.time()

    while True:
        now = time.time()
        errors: list[str] = []
        complete = False
        stage = "missing"
        event_count = 0
        expected_servers = 2
        try:
            state_path = args.run_dir / "state.json"
            if not state_path.exists():
                errors.append("state.json is missing")
            else:
                state = json.loads(state_path.read_text())
                stage = str(state.get("stage", "unknown"))
                expected_servers = 2 * int(state.get("pair_count", 1))
                complete = stage == "complete_clean"
                if stage == "cleanup_gate_failed":
                    errors.append(stage)
            event_path = args.run_dir / "affinity-events.jsonl"
            if event_path.exists():
                event_count = sum(1 for _ in event_path.open())
            if event_count != last_event_count:
                last_event_count = event_count
                last_progress_epoch = now
            if stage == "running_workload" and now - last_progress_epoch > 300:
                errors.append(f"no affinity event progress for {int(now - last_progress_epoch)}s")
            if stage in ACTIVE_STAGES:
                probe = json.loads(
                    ssh(
                        args.node,
                        "python3 - <<'PY'\n"
                        "import json,os\n"
                        "gpu=[]\n"
                        "for entry in os.listdir('/proc'):\n"
                        "  if not entry.isdigit(): continue\n"
                        "  try: cmd=open('/proc/'+entry+'/cmdline','rb').read().replace(b'\\0',b' ')\n"
                        "  except OSError: continue\n"
                        "  if cmd.startswith(b'/opt/sglang/bin/python -m sglang.launch_server'): gpu.append(int(entry))\n"
                        "print(json.dumps({'servers':len(gpu)}))\n"
                        "PY",
                    )
                )
                if stage != "starting_models" and int(probe.get("servers", 0)) != expected_servers:
                    errors.append(
                        f"expected {expected_servers} model servers, found {probe.get('servers', 0)}"
                    )
        except Exception as exc:
            errors.append(f"watchdog probe: {type(exc).__name__}: {exc}")

        fingerprint = hashlib.sha256("\n".join(errors).encode()).hexdigest()
        event = {
            "epoch": now,
            "stage": stage,
            "event_count": event_count,
            "last_progress_epoch": last_progress_epoch,
            "errors": errors,
            "fingerprint": fingerprint,
        }
        with watch_log.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        atomic_json(watch_state, event)
        if complete:
            queue(
                args.thread,
                f"PPD #45423 WATCHDOG COMPLETE: {args.run_dir.name} 已完成并通过最终清理门禁，请读取结果。",
            )
            return 0
        if errors and fingerprint != previous_fingerprint:
            queue(
                args.thread,
                f"PPD #45423 WATCHDOG ALERT: {args.run_dir.name}: " + "; ".join(errors),
            )
        previous_fingerprint = fingerprint
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
