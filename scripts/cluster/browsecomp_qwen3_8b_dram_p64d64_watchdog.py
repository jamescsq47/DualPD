#!/usr/bin/env python3
"""Quiet 10-minute monitoring for the one P64/D64 follow-up."""
import json
import os
import subprocess
import sys
import time

import browsecomp_qwen3_8b_dram_h100_watchdog as source


watch = source.watch
watch.CONTROL = watch.CONTROL.parent / "h100-browsecomp-qwen3-8b-dram-p64d64-c512-20261022"
watch.STATE = watch.CONTROL / "state.json"
watch.HEARTBEAT = watch.CONTROL / "heartbeat.json"
watch.WATCH_STATE = watch.CONTROL / "watchdog-state.json"
watch.WATCH_LOG = watch.CONTROL / "watchdog.jsonl"
watch.EXPECTED = 1
original_allocation = watch.allocation


def allocation(job_id):
    status, nodes = original_allocation(job_id)
    selected = os.environ.get("PD_DRAM_WATCH_NODE")
    if status != "RUNNING" or not selected:
        return status, nodes
    # A borrowed allocation can contain two nodes. Never fan this one-node
    # experiment's probes out to its unrelated sibling node.
    try:
        members = subprocess.check_output(["scontrol", "show", "hostnames", nodes],
                                          text=True, stderr=subprocess.STDOUT, timeout=30).split()
    except (subprocess.SubprocessError, OSError):
        return "NODE_MEMBERSHIP_QUERY_FAILED", selected
    if selected not in members:
        return "NODE_NOT_IN_ALLOCATION", selected
    return status, selected


def queue(thread, message):
    message = message.replace("P:D缩放消融已完成12/12", "DRAM追加P64/D64已完成1/1")
    message = message.replace("P:D缩放矩阵", "DRAM追加P64/D64实验")
    return source.original_queue(thread, message)


watch.queue = queue
watch.allocation = allocation


def main():
    # Do not report heartbeat=inf during the initial four-second cleanup
    # probe. A genuinely missing controller still alerts after 180 seconds.
    job_id = sys.argv[sys.argv.index("--job-id") + 1]
    watch.CONTROL.mkdir(parents=True, exist_ok=True)
    watch.append_event(dict(event="startup_wait", pid=os.getpid(), job_id=job_id))
    deadline = None
    while True:
        status, _ = watch.allocation(job_id)
        if status == "PENDING":
            time.sleep(30)
            continue
        if status != "RUNNING" or watch.HEARTBEAT.exists():
            break
        deadline = deadline or time.monotonic() + 180
        if time.monotonic() >= deadline:
            break
        time.sleep(5)
    return watch.main()


if __name__ == "__main__":
    raise SystemExit(main())
