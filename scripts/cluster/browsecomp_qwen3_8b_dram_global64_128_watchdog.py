#!/usr/bin/env python3
"""Quiet monitoring, explicitly labelled GLOBAL D2P64/P2D128 GiB."""
from pathlib import Path
import browsecomp_qwen3_8b_dram_d64p128_watchdog as previous

watch = previous.watch
watch.CONTROL = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-global64-128-c512-20261028")
watch.STATE = watch.CONTROL / "state.json"
watch.HEARTBEAT = watch.CONTROL / "heartbeat.json"
watch.WATCH_STATE = watch.CONTROL / "watchdog-state.json"
watch.WATCH_LOG = watch.CONTROL / "watchdog.jsonl"


def queue(thread, message):
    message = message.replace("P:D缩放消融已完成12/12", "DRAM全节点D→P64/P→D128 GiB已完成1/1")
    message = message.replace("P:D缩放矩阵", "DRAM全节点D→P64/P→D128 GiB实验")
    return previous.source.original_queue(thread, message)


watch.queue = queue

if __name__ == "__main__":
    raise SystemExit(previous.main())
