#!/usr/bin/env python3
"""Quiet watchdog for the paired P-credit c576 P->D ablation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/replay/cluster/browsecomp_qwen3_8b_pd_scale_h100_watchdog.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-paired-credit-c576-20261028-r10"
)

spec = importlib.util.spec_from_file_location("p2d_paired_watchdog_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
watch = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = watch
spec.loader.exec_module(watch)

watch.CONTROL = CONTROL
watch.STATE = CONTROL / "state.json"
watch.HEARTBEAT = CONTROL / "heartbeat.json"
watch.WATCH_STATE = CONTROL / "watchdog-state.json"
watch.WATCH_LOG = CONTROL / "watchdog.jsonl"
watch.EXPECTED = 1

original_allocation = watch.allocation


def allocation(job_id: str) -> tuple[str, str]:
    try:
        status, node = original_allocation(job_id)
    except (subprocess.TimeoutExpired, OSError):
        return "UNKNOWN", ""
    if not status.startswith("slurm_load_jobs error"):
        return status, node
    result = subprocess.run(
        ["sacct", "-X", "-n", "-j", job_id, "--format=State,NodeList", "--parsable2"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30, check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 1)
        if fields and fields[0]:
            return fields[0].split()[0].rstrip("+"), fields[1] if len(fields) > 1 else ""
    return "MISSING", ""


watch.allocation = allocation
original_queue = watch.queue


def queue(thread: str, message: str):
    message = message.replace(
        "BrowseComp Qwen3-8B H100 P:D缩放消融已完成12/12",
        "BrowseComp Qwen3-8B H100 c576 配对P-credit原生P→D消融已完成1/1",
    ).replace(
        "BrowseComp Qwen3-8B P:D缩放矩阵异常",
        "BrowseComp Qwen3-8B c576 配对P-credit原生P→D消融异常",
    )
    return original_queue(thread, message)


watch.queue = queue


if __name__ == "__main__":
    raise SystemExit(watch.main())
