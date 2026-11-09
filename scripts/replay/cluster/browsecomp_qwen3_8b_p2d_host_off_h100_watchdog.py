#!/usr/bin/env python3
"""Quiet watchdog for the one-case H100 P->D Host-off ablation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scripts/replay/cluster/browsecomp_qwen3_8b_pd_scale_h100_watchdog.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-host-off-c576-20261028-r1"
)

spec = importlib.util.spec_from_file_location("p2d_host_off_watchdog_base", SOURCE)
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

original_queue = watch.queue


def queue(thread: str, message: str):
    message = message.replace(
        "BrowseComp Qwen3-8B H100 P:D缩放消融已完成12/12",
        "BrowseComp Qwen3-8B H100 c576 仅关闭P→D Host消融已完成1/1",
    ).replace(
        "BrowseComp Qwen3-8B P:D缩放矩阵异常",
        "BrowseComp Qwen3-8B c576 仅关闭P→D Host消融异常",
    )
    return original_queue(thread, message)


watch.queue = queue


if __name__ == "__main__":
    raise SystemExit(watch.main())
