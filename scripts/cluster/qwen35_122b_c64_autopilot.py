#!/usr/bin/env python3
"""Run the four-method Qwen3.5-122B TP8 SWE500 c64 matrix.

This reuses the validated strict-two-node launch recipes while keeping a
separate durable state directory and immutable run directories for c64.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import qwen35_122b_completion_autopilot as base


core = base.core

core.JOB_ID = os.environ.get("QWEN35_122B_C64_JOB_ID", "25647805")
core.NODE0 = os.environ.get("QWEN35_122B_C64_NODE0", "node1700")
core.NODE1 = os.environ.get("QWEN35_122B_C64_NODE1", "node1701")
core.CONTROL_DIR = core.RUNS / "qwen35-122b-tp8-two-node-c64-autopilot"
core.STATE_PATH = core.CONTROL_DIR / "state.json"
core.LOCK_PATH = core.CONTROL_DIR / "controller.lock"
core.REPORT = (
    core.ROOT
    / "runs-host/SWEBENCH_QWEN35_122B_A10B_TP8_C64.md"
)

# The current pd source expects the newer FlashInfer all-reduce ABI installed
# in the validated colocated environment.  Keep the current source checkout,
# but do not combine it with the older FlashInfer extension from the pd env.
COLOCATED_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/"
    "pd_baseline/bin/python3.12"
)


def run_name(method: str) -> str:
    label = {
        "agentic": "agentic",
        "colocated": "colocated",
        "native": "native-oneway",
        "hicache": "hicache-mooncake",
    }[method]
    revision = {
        "colocated": "r3",
        # r1 failed before model loading because the old baseline CLI spelling
        # was not selected.  Keep failed artifacts immutable.
        "native": "r2",
    }.get(method, "r1")
    return f"q35-122b-{label}-swe500-c64-20261020-{revision}"


core.EXPERIMENTS = [
    {"concurrency": 64, "method": method, "run_id": run_name(method)}
    for method in ("agentic", "colocated", "hicache", "native")
]
core.SEED_RUNS = []

def prepare_and_start(experiment: dict[str, Any]) -> None:
    """Use current PD code for Colocated and the validated recipes otherwise."""
    if experiment["method"] != "colocated":
        base.prepare_and_start(experiment)
        return

    directory = core.run_dir(experiment)
    if directory.exists():
        raise RuntimeError(
            f"refusing to reuse existing next-run directory: {directory}"
        )
    prepare = [
        COLOCATED_PYTHON,
        core.COLOCATED_LAUNCHER,
        "prepare",
        "--job-id", core.JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", directory,
        "--python", COLOCATED_PYTHON,
        "--workload-python", core.PD_PYTHON,
        "--sglang-root", core.PD_ENGINE,
        "--model", core.MODEL,
        "--data-root", core.DATA,
        "--swe-sif-dir", core.SIF,
        "--workload-config", core.WORKLOAD,
        "--requests", "500",
        "--concurrency", str(experiment["concurrency"]),
        "--tp-size", "8",
        "--groups-per-node", "1",
        "--cpus-per-node", "224",
    ]
    core.command(prepare)
    core.command(
        [
            COLOCATED_PYTHON,
            core.COLOCATED_LAUNCHER,
            "start",
            "--job-id", core.JOB_ID,
            "--run-dir", directory,
        ],
        timeout=3600,
    )


# Agentic/native/HiCache retain the exact validated Qwen3.5 launch recipes.
# The HiCache recipe selects the c128 capacity profile because c64 is below
# that pressure point.
core.prepare_and_start = prepare_and_start
core.t450_result = lambda *_args, **_kwargs: None


START_MARKER = "<!-- QWEN35_C64_AUTOPILOT_START -->"
END_MARKER = "<!-- QWEN35_C64_AUTOPILOT_END -->"


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = []
    for experiment in core.EXPERIMENTS:
        result = core.completed_result(
            experiment["concurrency"],
            experiment["method"],
            core.run_dir(experiment),
        )
        if result is not None:
            results.append(result)

    current = (
        core.EXPERIMENTS[state["index"]]
        if state["index"] < len(core.EXPERIMENTS)
        else None
    )
    lines = [
        START_MARKER,
        "## 自动运行进度",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}；"
        f"状态：`{state['status']}`；完整结果 **{len(results)}/4**。",
    ]
    if current is not None:
        lines.append(
            f"> 当前：c64 {core.METHOD_LABEL[current['method']]}，"
            f"episode progress **{current_progress}/500**，"
            f"run `{current['run_id']}`。"
        )
    else:
        lines.append("> 四组 c64 实验均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")

    lines.extend(
        [
            "",
            "| 方法 | 完成/失败 | Resolved | T450 | T450 前 Decode tok/s | T500 | 全程 Prefill tok/s | 全程 Decode tok/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            "| {method} | {done}/{failed} | {solved}/500 | {t450} s | "
            "{t450_decode} | {t500} s | {prefill} | {decode} |".format(
                method=core.METHOD_LABEL[result["method"]],
                done=result["completed"],
                failed=result["failed"],
                solved=result["solved"],
                t450=core.value(result["t450_seconds"]),
                t450_decode=core.value(result["t450_decode_tps"]),
                t500=core.value(result["t500_seconds"]),
                prefill=core.value(result["scheduler_prefill_tps"]),
                decode=core.value(result["scheduler_decode_tps"]),
            )
        )

    lines.extend(["", "结果目录：", ""])
    for result in results:
        lines.append(f"- `{result['directory']}`")
    lines.extend([END_MARKER, ""])
    block = "\n".join(lines)
    text = core.REPORT.read_text()
    if START_MARKER in text and END_MARKER in text:
        before, rest = text.split(START_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    temporary = core.REPORT.with_suffix(core.REPORT.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, core.REPORT)


core.update_report = update_report


if __name__ == "__main__":
    raise SystemExit(core.main())
