#!/usr/bin/env python3
"""Run the MiniMax-M2.7 TP8/EP8 two-node SWE500 baseline matrix."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

if __package__:
    from . import qwen38_matrix_autopilot as core
else:
    import qwen38_matrix_autopilot as core


core.JOB_ID = os.environ.get("MINIMAX_M27_BASELINE_JOB_ID", "25776117")
core.MIN_START_SECONDS = int(
    os.environ.get("MINIMAX_M27_BASELINE_MIN_START_SECONDS", 6 * 3600)
)
core.NODE0 = os.environ.get("MINIMAX_M27_BASELINE_NODE0", "node1800")
core.NODE1 = os.environ.get("MINIMAX_M27_BASELINE_NODE1", "node2701")
core.MODEL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/models/MiniMax-M2.7-d494266a"
)
core.BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
core.BASELINE_ENGINE = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/sglang-src/pd_baseline_b1ef6"
)
core.CONTROL_DIR = core.RUNS / "minimax-m27-tp8-two-node-baseline-matrix-r1"
core.STATE_PATH = core.CONTROL_DIR / "state.json"
core.LOCK_PATH = core.CONTROL_DIR / "controller.lock"
core.REPORT = (
    core.ROOT / "runs-host/SWEBENCH_MINIMAX_M27_TP8_TWO_NODE.md"
)

METHODS = ("native", "hicache", "colocated")
METHOD_LABEL = {
    "native": "No-reverse PD",
    "hicache": "HiCache/Mooncake PD",
    "colocated": "Colocated",
}


def run_name(concurrency: int, method: str) -> str:
    label = {
        "native": "native-oneway-1p1d",
        "hicache": "hicache-mooncake-1p1d",
        "colocated": "colocated-2replica",
    }[method]
    # c128 No-reverse r4 is the first valid formal result.  HiCache r4 failed
    # before serving because its Host pool was smaller than MiniMax's device
    # KV pool, and r5 collided with the late r4 startup.  Keep those immutable
    # failure artifacts and use fresh r6 identifiers for every pending case.
    if (concurrency, method) == (128, "native"):
        revision = "r4"
    elif (concurrency, method) == (256, "hicache"):
        # r6 reached T450 but verifier forks hit host task pressure; r7/r8
        # confirmed the default multithreaded squashfuse mount is the source.
        # r9 uses one FUSE worker per SIF while preserving c256/verifier=16.
        revision = "r11"
    else:
        revision = "r6"
    return (
        f"minimax-m27-tp8ep8-two-node-{label}-swe500-c{concurrency}"
        f"-20261016-{revision}"
    )


core.EXPERIMENTS = [
    {"concurrency": concurrency, "method": method,
     "run_id": run_name(concurrency, method)}
    for concurrency in (128, 192, 256)
    for method in METHODS
]
core.SEED_RUNS = []


def common_prepare(experiment: dict[str, Any]) -> list[str | Path]:
    return [
        "--job-id", core.JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", core.run_dir(experiment),
        "--python", core.BASELINE_PYTHON,
        "--workload-python", core.PD_PYTHON,
        "--sglang-root", core.BASELINE_ENGINE,
        "--model", core.MODEL,
        "--reasoning-parser", "minimax-append-think",
        "--tool-call-parser", "minimax-m2",
        "--auto-kernel-backends",
        "--data-root", core.DATA,
        "--swe-sif-dir", core.SIF,
        "--workload-config", core.WORKLOAD,
        "--requests", "500",
        "--concurrency", str(experiment["concurrency"]),
        "--tp-size", "8",
        "--ep-size", "8",
        "--mamba-buffer-strategy", "no_buffer",
    ]


def prepare_and_start(experiment: dict[str, Any]) -> None:
    directory = core.run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse existing run directory: {directory}")
    method = experiment["method"]
    if method == "colocated":
        prepare = [
            core.BASELINE_PYTHON,
            core.COLOCATED_LAUNCHER,
            "prepare",
            *common_prepare(experiment),
            "--groups-per-node", "1",
            "--node0-groups", "1",
            "--node1-groups", "1",
            "--cpus-per-node", "224",
        ]
        core.command(prepare)
        core.command(
            [
                core.BASELINE_PYTHON,
                core.COLOCATED_LAUNCHER,
                "start",
                "--job-id", core.JOB_ID,
                "--run-dir", directory,
            ],
            timeout=3600,
        )
        return

    prepare = [
        core.BASELINE_PYTHON,
        core.NATIVE_LAUNCHER,
        "prepare",
        *common_prepare(experiment),
        "--prefill-groups-per-node", "1",
        "--decode-groups-per-node", "1",
        # The commit-pinned clean baseline predates the v0.5.21 CLI rename.
        "--mamba-cache-strategy-flag", "scheduler",
        "--local-root",
        f"/var/tmp/shangy/dualpd-minimax-baseline/{experiment['run_id']}",
        "--launch-transport", "srun",
    ]
    if method == "hicache":
        prepare.extend(
            [
                "--native-mooncake",
                "--disable-mlir-multithreading",
                # Match Agentic-PD's physical per-node Host budgets.
                "--prefill-hicache-size-gb", "64",
                "--decode-hicache-size-gb", "64",
                "--mooncake-store-size-gb", "512",
                "--mooncake-prefetch-policy", "wait_complete",
                "--workload-job-id", core.JOB_ID,
                "--workload-node", core.NODE0,
                "--storage-job-id", core.JOB_ID,
                "--storage-node", core.NODE1,
            ]
        )
    core.command(prepare)
    core.command(
        [
            core.BASELINE_PYTHON,
            core.NATIVE_LAUNCHER,
            "start",
            "--job-id", core.JOB_ID,
            "--run-dir", directory,
        ],
        timeout=3600,
    )


core.prepare_and_start = prepare_and_start

START_MARKER = "<!-- MINIMAX_M27_BASELINE_AUTOPILOT_START -->"
END_MARKER = "<!-- MINIMAX_M27_BASELINE_AUTOPILOT_END -->"


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = []
    for experiment in core.EXPERIMENTS:
        directory = core.run_dir(experiment)
        result = core.completed_result(
            experiment["concurrency"], experiment["method"], directory
        ) or core.t450_result(
            experiment["concurrency"], experiment["method"], directory
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
        "## Baseline 九组自动运行状态",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}；"
        f"状态：`{state['status']}`；正式结果 **{len(results)}/9**。",
    ]
    if current is not None:
        lines.append(
            f"> 当前：c{current['concurrency']} {METHOD_LABEL[current['method']]}，"
            f"episode progress **{current_progress}/500**，"
            f"run `{current['run_id']}`。"
        )
    elif state.get("status") == "complete":
        lines.append("> c128/c192/c256 的三种 baseline 共九组均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend(
        [
            "",
            "| 并发 | 方法 | 完成/失败 | Resolved | **T450** | 0–T450 Decode tok/s | 口径/异常 |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        caveat = []
        if result.get("result_scope") == "T450-only":
            caveat.append("T450-only")
        if result.get("environment_errors"):
            caveat.append(f"env={result['environment_errors']}")
        lines.append(
            "| {c} | {method} | {done}/{failed} | {solved}/500 | "
            "{t450} s | {decode} | {caveat} |".format(
                c=result["concurrency"],
                method=METHOD_LABEL[result["method"]],
                done=result["completed"],
                failed=result["failed"],
                solved=result["solved"],
                t450=core.value(result["t450_seconds"]),
                decode=core.value(result.get("t450_decode_tps")),
                caveat=", ".join(caveat) or "—",
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
