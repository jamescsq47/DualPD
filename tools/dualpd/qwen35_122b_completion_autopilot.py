#!/usr/bin/env python3
"""Complete the strict two-node Qwen3.5-122B TP8 SWE500 matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import qwen38_matrix_autopilot as core


core.JOB_ID = os.environ.get("QWEN35_122B_JOB_ID", "25569233")
core.NODE0 = "node1801"
core.NODE1 = "node2701"
core.MODEL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3.5-122B-A10B"
)
core.BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
# Keep native one-way PD and HiCache on the exact clean baseline source commit
# used by the earlier formal runs.  The mutable pd_baseline checkout was later
# fast-forwarded to pd_multi_node and now contains Agentic modules.  The core
# controller's dependency environment supplies the commit-pinned Torch 2.9.1
# and Transformers 5.3.0 versions.
core.BASELINE_ENGINE = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/sglang-src/pd_baseline_b1ef6"
)
core.CONTROL_DIR = (
    core.RUNS / "qwen35-122b-tp8-two-node-completion-autopilot"
)
core.STATE_PATH = core.CONTROL_DIR / "state.json"
core.LOCK_PATH = core.CONTROL_DIR / "controller.lock"
core.REPORT = (
    core.ROOT
    / "examples/pd/runs-host/SWEBENCH_QWEN35_122B_A10B_TP8_TWO_NODE.md"
)


def run_name(concurrency: int, method: str) -> str:
    label = {
        "agentic": "agentic-1p1d",
        "colocated": "colocated-2replica",
        "native": "native-oneway-1p1d",
        "hicache": "hicache-mooncake-1p1d",
    }[method]
    return (
        f"qwen35-122b-formal2n-{label}-swe500-c{concurrency}"
        "-20261011-r1"
    )


core.EXPERIMENTS = [
    {"concurrency": concurrency, "method": method, "run_id": run_name(concurrency, method)}
    for concurrency, methods in (
        (192, ("agentic", "colocated", "native", "hicache")),
        (256, ("hicache",)),
    )
    for method in methods
]
core.SEED_RUNS = []


def prepare_and_start(experiment: dict[str, Any]) -> None:
    directory = core.run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse existing next-run directory: {directory}")
    method = experiment["method"]
    concurrency = str(experiment["concurrency"])
    if method == "agentic":
        common = [
            core.PD_PYTHON,
            core.AGENTIC_LAUNCHER,
            "--job-id", core.JOB_ID,
            "--prefill-node", core.NODE0,
            "--decode-node", core.NODE1,
            "--python", core.PD_PYTHON,
            "--sglang-root", core.PD_ENGINE,
            "--model", core.MODEL,
            "--data-root", core.DATA,
            "--swe-sif-dir", core.SIF,
            "--workload-config", core.WORKLOAD,
            "--run-dir", directory,
            "--run-id", experiment["run_id"],
            "--requests", "500",
            "--concurrency", concurrency,
            "--tp-size", "8",
            "--groups-per-node", "1",
            "--d2p-host-gib-per-rank", "64",
            "--p2d-host-gib-per-rank", "32",
            "--d2p-direct-lanes", "2",
            "--p2d-direct-lanes", "1",
            "--host-lanes", "2",
            "--host-load-workers", "1",
            "--nixl-progress-threads", "2",
            "--forward-stream-priority", "-5",
            "--flashinfer-allreduce-max-tokens", "8192",
            "--launch-transport", "ssh",
        ]
        core.command([*common, "prepare"])
        core.command([*common, "--detach", "start-control"])
        core.command([*common, "control-ready"], timeout=900)
        core.command([*common, "--detach", "start-prefill"], timeout=2400)
        core.command([*common, "--detach", "start-decode"], timeout=600)
        core.command([*common, "wait-ready"], timeout=2400)
        core.command([*common, "--detach", "start-router"], timeout=600)
        core.command([*common, "--detach", "run-workload"], timeout=600)
        return

    if method == "colocated":
        prepare = [
            core.BASELINE_PYTHON,
            core.COLOCATED_LAUNCHER,
            "prepare",
            "--job-id", core.JOB_ID,
            "--run-id", experiment["run_id"],
            "--run-dir", directory,
            "--python", core.BASELINE_PYTHON,
            "--workload-python", core.PD_PYTHON,
            "--sglang-root", core.BASELINE_ENGINE,
            "--model", core.MODEL,
            "--data-root", core.DATA,
            "--swe-sif-dir", core.SIF,
            "--workload-config", core.WORKLOAD,
            "--requests", "500",
            "--concurrency", concurrency,
            "--tp-size", "8",
            "--groups-per-node", "1",
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
        "--job-id", core.JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", directory,
        "--python", core.BASELINE_PYTHON,
        "--workload-python", core.PD_PYTHON,
        "--sglang-root", core.BASELINE_ENGINE,
        "--model", core.MODEL,
        "--data-root", core.DATA,
        "--swe-sif-dir", core.SIF,
        "--workload-config", core.WORKLOAD,
        "--requests", "500",
        "--concurrency", concurrency,
        "--tp-size", "8",
        "--prefill-groups-per-node", "1",
        "--decode-groups-per-node", "1",
        # The commit-pinned Qwen3.5 baseline exposes the older CLI spelling.
        # This changes only argument compatibility; the strategy values remain
        # no_buffer for Prefill and extra_buffer for Decode/HiCache.
        "--mamba-cache-strategy-flag", "scheduler",
        # Do not reuse Triton/JIT artifacts produced by earlier source and
        # Python combinations.  A stale resolve_future_token_ids binary caused
        # a late SIGBUS in the first formal c192 native run.
        "--local-root",
        f"/var/tmp/shangy/dualpd-native-pd-formal/{experiment['run_id']}",
        "--launch-transport", "ssh",
    ]
    if method == "hicache":
        prepare.extend(
            [
                "--native-mooncake",
                # Qwen3.5 uses a hybrid Attention/Mamba cache.  The clean
                # baseline commit predates native Mooncake support for the
                # grouped Host pools, so use the same narrow run-local
                # compatibility layer as the completed formal c128 run.
                "--mooncake-compat",
                str(
                    core.ROOT
                    / "examples/pd/scripts/compat/qwen35_mooncake"
                ),
                "--mamba-full-memory-ratio", "0.5",
                "--prefill-hicache-size-gb", "26",
                "--decode-hicache-size-gb", "26",
                "--mamba-hicache-size-gb", "16",
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

# These are formal SWE500 runs.  Do not stop merely because the 450th result
# exists; wait for the complete summary and use its exact measurement epoch.
core.t450_result = lambda *_args, **_kwargs: None


START_MARKER = "<!-- QWEN35_COMPLETION_AUTOPILOT_START -->"
END_MARKER = "<!-- QWEN35_COMPLETION_AUTOPILOT_END -->"


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
        "## 10. 严格双节点矩阵补全（自动运行）",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}；"
        f"状态：`{state['status']}`；完整结果 **{len(results)}/5**。",
    ]
    if current is not None:
        lines.append(
            f"> 当前：c{current['concurrency']} "
            f"{core.METHOD_LABEL[current['method']]}，"
            f"episode progress **{current_progress}/500**，"
            f"run `{current['run_id']}`。"
        )
    else:
        lines.append("> 五组补全实验均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend(
        [
            "",
            "| 并发 | 方法 | 完成/失败 | Resolved | T450 | T500 | Prefill tok/s | Decode tok/s |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        lines.append(
            "| {c} | {method} | {done}/{failed} | {solved}/500 | {t450} s | "
            "{t500} s | {prefill} | {decode} |".format(
                c=result["concurrency"],
                method=core.METHOD_LABEL[result["method"]],
                done=result["completed"],
                failed=result["failed"],
                solved=result["solved"],
                t450=core.value(result["t450_seconds"]),
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
