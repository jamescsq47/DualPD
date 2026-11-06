#!/usr/bin/env python3
"""Run the MiniMax-M2.7 c320/c384 two-node four-method matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import qwen38_matrix_autopilot as core


core.JOB_ID = os.environ.get("MINIMAX_M27_HIGH_JOB_ID", "26049949")
core.MIN_START_SECONDS = int(
    os.environ.get("MINIMAX_M27_HIGH_MIN_START_SECONDS", 4 * 3600)
)
core.NODE0 = os.environ.get("MINIMAX_M27_HIGH_NODE0", "node1800")
core.NODE1 = os.environ.get("MINIMAX_M27_HIGH_NODE1", "node2701")
core.MODEL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/models/MiniMax-M2.7-d494266a"
)
core.BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
core.BASELINE_ENGINE = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/sglang-src/pd_baseline_b1ef6"
)
core.CONTROL_DIR = core.RUNS / "minimax-m27-tp8-two-node-high-concurrency-r1"
core.STATE_PATH = core.CONTROL_DIR / "state.json"
core.LOCK_PATH = core.CONTROL_DIR / "controller.lock"
core.REPORT = (
    core.ROOT / "examples/pd/runs-host/SWEBENCH_MINIMAX_M27_TP8_TWO_NODE.md"
)


METHODS = ("agentic", "native", "hicache", "colocated")
METHOD_LABEL = {
    "agentic": "Agentic-PD",
    "native": "No-reverse PD",
    "hicache": "HiCache/Mooncake PD",
    "colocated": "Colocated",
}


def run_name(concurrency: int, method: str) -> str:
    label = {
        "agentic": "agentic",
        "native": "native-oneway-1p1d",
        "hicache": "hicache-mooncake-1p1d",
        "colocated": "colocated-2replica",
    }[method]
    return (
        f"minimax-m27-tp8ep8-two-node-{label}-swe500-c{concurrency}"
        "-20261017-r1"
    )


core.EXPERIMENTS = [
    {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_name(concurrency, method),
    }
    for concurrency in (320, 384)
    for method in METHODS
]
core.SEED_RUNS = []


def baseline_common(experiment: dict[str, Any]) -> list[str | Path]:
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
    if method == "agentic":
        common: list[str | Path] = [
            core.PD_PYTHON,
            core.AGENTIC_LAUNCHER,
            "--job-id", core.JOB_ID,
            "--prefill-node", core.NODE0,
            "--decode-node", core.NODE1,
            "--python", core.PD_PYTHON,
            "--sglang-root", core.PD_ENGINE,
            "--model", core.MODEL,
            "--model-family", "minimax_m2",
            "--data-root", core.DATA,
            "--swe-sif-dir", core.SIF,
            "--workload-config", core.WORKLOAD,
            "--run-dir", directory,
            "--run-id", experiment["run_id"],
            "--requests", "500",
            "--concurrency", str(experiment["concurrency"]),
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
            "--launch-transport", "srun",
        ]
        core.command([*common, "prepare"])
        core.command([*common, "--detach", "start-control"])
        core.command([*common, "control-ready"], timeout=900)
        core.command([*common, "--detach", "start-prefill"], timeout=2400)
        core.command([*common, "--detach", "start-decode"], timeout=900)
        core.command([*common, "wait-ready"], timeout=2400)
        core.command([*common, "--detach", "start-router"], timeout=600)
        core.command([*common, "--detach", "run-workload"], timeout=600)
        return

    if method == "colocated":
        core.command(
            [
                core.BASELINE_PYTHON,
                core.COLOCATED_LAUNCHER,
                "prepare",
                *baseline_common(experiment),
                "--groups-per-node", "1",
                "--node0-groups", "1",
                "--node1-groups", "1",
                "--cpus-per-node", "224",
            ]
        )
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

    prepare: list[str | Path] = [
        core.BASELINE_PYTHON,
        core.NATIVE_LAUNCHER,
        "prepare",
        *baseline_common(experiment),
        "--prefill-groups-per-node", "1",
        "--decode-groups-per-node", "1",
        "--mamba-cache-strategy-flag", "scheduler",
        "--local-root", f"/var/tmp/shangy/dualpd-minimax-high/{experiment['run_id']}",
        "--launch-transport", "srun",
    ]
    if method == "hicache":
        prepare.extend(
            [
                "--native-mooncake",
                "--disable-mlir-multithreading",
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


def stop_experiment(experiment: dict[str, Any]) -> None:
    directory = core.run_dir(experiment)
    method = experiment["method"]

    def stop(argv: list[str | Path], timeout: int) -> None:
        try:
            core.command(argv, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            core.log("stop request timed out; authoritative scoped cleanup follows")

    if method == "agentic":
        common: list[str | Path] = [
            core.PD_PYTHON,
            core.AGENTIC_LAUNCHER,
            "--job-id", core.JOB_ID,
            "--prefill-node", core.NODE0,
            "--decode-node", core.NODE1,
            "--python", core.PD_PYTHON,
            "--run-dir", directory,
            "--launch-transport", "srun",
        ]
        for component, role in (
            ("workload", None),
            ("router", None),
            ("worker", "decode"),
            ("worker", "prefill"),
            ("control", None),
        ):
            argv = [*common, "stop", "--component", component]
            if role:
                argv.extend(["--worker-role", role])
            stop(argv, 180)
    elif method == "colocated":
        stop(
            [
                core.BASELINE_PYTHON,
                core.COLOCATED_LAUNCHER,
                "stop",
                "--job-id", core.JOB_ID,
                "--run-dir", directory,
            ],
            300,
        )
    else:
        stop(
            [
                core.BASELINE_PYTHON,
                core.NATIVE_LAUNCHER,
                "stop",
                "--job-id", core.JOB_ID,
                "--run-dir", directory,
            ],
            300,
        )


core.stop_experiment = stop_experiment


SPAWN_MARKERS = (
    "resource temporarily unavailable",
    "cannot fork",
    "failed to spawn",
    "pipe != nullptr",
)


def hidden_spawn_errors(directory: Path) -> int:
    count = 0
    for row in core.load_progress(directory):
        verifier = row.get("verifier") or {}
        text = " ".join(
            str(value).lower()
            for value in (
                verifier.get("output_tail"),
                verifier.get("error"),
                row.get("environment_error"),
            )
            if value
        )
        if any(marker in text for marker in SPAWN_MARKERS):
            count += 1
    return count


_completed_result = core.completed_result
_t450_result = core.t450_result


def completed_result(
    concurrency: int, method: str, directory: Path
) -> dict[str, Any] | None:
    result = _completed_result(concurrency, method, directory)
    if result is not None:
        hidden = hidden_spawn_errors(directory)
        result["hidden_spawn_errors"] = hidden
        result["environment_errors"] += hidden
    return result


def t450_result(
    concurrency: int, method: str, directory: Path
) -> dict[str, Any] | None:
    result = _t450_result(concurrency, method, directory)
    if result is not None:
        hidden = hidden_spawn_errors(directory)
        result["hidden_spawn_errors"] = hidden
        result["environment_errors"] += hidden
    return result


core.completed_result = completed_result
core.t450_result = t450_result


START_MARKER = "<!-- MINIMAX_M27_HIGH_CONCURRENCY_START -->"
END_MARKER = "<!-- MINIMAX_M27_HIGH_CONCURRENCY_END -->"


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
        "## c320/c384 自动运行状态",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}；"
        f"状态：`{state['status']}`；正式结果 **{len(results)}/8**。",
    ]
    if current is not None:
        lines.append(
            f"> 当前：c{current['concurrency']} {METHOD_LABEL[current['method']]}，"
            f"episode progress **{current_progress}/500**，"
            f"run `{current['run_id']}`。"
        )
    elif state.get("status") == "complete":
        lines.append("> c320/c384 四种方法共八组均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend(
        [
            "",
            "| 并发 | 方法 | 完成/失败 | Resolved | **T450** | 0–T450 Decode tok/s | 异常 |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        lines.append(
            "| {c} | {method} | {done}/{failed} | {solved}/500 | "
            "{t450} s | {decode} | env={env}, spawn={spawn} |".format(
                c=result["concurrency"],
                method=METHOD_LABEL[result["method"]],
                done=result["completed"],
                failed=result["failed"],
                solved=result["solved"],
                t450=core.value(result["t450_seconds"]),
                decode=core.value(result.get("t450_decode_tps")),
                env=result["environment_errors"],
                spawn=result.get("hidden_spawn_errors", 0),
            )
        )
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


_poll = core.poll


def monitored_poll(state: dict[str, Any]) -> bool:
    finished = _poll(state)
    if finished or state["index"] >= len(core.EXPERIMENTS):
        return finished
    experiment = core.EXPERIMENTS[state["index"]]
    directory = core.run_dir(experiment)
    if not directory.exists():
        return finished
    hidden = hidden_spawn_errors(directory)
    if hidden > 2:
        raise RuntimeError(f"verifier spawn/fork errors={hidden}")
    started = float(state.get("last_started_epoch") or 0.0)
    if started and time.time() - started > 5 * 60:
        gpu_count = core.gpu_process_count()
        if gpu_count < 16:
            raise RuntimeError(
                f"active experiment has only {gpu_count}/16 GPU compute processes"
            )
    last_progress = float(state.get("last_progress_epoch") or 0.0)
    if last_progress and time.time() - last_progress > 25 * 60:
        raise RuntimeError("no terminal episode progress for more than 25 minutes")
    return finished


core.poll = monitored_poll


if __name__ == "__main__":
    raise SystemExit(core.main())
