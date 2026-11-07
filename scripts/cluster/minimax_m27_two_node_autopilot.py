#!/usr/bin/env python3
"""Run the MiniMax-M2.7 TP8/EP8 two-node Agentic-PD SWE500 matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.request

import qwen38_matrix_autopilot as core


core.JOB_ID = os.environ.get("MINIMAX_M27_JOB_ID", "25776117")
core.MIN_START_SECONDS = int(
    os.environ.get("MINIMAX_M27_MIN_START_SECONDS", 6 * 3600)
)
core.NODE0 = os.environ.get("MINIMAX_M27_PREFILL_NODE", "node1800")
core.NODE1 = os.environ.get("MINIMAX_M27_DECODE_NODE", "node2701")
core.MODEL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/models/MiniMax-M2.7-d494266a"
)
core.CONTROL_DIR = core.RUNS / "minimax-m27-tp8-two-node-autopilot-r2"
core.STATE_PATH = core.CONTROL_DIR / "state.json"
core.LOCK_PATH = core.CONTROL_DIR / "controller.lock"
core.REPORT = (
    core.ROOT / "runs-host/SWEBENCH_MINIMAX_M27_TP8_TWO_NODE.md"
)


def run_name(concurrency: int) -> str:
    return (
        f"minimax-m27-agentic-tp8ep8-two-node-swe500-c{concurrency}"
        "-20261015-r2"
    )


core.EXPERIMENTS = [
    {"concurrency": concurrency, "method": "agentic", "run_id": run_name(concurrency)}
    for concurrency in (64, 96, 128, 192, 256)
]
core.SEED_RUNS = []


def prepare_and_start(experiment: dict[str, Any]) -> None:
    directory = core.run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse existing next-run directory: {directory}")
    common = [
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
        "--launch-transport", "ssh",
    ]
    core.command([*common, "prepare"])
    core.command([*common, "--detach", "start-control"])
    core.command([*common, "control-ready"], timeout=900)
    core.command([*common, "--detach", "start-prefill"], timeout=2400)
    core.command([*common, "--detach", "start-decode"], timeout=900)
    core.command([*common, "wait-ready"], timeout=2400)
    core.command([*common, "--detach", "start-router"], timeout=600)
    core.command([*common, "--detach", "run-workload"], timeout=600)


core.prepare_and_start = prepare_and_start
# T450 is the experiment boundary.  The remaining verifier tail is useful for
# accuracy bookkeeping but must not hold the scarce two-node allocation or
# delay the next concurrency once 450 terminal episodes are durable.


START_MARKER = "<!-- MINIMAX_M27_AUTOPILOT_START -->"
END_MARKER = "<!-- MINIMAX_M27_AUTOPILOT_END -->"


def _completion_times(directory: Path) -> tuple[float | None, ...]:
    progress = core.load_progress(directory)
    timestamps = sorted(
        float(row["completed_at"])
        for row in progress
        if row.get("completed_at") is not None
    )
    marker = directory / "workload/measurement_start.json"
    if not marker.is_file():
        return (None, None, None, None, None)
    start = float(json.loads(marker.read_text())["measurement_start_wall"])
    values = [
        timestamps[index] - start if len(timestamps) > index else None
        for index in (49, 249, 449, 499)
    ]
    steady = None
    if values[0] is not None and values[2] is not None and values[2] > values[0]:
        steady = 400.0 / (values[2] - values[0])
    return (*values, steady)


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = []
    for experiment in core.EXPERIMENTS:
        directory = core.run_dir(experiment)
        result = core.completed_result(
            experiment["concurrency"], "agentic", directory
        ) or core.t450_result(
            experiment["concurrency"], "agentic", directory
        )
        if result is not None:
            t50, t250, t450, t500, steady = _completion_times(directory)
            result.update(
                {"t50": t50, "t250": t250, "t450": t450,
                 "t500": t500, "steady": steady}
            )
            results.append(result)
    current = (
        core.EXPERIMENTS[state["index"]]
        if state["index"] < len(core.EXPERIMENTS)
        else None
    )
    lines = [
        START_MARKER,
        "## 自动运行状态",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}；"
        f"状态：`{state['status']}`；正式结果 **{len(results)}/5**。",
    ]
    if current is not None:
        lines.append(
            f"> 当前：c{current['concurrency']}，episode progress "
            f"**{current_progress}/500**，run `{current['run_id']}`。"
        )
    elif state.get("status") == "complete":
        lines.append("> c64、c96、c128、c192、c256 五组均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend([
        "",
        "| 并发 | 完成/失败 | Resolved | T50 | T250 | **T450** | T500 | T50–T450 完成吞吐 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        lines.append(
            "| {c} | {done}/{failed} | {solved}/500 | {t50} s | {t250} s | "
            "{t450} s | {t500} s | {steady} req/s |".format(
                c=result["concurrency"], done=result["completed"],
                failed=result["failed"], solved=result["solved"],
                t50=core.value(result["t50"]),
                t250=core.value(result["t250"]),
                t450=core.value(result["t450"]),
                t500=core.value(result["t500"]),
                steady=core.value(result["steady"], 3),
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


_base_poll = core.poll


def _require_live_experiment(state: dict[str, Any]) -> None:
    """Fail one quiet watchdog tick unless the current run is demonstrably live."""

    if state.get("status") == "complete" or state["index"] >= len(core.EXPERIMENTS):
        return
    slurm_state, _seconds_left = core.job_state()
    if slurm_state != "RUNNING":
        raise RuntimeError(f"Slurm job {core.JOB_ID} is {slurm_state}")

    experiment = core.EXPERIMENTS[state["index"]]
    directory = core.run_dir(experiment)
    config_path = directory / "multinode-v3.json"
    if not config_path.is_file():
        raise RuntimeError(f"active run has no launch config: {config_path}")
    config = json.loads(config_path.read_text())
    endpoints = [("router", f"http://{config['nodes'][0]['host_ip']}:{config['router']['port']}/health")]
    endpoints.extend(
        (node["engine_id"], f"http://{node['host_ip']}:{node['port']}/model_info")
        for node in config["nodes"]
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    failures = []
    for label, url in endpoints:
        try:
            with opener.open(url, timeout=3.0) as response:
                if int(response.status) != 200:
                    failures.append(f"{label}=HTTP {response.status}")
        except Exception as error:
            failures.append(f"{label}={type(error).__name__}: {error}")
    if failures:
        raise RuntimeError("active experiment endpoint failure: " + "; ".join(failures))

    # One or two long verifier containers are normal near the tail.  Two full
    # watchdog intervals without any terminal completion are not normal for
    # this c64/c96/c128 matrix and require agent inspection.
    last_progress = float(state.get("last_progress_epoch") or 0.0)
    if last_progress and time.time() - last_progress > 20 * 60:
        raise RuntimeError(
            "active experiment made no terminal progress for more than 20 minutes"
        )


def poll_with_health_gate(state: dict[str, Any]) -> bool:
    finished = _base_poll(state)
    if not finished:
        _require_live_experiment(state)
    return finished


core.poll = poll_with_health_gate


if __name__ == "__main__":
    raise SystemExit(core.main())
