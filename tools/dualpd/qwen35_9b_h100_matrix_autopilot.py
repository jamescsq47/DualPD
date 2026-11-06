#!/usr/bin/env python3
"""Run the single-node Qwen3.5-9B SWE500 H100 matrix serially.

The controller owns only experiment orchestration.  Every run is immutable.
The next run is not admitted until GPU processes, matrix listeners, Apptainer
processes, and filesystem helpers have all remained at zero.  If that hard
gate cannot be reached, the controller replaces the allocation before it can
advance the matrix.
"""

from __future__ import annotations

import argparse
import bisect
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
CONTROL_DIR = RUNS / "qwen35-9b-tp1-single-h100-miles-pr51-stock256-matrix-v28-fixed-evict-mlir1-r22"
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
REPORT = (
    ROOT
    / "examples/pd/runs-host/SWEBENCH_QWEN35_9B_TP1_H100_MILES_PR51.md"
)

JOB_ID = "25734937"
NODE = "node2800"
PD_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3.5-9B")
DATA = Path("/orcd/scratch/orcd/003/shangy/dualpd/datasets")
SIF = Path("/orcd/pool/004/shangy/dualpd/containers/swebench-verified-sif")
WORKLOAD = (
    ROOT
    / "examples/pd/configs/experiments/"
    "swe_bench_verified_miles_pr51_8k_t64_500_apptainer.yaml"
)
PD_ENGINE = WORKSPACE / "sglang_pd"
BASELINE_ENGINE = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/sglang-src/pd_baseline_v0510_compat"
)
MOONCAKE_COMPAT = (
    ROOT / "examples/pd/scripts/compat/qwen35_mooncake"
)
FIXED_MOONCAKE_MASTER = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/"
    "mooncake-v0.3.12.post1-fixed-evict/build-fixed/"
    "mooncake-store/src/mooncake_master"
)
AGENTIC_LAUNCHER = ROOT / "tools/dualpd/qwen35_slurm.py"
COLOCATED_LAUNCHER = ROOT / "tools/dualpd/qwen35_colocated_slurm.py"
NATIVE_LAUNCHER = ROOT / "tools/dualpd/qwen35_native_pd_slurm.py"

METHOD_LABEL = {
    "agentic": "当前新方法（2P:6D）",
    "colocated": "Colocated（8副本）",
    "native": "No-reverse PD（2P:6D）",
    "hicache": "原生 HiCache/Mooncake（2P:6D）",
}
METHOD_ORDER = ("agentic", "colocated", "native", "hicache")
CONCURRENCIES = (128, 192, 256)
# H100 TP1 leaves about 30.14 GiB of Attention KV plus an observed 21.40 GiB
# of recurrent state on each worker at mem_fraction_static=0.8.  Stock HiCache
# requires each host tier to be strictly larger than its device tier.  These
# per-rank capacities satisfy that invariant and total 456 GB across 8 ranks.
HICACHE_ATTENTION_GB_PER_RANK = 33
HICACHE_MAMBA_GB_PER_RANK = 24

# Union of every listener owned by the four launchers.  Admission and cleanup
# check the whole set so a stale router/bootstrap/Mooncake process cannot make
# a later method fail with an ambiguous bind error.
MATRIX_PORTS = tuple(sorted({
    23900, 23901, 23902, 23903, 23904, 23910, 23911,
    24000, 24001, 24002, 24003, 24010, 24011, 24020, 24021, 24030, 24031,
    24040, 24041, 24050, 24051, 24060, 24070,
    24200, 24210,
    24300, 24310, 24320, 24330, 24340, 24350,
    24400, 24401, 24500, 24510,
    25103, 25151, 25152, 25180, 25190,
    61900, 61910, 62000, 62010, 62020, 62030, 62040, 62050,
}))
# One router request is allowed to wait 3600s.  Leave enough watchdog margin
# for that timeout to fire and for the workload's bounded HTTP retry to make
# forward progress before declaring the entire experiment stalled.
NO_PROGRESS_TIMEOUT_SECONDS = 5400
# Normal SWE sandbox bursts can exceed 12k threads even at c192, while the
# kernel/PID namespace limit is orders of magnitude higher.  Keep this only as
# a last-resort runaway detector; physical-memory, service-health, and
# zero-residue gates are the normal safety mechanisms.
MAX_USER_THREADS = 100_000
# Committed_AS is not a hard capacity limit on these nodes: vm.overcommit_memory
# is 0 and CommitLimit is only about half of the 2 TiB of physical RAM.  The
# 768 GiB of intentionally reserved Host KV therefore pushes the ratio above
# one while more than 1 TiB of RAM is still available.  Gate on real available
# memory instead and retain enough headroom for cleanup and Slurm daemons.
MIN_MEM_AVAILABLE_KIB = 256 * 1024 * 1024
SANDBOX_HELPER_NAMES = (
    "apptainer",
    "appinit",
    "starter",
    "starter-suid",
    "squashfuse_ll",
    "fuse2fs",
)
DECODE_COUNTER = "sglang_realtime_tokens_total|mode=decode"


class DirtyAllocationError(RuntimeError):
    """The node failed the post-run zero-residue gate."""


def run_name(concurrency: int, method: str) -> str:
    names = {
        "agentic": "agentic-2p6d",
        "colocated": "colocated8",
        "native": "native-2p6d",
        "hicache": "hicache-2p6d",
    }
    completed_c256 = {
        "agentic": "q35-9b-h100-agentic-2p6d-swe500-miles-c256-20261010-r2",
        "colocated": "q35-9b-h100-colocated8-swe500-miles-c256-20261010-r3",
        "native": "q35-9b-h100-native-oneway-2p6d-swe500-miles-c256-20261010-r2",
    }
    if concurrency == 256 and method in completed_c256:
        return completed_c256[method]
    # Preserve every completed artifact.  Failed c256 HiCache directories
    # remain immutable, so only that final cell receives a fresh r22 ID.
    if concurrency == 128 and method != "hicache":
        suffix = "r15"
    elif concurrency == 192 and method != "hicache":
        suffix = "r16"
    elif concurrency == 256 and method == "hicache":
        suffix = "r22"
    else:
        suffix = "r17"
    return (
        f"q35-9b-h100-{names[method]}-"
        f"swe500-miles-stock256-c{concurrency}-20261012-{suffix}"
    )


ALL_EXPERIMENTS = [
    {"concurrency": concurrency, "method": method,
     "run_id": run_name(concurrency, method)}
    for concurrency in CONCURRENCIES
    for method in METHOD_ORDER
]
# Reuse the three already validated c256 non-HiCache artifacts.  Run the six
# new lower-concurrency comparisons first, then place all three HiCache runs at
# the end so an unresolved baseline issue cannot block the primary comparison.
EXPERIMENTS = [
    experiment
    for experiment in ALL_EXPERIMENTS
    if experiment["concurrency"] in {128, 192}
    and experiment["method"] != "hicache"
] + [
    experiment
    for experiment in ALL_EXPERIMENTS
    if experiment["method"] == "hicache"
]

START_MARKER = "<!-- AUTOPILOT_RESULTS_START -->"
END_MARKER = "<!-- AUTOPILOT_RESULTS_END -->"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{stamp}] {message}", flush=True)


def command(
    argv: list[str | Path], *, timeout: int = 3600, check: bool = True
) -> subprocess.CompletedProcess[str]:
    rendered = [str(value) for value in argv]
    log("exec: " + " ".join(rendered))
    completed = subprocess.run(
        rendered,
        cwd=WORKSPACE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    if check and completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(rendered)}"
        )
    return completed


def on_node(argv: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Run a lightweight probe inside the sole H100 allocation.

    Runtime components use the already validated SSH path.  Lightweight
    health/admission probes use an overlapping one-CPU Slurm step so every
    check is tied to the current allocation.
    """

    rendered = [str(value) for value in argv]
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, 4):
        try:
            last = command(
                [
                    "srun",
                    "--jobid", JOB_ID,
                    "--overlap",
                    "--exact",
                    "--nodes=1",
                    "--ntasks=1",
                    "--cpus-per-task=1",
                    "-w", NODE,
                    *rendered,
                ],
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log(f"node probe attempt {attempt}/3 timed out: {' '.join(rendered)}")
        else:
            if last.returncode == 0:
                return last
        if attempt < 3:
            time.sleep(attempt)
    if last is None:
        raise RuntimeError(
            f"node probe timed out after 3 attempts: {' '.join(rendered)}"
        )
    raise RuntimeError(
        "node probe failed after 3 attempts with exit "
        f"{last.returncode}: {' '.join(rendered)}"
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_allocation(status: str = "running") -> None:
    atomic_json(
        ALLOCATION_PATH,
        {
            "job_id": JOB_ID,
            "node": NODE,
            "status": status,
            "updated_epoch": time.time(),
        },
    )


def load_allocation() -> None:
    global JOB_ID, NODE
    if not ALLOCATION_PATH.is_file():
        save_allocation()
        return
    allocation = json.loads(ALLOCATION_PATH.read_text())
    JOB_ID = str(allocation["job_id"])
    NODE = str(allocation.get("node") or NODE)


def run_dir(experiment: dict[str, Any]) -> Path:
    return RUNS / experiment["run_id"]


def load_progress(directory: Path) -> list[dict[str, Any]]:
    path = directory / "episode_progress.jsonl"
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = row.get("instance_id")
            if instance_id:
                latest[str(instance_id)] = row
    return list(latest.values())


def _counter_at_boundary(
    points: list[tuple[float, float]], boundary: float
) -> float:
    """Interpolate a monotonic counter at a wall-clock boundary."""
    timestamps = [point[0] for point in points]
    index = bisect.bisect_left(timestamps, boundary)
    if index == 0:
        return points[0][1]
    if index == len(points):
        return points[-1][1]
    before, after = points[index - 1], points[index]
    if after[0] == before[0]:
        return after[1]
    fraction = (boundary - before[0]) / (after[0] - before[0])
    return before[1] + fraction * (after[1] - before[1])


def t450_decode_metrics(
    directory: Path, workload_start: float, t450_end: float
) -> tuple[float | None, float | None]:
    """Return decode tokens and throughput from workload start through T450.

    Counters are differenced per endpoint before summing so a temporarily
    missing endpoint cannot make an aggregate counter appear to decrease.
    """
    metrics_path = directory / "workload/engine_metrics.jsonl"
    if not metrics_path.is_file() or t450_end <= workload_start:
        return None, None
    series: dict[str, list[tuple[float, float]]] = {}
    expected_endpoints = 0
    try:
        with metrics_path.open(errors="replace") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("role") != "decode":
                    continue
                expected_endpoints = max(
                    expected_endpoints, int(row.get("engine_count", 0))
                )
                for endpoint in row.get("endpoint_metrics", []):
                    value = endpoint.get("metrics", {}).get(DECODE_COUNTER)
                    if value is None:
                        continue
                    series.setdefault(str(endpoint["endpoint"]), []).append(
                        (float(row["ts"]), float(value))
                    )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None
    if not series or len(series) < expected_endpoints:
        return None, None
    total = 0.0
    for points in series.values():
        points.sort()
        if len(points) < 2:
            return None, None
        start_value = _counter_at_boundary(points, workload_start)
        end_value = _counter_at_boundary(points, t450_end)
        total += max(0.0, end_value - start_value)
    return total, total / (t450_end - workload_start)


def completed_result(experiment: dict[str, Any]) -> dict[str, Any] | None:
    directory = run_dir(experiment)
    if (directory / "INVALID.md").is_file():
        return None
    summary_path = directory / "workload/summary.json"
    requests_path = directory / "workload/requests.jsonl"
    progress = load_progress(directory)
    if (
        len(progress) != 500
        or not summary_path.is_file()
        or not requests_path.is_file()
    ):
        return None
    summary = json.loads(summary_path.read_text())
    if int(summary.get("requests", -1)) != 500:
        return None
    failed = int(summary.get("failed", 0))
    completed = int(summary.get("completed", -1))
    # Match the A100 Qwen3.5-9B accounting: a small number of SWE environment
    # failures are terminal samples and remain in the 500-task denominator.
    # Requiring failed==0 makes a complete comparable run look unfinished and
    # strands the serial controller after the workload has exited.
    if completed + failed != 500:
        return None
    request_rows = []
    with requests_path.open(errors="replace") as stream:
        for line in stream:
            try:
                request_rows.append(json.loads(line))
            except json.JSONDecodeError:
                return None
    if len(request_rows) != 500:
        return None
    scheduled = [
        float(row["scheduled_ts"])
        for row in request_rows
        if row.get("scheduled_ts") is not None
    ]
    if len(scheduled) != 500:
        return None
    timestamps = sorted(
        float(row["completed_at"])
        for row in progress
        if row.get("completed_at") is not None
    )
    if len(timestamps) != 500:
        return None
    workload_start = min(scheduled)
    t450 = timestamps[449] - workload_start
    t450_decode_tokens, t450_decode_tps = t450_decode_metrics(
        directory, workload_start, timestamps[449]
    )
    return {
        "concurrency": experiment["concurrency"],
        "method": experiment["method"],
        "run_id": experiment["run_id"],
        "directory": str(directory),
        "completed": 500,
        "normal_completed": completed,
        "failed": failed,
        "environment_errors": sum(
            bool(row.get("environment_error")) for row in progress
        ),
        "solved": sum(float(row.get("reward") or 0) > 0 for row in progress),
        "t450_seconds": t450,
        "t500_seconds": timestamps[499] - workload_start,
        "t50_seconds": timestamps[49] - workload_start,
        "t250_seconds": timestamps[249] - workload_start,
        "t50_t450_agent_s": 400.0 / (timestamps[449] - timestamps[49]),
        "t450_decode_tokens": t450_decode_tokens,
        "t450_decode_tps": t450_decode_tps,
        "scheduler_prefill_tps": summary.get(
            "scheduler_prefill_compute_tokens_per_second"
        ),
        "scheduler_decode_tps": summary.get(
            "scheduler_decode_tokens_per_second"
        ),
        "max_queue": summary.get("max_engine_queue_requests"),
    }


def all_results() -> list[dict[str, Any]]:
    results = []
    for experiment in ALL_EXPERIMENTS:
        result = completed_result(experiment)
        if result:
            results.append(result)
    return results


def value(number: Any, digits: int = 2) -> str:
    return "—" if number is None else f"{float(number):,.{digits}f}"


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = all_results()
    result_by_run = {result["run_id"]: result for result in results}
    current = EXPERIMENTS[state["index"]] if state["index"] < len(EXPERIMENTS) else None
    lines = [
        START_MARKER,
        "## 自动运行进度与结果",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}。",
        f"> 状态：`{state['status']}`；当前完整产物 **{len(results)}/12**；"
        "执行顺序为c128/c192非HiCache，最后执行c128/c192/c256 HiCache。",
    ]
    if current:
        current_rows = load_progress(run_dir(current))
        terminal_records = len(current_rows) if current_rows else current_progress
        normal_records = sum(
            str(row.get("status")) == "Status.COMPLETED" for row in current_rows
        )
        environment_errors = sum(
            bool(row.get("environment_error")) for row in current_rows
        )
        detail = ""
        if current_rows:
            detail = (
                f"（正常完成{normal_records}，environment error "
                f"{environment_errors}）"
            )
        lines.append(
            f"> 当前：c{current['concurrency']} {METHOD_LABEL[current['method']]}，"
            f"terminal records **{terminal_records}/500**{detail}，"
            f"run `{current['run_id']}`。"
        )
    elif state["status"] == "complete":
        lines.append("> 新增9组实验均已完成；加上复用的c256三组，共12组。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend([
        "",
        "### 核心指标",
        "",
        "| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for experiment in ALL_EXPERIMENTS:
        result = result_by_run.get(experiment["run_id"])
        if result is None:
            lines.append(
                f"| {experiment['concurrency']} | "
                f"{METHOD_LABEL[experiment['method']]} | — | — | — | — | — |"
            )
            continue
        decode_tokens = result["t450_decode_tokens"]
        lines.append(
            "| {c} | {method} | {completed}/{failed} | {solved}/500 | **{t450}** | **{decode_tps}** | {decode_tokens} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                completed=result["normal_completed"], failed=result["failed"],
                solved=result["solved"], t450=value(result["t450_seconds"]),
                decode_tps=value(result["t450_decode_tps"]),
                decode_tokens=(
                    "—" if decode_tokens is None else f"{decode_tokens / 1e6:.3f}M"
                ),
            )
        )
    lines.extend([
        "",
        "### 全程与辅助指标",
        "",
        "| 并发 | 方法 | T50 (s) | T250 (s) | T500 (s) | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for experiment in ALL_EXPERIMENTS:
        result = result_by_run.get(experiment["run_id"])
        if result is None:
            lines.append(
                f"| {experiment['concurrency']} | "
                f"{METHOD_LABEL[experiment['method']]} | — | — | — | — | — | — | — | "
                "未形成有效T450 |"
            )
            continue
        caveat = []
        if result["environment_errors"]:
            caveat.append(f"env={result['environment_errors']}")
        if result["failed"]:
            caveat.append(f"failed={result['failed']}")
        lines.append(
            "| {c} | {method} | {t50} | {t250} | {t500} | {rate} | {p} | {d} | {q} | {note} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                t50=value(result["t50_seconds"]),
                t250=value(result["t250_seconds"]),
                t500=value(result["t500_seconds"]),
                rate=value(result["t50_t450_agent_s"], 5),
                p=value(result["scheduler_prefill_tps"]),
                d=value(result["scheduler_decode_tps"]),
                q=value(result["max_queue"], 0), note=", ".join(caveat) or "—",
            )
        )
    lines.extend(["", "原始目录：", ""])
    for result in results:
        lines.append(
            f"- c{result['concurrency']} {METHOD_LABEL[result['method']]}: "
            f"`{result['directory']}`"
        )
    lines.extend([END_MARKER, ""])
    block = "\n".join(lines)
    text = REPORT.read_text()
    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(before.rstrip() + "\n\n" + block + after.lstrip("\n"))
    os.replace(temporary, REPORT)


def prepare_and_start(experiment: dict[str, Any]) -> None:
    directory = run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse run directory: {directory}")
    method = experiment["method"]
    common_paths = [
        "--job-id", JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", directory,
        "--model", MODEL,
        "--data-root", DATA,
        "--swe-sif-dir", SIF,
        "--workload-config", WORKLOAD,
        "--requests", "500",
        "--concurrency", str(experiment["concurrency"]),
        "--tp-size", "1",
        "--reasoning-parser", "qwen3",
        "--deterministic-inference",
    ]
    if method == "agentic":
        common = [
            PD_PYTHON, AGENTIC_LAUNCHER,
            *common_paths,
            "--prefill-node", NODE,
            "--decode-node", NODE,
            "--python", PD_PYTHON,
            "--sglang-root", PD_ENGINE,
            "--single-node-2p6d",
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
        command([*common, "prepare"])
        command([*common, "--detach", "start-control"])
        command([*common, "control-ready"], timeout=900)
        command([*common, "--detach", "start-prefill"], timeout=2400)
        command([*common, "--detach", "start-decode"], timeout=1200)
        command([*common, "wait-ready"], timeout=2400)
        command([*common, "--detach", "start-router"], timeout=600)
        command([*common, "--detach", "run-workload"], timeout=600)
        return
    if method == "colocated":
        prepare = [
            PD_PYTHON, COLOCATED_LAUNCHER, "prepare",
            *common_paths,
            "--python", BASELINE_PYTHON,
            "--workload-python", PD_PYTHON,
            "--sglang-root", BASELINE_ENGINE,
            "--single-node",
            "--groups-per-node", "8",
            "--node0-groups", "8",
            "--cpus-per-node", "224",
        ]
        command(prepare)
        command(
            [PD_PYTHON, COLOCATED_LAUNCHER, "start", "--job-id", JOB_ID,
             "--run-dir", directory],
            timeout=3600,
        )
        return
    prepare = [
        BASELINE_PYTHON, NATIVE_LAUNCHER, "prepare",
        *common_paths,
        "--python", BASELINE_PYTHON,
        "--workload-python", PD_PYTHON,
        "--sglang-root", BASELINE_ENGINE,
        "--single-node-2p6d",
        "--prefill-groups-per-node", "2",
        "--decode-groups-per-node", "6",
        "--launch-transport", "ssh",
    ]
    if method == "hicache":
        prepare.extend([
            "--native-mooncake",
            "--mooncake-compat", MOONCAKE_COMPAT,
            "--mooncake-master-binary", FIXED_MOONCAKE_MASTER,
            "--disable-mlir-multithreading",
            "--prefill-hicache-size-gb", str(HICACHE_ATTENTION_GB_PER_RANK),
            "--decode-hicache-size-gb", str(HICACHE_ATTENTION_GB_PER_RANK),
            "--mamba-hicache-size-gb", str(HICACHE_MAMBA_GB_PER_RANK),
            "--mooncake-store-size-gb", "256",
            "--mooncake-prefetch-policy", "wait_complete",
            "--workload-job-id", JOB_ID,
            "--workload-node", NODE,
            "--storage-job-id", JOB_ID,
            "--storage-node", NODE,
        ])
    command(prepare)
    command(
        [BASELINE_PYTHON, NATIVE_LAUNCHER, "start", "--job-id", JOB_ID,
         "--run-dir", directory],
        timeout=3600,
    )


def stop_experiment(experiment: dict[str, Any]) -> None:
    directory = run_dir(experiment)
    method = experiment["method"]
    if method == "agentic":
        common = [
            PD_PYTHON, AGENTIC_LAUNCHER,
            "--job-id", JOB_ID,
            "--prefill-node", NODE,
            "--decode-node", NODE,
            "--python", PD_PYTHON,
            "--run-dir", directory,
            "--single-node-2p6d",
            "--tp-size", "1",
            "--launch-transport", "ssh",
        ]
        for component, role in (
            ("workload", None), ("router", None), ("worker", "decode"),
            ("worker", "prefill"), ("control", None),
        ):
            args = [*common, "stop", "--component", component]
            if role:
                args.extend(["--worker-role", role])
            command(args, timeout=900, check=False)
    elif method == "colocated":
        command(
            [PD_PYTHON, COLOCATED_LAUNCHER, "stop", "--job-id", JOB_ID,
             "--run-dir", directory], timeout=1800, check=False
        )
    else:
        command(
            [BASELINE_PYTHON, NATIVE_LAUNCHER, "stop", "--job-id", JOB_ID,
             "--run-dir", directory], timeout=1800, check=False
        )


def gpu_process_count() -> int:
    completed = on_node(
        ["bash", "-lc",
         "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
         "2>/dev/null | sed '/^$/d' | wc -l"],
        timeout=60,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def sandbox_helper_counts() -> dict[str, int]:
    """Count every sandbox helper owned by this user on the exclusive node.

    This intentionally does not filter on run ID or process group.  Apptainer
    may re-parent squashfuse_ll to PID 1 and its argv then contains neither.
    Exact comm names plus UID keep the check narrow while still detecting the
    residue that exhausted the previous long-lived allocation.
    """

    probe = (
        "import json,os,sys\n"
        "names=tuple(sys.argv[1:]); counts={name:0 for name in names}; uid=os.getuid()\n"
        "for entry in os.listdir('/proc'):\n"
        "  if not entry.isdigit():continue\n"
        "  root='/proc/'+entry\n"
        "  try:\n"
        "    if os.stat(root).st_uid != uid:continue\n"
        "    comm=open(root+'/comm',errors='replace').read().strip()\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError):continue\n"
        "  if comm in counts:counts[comm]+=1\n"
        "print(json.dumps(counts,sort_keys=True))\n"
    )
    completed = on_node(
        ["python3", "-c", probe, *SANDBOX_HELPER_NAMES],
        timeout=60,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def node_resource_snapshot() -> dict[str, float | int]:
    """Return node-level process and commit pressure for the live safety gate."""

    probe = (
        "import json,os\n"
        "uid=os.getuid(); pids=threads=0\n"
        "for entry in os.listdir('/proc'):\n"
        "  if not entry.isdigit():continue\n"
        "  root='/proc/'+entry\n"
        "  try:\n"
        "    if os.stat(root).st_uid != uid:continue\n"
        "    threads+=len(os.listdir(root+'/task')); pids+=1\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError):continue\n"
        "mem={}\n"
        "for line in open('/proc/meminfo'):\n"
        "  key,value,*_=line.replace(':','').split()\n"
        "  if key in {'MemAvailable','CommitLimit','Committed_AS'}:mem[key]=int(value)\n"
        "print(json.dumps({'user_pids':pids,'user_threads':threads,"
        "'mem_available_kib':mem['MemAvailable'],'commit_limit_kib':mem['CommitLimit'],"
        "'committed_as_kib':mem['Committed_AS'],"
        "'commit_ratio':mem['Committed_AS']/mem['CommitLimit']},sort_keys=True))\n"
    )
    completed = on_node(["python3", "-c", probe], timeout=60)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def runtime_pressure_issue() -> str:
    snapshot = node_resource_snapshot()
    threads = int(snapshot["user_threads"])
    mem_available_kib = int(snapshot["mem_available_kib"])
    if threads >= MAX_USER_THREADS:
        return f"user thread safety limit reached: {threads} >= {MAX_USER_THREADS}"
    if mem_available_kib <= MIN_MEM_AVAILABLE_KIB:
        return (
            "available-memory safety limit reached: "
            f"{mem_available_kib} KiB <= {MIN_MEM_AVAILABLE_KIB} KiB"
        )
    return ""


def terminate_sandbox_helpers() -> None:
    """Terminate exact-name helpers owned by this user on the exclusive node."""

    script = (
        "import json,os,signal,sys,time\n"
        "names=set(sys.argv[1:]); uid=os.getuid(); excluded={os.getpid(),os.getppid()}\n"
        "def targets():\n"
        "  found=[]\n"
        "  for entry in os.listdir('/proc'):\n"
        "    if not entry.isdigit() or int(entry) in excluded:continue\n"
        "    root='/proc/'+entry\n"
        "    try:\n"
        "      if os.stat(root).st_uid != uid:continue\n"
        "      comm=open(root+'/comm',errors='replace').read().strip()\n"
        "    except (FileNotFoundError,PermissionError,ProcessLookupError):continue\n"
        "    if comm in names:found.append(int(entry))\n"
        "  return found\n"
        "initial=targets()\n"
        "for pid in initial:\n"
        "  try:os.kill(pid,signal.SIGTERM)\n"
        "  except ProcessLookupError:pass\n"
        "deadline=time.monotonic()+20\n"
        "while targets() and time.monotonic()<deadline:time.sleep(.5)\n"
        "remaining=targets()\n"
        "for pid in remaining:\n"
        "  try:os.kill(pid,signal.SIGKILL)\n"
        "  except ProcessLookupError:pass\n"
        "print(json.dumps({'initial':len(initial),'killed':len(remaining)}))\n"
    )
    on_node(
        ["python3", "-c", script, *SANDBOX_HELPER_NAMES],
        timeout=60,
    )


def scoped_process_count(run_id: str) -> int:
    probe = (
        "import os,sys\n"
        "needle=sys.argv[1].encode()\n"
        "excluded={os.getpid(),os.getppid()}\n"
        "def hit(name):\n"
        "  if not name.isdigit() or int(name) in excluded:return False\n"
        "  try:return needle in open('/proc/'+name+'/cmdline','rb').read()\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError):return False\n"
        "print(sum(hit(name) for name in os.listdir('/proc')))\n"
    )
    completed = on_node(
        ["python3", "-c", probe, run_id],
        timeout=60,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def workload_process_count(run_id: str) -> int:
    probe = (
        "import os,sys\n"
        "needle=sys.argv[1].encode()\n"
        "excluded={os.getpid(),os.getppid()}\n"
        "def hit(name):\n"
        "  if not name.isdigit() or int(name) in excluded:return False\n"
        "  try:cmd=open('/proc/'+name+'/cmdline','rb').read()\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError):return False\n"
        "  return needle in cmd and (b'dualpd.inference.py' in cmd or b'/workload' in cmd)\n"
        "print(sum(hit(name) for name in os.listdir('/proc')))\n"
    )
    completed = on_node(
        ["python3", "-c", probe, run_id],
        timeout=60,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def terminate_scoped_processes(run_id: str) -> None:
    """Terminate only orphaned processes whose argv names this exact run."""

    script = (
        "import os,signal,sys,time\n"
        "needle=sys.argv[1].encode()\n"
        "excluded={os.getpid(),os.getppid()}\n"
        "targets=[]\n"
        "for name in os.listdir('/proc'):\n"
        "  if not name.isdigit() or int(name) in excluded:continue\n"
        "  try:\n"
        "    pid=int(name); cmd=open('/proc/'+name+'/cmdline','rb').read()\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError):continue\n"
        "  if needle in cmd:targets.append(pid)\n"
        "groups=set()\n"
        "for pid in targets:\n"
        "  try:groups.add(os.getpgid(pid))\n"
        "  except ProcessLookupError:pass\n"
        "groups.discard(os.getpgrp())\n"
        "for group in groups:\n"
        "  try:os.killpg(group,signal.SIGTERM)\n"
        "  except ProcessLookupError:pass\n"
        "time.sleep(5)\n"
        "for group in groups:\n"
        "  try:os.killpg(group,signal.SIGKILL)\n"
        "  except ProcessLookupError:pass\n"
        "print(len(targets))\n"
    )
    on_node(
        ["python3", "-c", script, run_id],
        timeout=60,
    )


def occupied_matrix_ports() -> list[int]:
    wanted = ",".join(map(str, MATRIX_PORTS))
    probe = (
        "wanted=set(map(int,sys.argv[1].split(',')))\n"
        "occupied=set()\n"
        "for line in subprocess.check_output(['ss','-H','-ltn'],text=True).splitlines():\n"
        "  address=line.split()[3]\n"
        "  try:port=int(address.rsplit(':',1)[1])\n"
        "  except (IndexError,ValueError):continue\n"
        "  if port in wanted:occupied.add(port)\n"
        "print(','.join(map(str,sorted(occupied))))\n"
    )
    completed = on_node(
        ["python3", "-c", "import subprocess,sys\n" + probe, wanted],
        timeout=60,
    )
    last = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return [int(value) for value in last.split(",") if value]


def expected_runtime_ports(experiment: dict[str, Any]) -> set[int]:
    if experiment["method"] == "agentic":
        return {23900, 23910, 24000, 24010, 24020, 24030, 24040, 24050}
    if experiment["method"] == "native":
        return {
            24200, 24210, 24300, 24310, 24320, 24330, 24340, 24350,
            24400, 24401,
        }
    if experiment["method"] == "hicache":
        return {
            24200, 24210, 24300, 24310, 24320, 24330, 24340, 24350,
            24400, 24401,
            25103, 25151, 25152, 25180, 25190,
        }
    # Colocated uses eight independent OpenAI endpoints.
    return {24000, 24010, 24020, 24030, 24040, 24050, 24060, 24070}


def runtime_health_issue(experiment: dict[str, Any]) -> str:
    active_gpus = gpu_process_count()
    if active_gpus != 8:
        return f"expected 8 GPU processes, found {active_gpus}"
    occupied = set(occupied_matrix_ports())
    missing = expected_runtime_ports(experiment) - occupied
    if missing:
        return "missing service ports " + ",".join(map(str, sorted(missing)))
    return ""


def assert_clean_for_start() -> None:
    gpu_count = gpu_process_count()
    ports = occupied_matrix_ports()
    helpers = sandbox_helper_counts()
    if gpu_count or ports or any(helpers.values()):
        raise RuntimeError(
            f"{NODE} is not clean for admission: GPU={gpu_count}, ports={ports}, "
            f"sandbox_helpers={helpers}"
        )


def wait_clean(run_id: str) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        gpu_count = gpu_process_count()
        scoped_count = scoped_process_count(run_id)
        ports = occupied_matrix_ports()
        helpers = sandbox_helper_counts()
        if (
            gpu_count == 0
            and scoped_count == 0
            and not ports
            and not any(helpers.values())
        ):
            time.sleep(10)
            if (
                gpu_process_count() == 0
                and scoped_process_count(run_id) == 0
                and not occupied_matrix_ports()
                and not any(sandbox_helper_counts().values())
            ):
                return
        log(
            f"waiting for exact stop: GPU={gpu_count}, scoped={scoped_count}, "
            f"ports={ports}, sandbox_helpers={helpers}"
        )
        time.sleep(15)
    raise DirtyAllocationError(
        "completed run did not release all GPU processes, run-scoped processes, "
        "listeners, and sandbox helpers"
    )


def wait_for_allocation(job_id: str, timeout: int = 3600) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completed = command(
            ["squeue", "-h", "-j", job_id, "-o", "%T|%N|%R"],
            timeout=30,
            check=False,
        )
        row = completed.stdout.strip().splitlines()
        if row:
            state, node, reason = row[-1].split("|", 2)
            if state == "RUNNING" and node and node not in {"(null)", "(None)"}:
                return node
            if state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}:
                raise RuntimeError(
                    f"replacement allocation {job_id} entered {state}: {reason}"
                )
        time.sleep(10)
    raise RuntimeError(
        f"replacement allocation {job_id} did not start within {timeout}s"
    )


def rebuild_allocation(reason: str) -> None:
    """Cancel the dirty allocation and admit work only on a clean replacement."""

    global JOB_ID, NODE
    old_job_id = JOB_ID
    log(f"hard gate failed ({reason}); cancelling allocation {old_job_id}")
    command(["scancel", old_job_id], timeout=60, check=False)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        queued = command(
            ["squeue", "-h", "-j", old_job_id, "-o", "%T"],
            timeout=30,
            check=False,
        ).stdout.strip()
        if not queued:
            break
        time.sleep(5)
    else:
        raise RuntimeError(
            f"dirty allocation {old_job_id} did not terminate within 600s"
        )

    submitted = command(
        [
            "sbatch", "--parsable",
            "--job-name=dualpd-qwen35-9b-swe-h100-1n",
            "--partition=pi_songhan",
            "--account=mit_general",
            "--nodes=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=224",
            "--gres=gpu:h100:8",
            "--mem=1800G",
            "--exclusive",
            "--time=7-00:00:00",
            "--nodelist=node2700",
            f"--output={WORKSPACE}/slurm-agentic-qwen35-swe-allocation-%j.out",
            '--wrap=trap "exit 0" TERM INT; while true; do sleep 300; done',
        ],
        timeout=60,
    )
    JOB_ID = submitted.stdout.strip().splitlines()[-1].split(";", 1)[0]
    NODE = ""
    save_allocation("pending")
    NODE = wait_for_allocation(JOB_ID)
    save_allocation("running")
    command(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            NODE, f"grep -q job_{JOB_ID} /proc/self/cgroup",
        ],
        timeout=30,
    )
    assert_clean_for_start()
    log(f"replacement allocation {JOB_ID} on {NODE} passed the hard zero gate")


def cleanup_experiment(experiment: dict[str, Any]) -> None:
    try:
        if run_dir(experiment).exists():
            stop_experiment(experiment)
        if scoped_process_count(experiment["run_id"]):
            log("terminating exact-run orphaned processes after graceful stop")
            terminate_scoped_processes(experiment["run_id"])
        if workload_process_count(experiment["run_id"]):
            raise DirtyAllocationError(
                "workload is still live; refusing to kill shared sandbox helpers"
            )
        helpers = sandbox_helper_counts()
        if any(helpers.values()):
            log(f"terminating sandbox helper residue: {helpers}")
            terminate_sandbox_helpers()
        wait_clean(experiment["run_id"])
    except Exception as error:
        rebuild_allocation(f"{type(error).__name__}: {error}")


def start_experiment(experiment: dict[str, Any]) -> None:
    assert_clean_for_start()
    try:
        prepare_and_start(experiment)
    except BaseException:
        log(f"start failed for {experiment['run_id']}; cleaning partial run")
        try:
            cleanup_experiment(experiment)
        except BaseException as cleanup_error:
            log(f"partial-run cleanup also failed: {cleanup_error}")
        raise


def job_seconds_left() -> int:
    output = command(["scontrol", "show", "job", "-o", JOB_ID], timeout=30).stdout
    fields = dict(token.split("=", 1) for token in output.split() if "=" in token)
    if fields.get("JobState") != "RUNNING":
        raise RuntimeError(f"Slurm job {JOB_ID} is {fields.get('JobState')}")
    end_epoch = int(
        subprocess.check_output(
            ["date", "-u", "-d", fields["EndTime"], "+%s"], text=True
        ).strip()
    )
    return end_epoch - int(time.time())


def poll(state: dict[str, Any]) -> bool:
    if state["index"] >= len(EXPERIMENTS):
        state.update(status="complete", error="", progress=500)
        atomic_json(STATE_PATH, state)
        update_report(state, 500)
        return True
    experiment = EXPERIMENTS[state["index"]]
    progress = load_progress(run_dir(experiment))
    state.update(
        status="running", current_run_id=experiment["run_id"],
        progress=len(progress), last_poll_epoch=time.time(), error="",
    )
    atomic_json(STATE_PATH, state)
    pressure_issue = runtime_pressure_issue()
    if pressure_issue:
        invalid = run_dir(experiment) / "INVALID.md"
        invalid.write_text(
            "# Invalid experiment\n\n"
            f"Runtime resource safety failure: {pressure_issue}\n"
        )
        cleanup_experiment(experiment)
        raise RuntimeError(f"runtime resource safety failure: {pressure_issue}")
    # Validate the complete service topology before accepting even a finished
    # workload.  A stock Mooncake master can fail while local HiCache/recompute
    # still lets the harness collect 500 callbacks; such a run is not a valid
    # HiCache/Mooncake baseline.
    health_issue = runtime_health_issue(experiment)
    if health_issue:
        invalid = run_dir(experiment) / "INVALID.md"
        invalid.write_text(
            "# Invalid experiment\n\n"
            f"Runtime health failure: {health_issue}\n"
        )
        cleanup_experiment(experiment)
        raise RuntimeError(f"runtime health failure: {health_issue}")
    result = completed_result(experiment)
    update_report(state, len(progress))
    if result is None:
        now = time.time()
        old_progress = int(state.get("last_observed_progress", -1))
        progress_advanced = len(progress) > old_progress
        if progress_advanced:
            state["last_observed_progress"] = len(progress)
            state["last_progress_epoch"] = now
            atomic_json(STATE_PATH, state)
        if (
            not progress_advanced
            and state.get("last_started_epoch")
            and now - float(state["last_started_epoch"]) > 300
            and now - float(state.get("last_progress_epoch", now)) > 900
        ):
            if workload_process_count(experiment["run_id"]) == 0:
                cleanup_experiment(experiment)
                raise RuntimeError(
                    "workload exited before producing a complete artifact"
                )
            last_progress = float(
                state.get("last_progress_epoch", state["last_started_epoch"])
            )
            if now - last_progress > NO_PROGRESS_TIMEOUT_SECONDS:
                cleanup_experiment(experiment)
                raise RuntimeError(
                    f"no new terminal episode for {NO_PROGRESS_TIMEOUT_SECONDS}s"
                )
        log(f"{experiment['run_id']}: {len(progress)}/500")
        return False
    log(f"validated {experiment['run_id']}: 500/500; stopping exact run")
    cleanup_experiment(experiment)
    state["index"] += 1
    state["last_completed_run_id"] = experiment["run_id"]
    state["progress"] = 500
    if state["index"] >= len(EXPERIMENTS):
        state.update(status="complete", error="")
        atomic_json(STATE_PATH, state)
        update_report(state, 500)
        log("all 9 newly scheduled experiments complete")
        return True
    if job_seconds_left() < 5 * 3600:
        raise RuntimeError("less than five hours remain in the sole H100 allocation")
    next_experiment = EXPERIMENTS[state["index"]]
    start_experiment(next_experiment)
    state.update(
        status="running", current_run_id=next_experiment["run_id"],
        progress=0, last_started_epoch=time.time(), last_progress_epoch=time.time(),
        last_observed_progress=0, error="",
    )
    atomic_json(STATE_PATH, state)
    update_report(state, 0)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    load_allocation()
    lock_stream = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another Qwen3.5-9B matrix controller holds the lock")
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text())
    else:
        first_incomplete = 0
        while (
            first_incomplete < len(EXPERIMENTS)
            and completed_result(EXPERIMENTS[first_incomplete]) is not None
        ):
            first_incomplete += 1
        state = {
            "schema_version": 1,
            "index": first_incomplete,
            "status": "ready",
            "error": "",
        }
        atomic_json(STATE_PATH, state)
    if args.start and state["status"] == "ready":
        try:
            if job_seconds_left() < 5 * 3600:
                raise RuntimeError("less than five hours remain in the sole H100 allocation")
            experiment = EXPERIMENTS[state["index"]]
            start_experiment(experiment)
            now = time.time()
            state.update(
                status="running", current_run_id=experiment["run_id"], progress=0,
                last_started_epoch=now, last_progress_epoch=now,
                last_observed_progress=0, error="",
            )
            atomic_json(STATE_PATH, state)
            update_report(state, 0)
        except BaseException as error:
            state.update(
                status="paused", error=f"{type(error).__name__}: {error}",
                last_poll_epoch=time.time(),
            )
            atomic_json(STATE_PATH, state)
            update_report(state, int(state.get("progress", 0)))
            log(f"PAUSED: {state['error']}")
            return 1
    while True:
        try:
            finished = poll(state)
        except BaseException as error:
            state.update(
                status="paused", error=f"{type(error).__name__}: {error}",
                last_poll_epoch=time.time(),
            )
            atomic_json(STATE_PATH, state)
            update_report(state, int(state.get("progress", 0)))
            log(f"PAUSED: {state['error']}")
            return 1
        if finished or args.once:
            return 0
        time.sleep(max(15, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
