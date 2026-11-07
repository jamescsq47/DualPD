#!/usr/bin/env python3
"""Safely finish the two-node Qwen3.8-27B SWE500 concurrency matrix.

The controller owns no model process.  It polls an already allocated Slurm job,
validates completed workload artifacts, updates the matrix Markdown atomically,
stops the exact completed run, and then starts the next immutable experiment.
It deliberately stops on ambiguous state instead of killing processes by name.
"""

from __future__ import annotations

import argparse
import bisect
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
CONTROL_DIR = RUNS / "qwen38-27b-tp2-2p6d-matrix-autopilot"
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
REPORT = ROOT / "runs-host/SWEBENCH_QWEN38_27B_TP2_2P6D_H100_MATRIX.md"

JOB_ID = os.environ.get("DUALPD_MATRIX_JOB_ID", "25255821")
MIN_START_SECONDS = int(os.environ.get("DUALPD_MATRIX_MIN_START_SECONDS", 4 * 3600))
NODE0 = os.environ.get("DUALPD_MATRIX_NODE0", "node2800")
NODE1 = os.environ.get("DUALPD_MATRIX_NODE1", "node2801")
PD_PYTHON = Path("/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12")
BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd_baseline/bin/python3.12"
)
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3.8-27B")
DATA = Path("/orcd/scratch/orcd/003/shangy/dualpd/datasets")
SIF = Path("/orcd/pool/004/shangy/dualpd/containers/swebench-verified-sif")
WORKLOAD = ROOT / "configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer.yaml"
PD_ENGINE = WORKSPACE / "sglang_pd"
# Keep the HiCache/Mooncake run on the clean b1ef6 baseline snapshot.  The
# mutable home worktree is also used for the current Agentic-PD branch and can
# therefore fail the baseline identity gate after a branch refresh.  This
# snapshot is the same baseline code identity used by the successful c128/c192
# runs (notably decode_kvcache_offload_manager.py sha256 7fb5df97...).
BASELINE_ENGINE = RUNS.parent / "sglang-src/pd_baseline_b1ef6"
MOONCAKE_COMPAT = ROOT / "scripts/compat/qwen35_mooncake"
AGENTIC_LAUNCHER = ROOT / "scripts/cluster/qwen35_slurm.py"
COLOCATED_LAUNCHER = ROOT / "scripts/cluster/qwen35_colocated_slurm.py"
NATIVE_LAUNCHER = ROOT / "scripts/cluster/qwen35_native_pd_slurm.py"

START_MARKER = "<!-- AUTOPILOT_RESULTS_START -->"
END_MARKER = "<!-- AUTOPILOT_RESULTS_END -->"
DECODE_COUNTER = "sglang_realtime_tokens_total|mode=decode"


def run_name(concurrency: int, method: str) -> str:
    names = {
        "agentic": "mixed2p6d-agentic",
        "colocated": "colocated8",
        "native": "mixed2p6d-native-oneway",
        "hicache": "mixed2p6d-hicache-mooncake",
    }
    revision = "r10" if concurrency == 256 and method == "hicache" else "r1"
    return (
        f"qwen38-27b-tp2-{names[method]}-swe500-c{concurrency}"
        f"-20261009-auto-{revision}"
    )


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "concurrency": 192,
        "method": "agentic",
        "run_id": "qwen38-27b-tp2-mixed2p6d-agentic-swe500-c192-20261009-r2",
    },
    {
        "concurrency": 192,
        "method": "colocated",
        "run_id": "qwen38-27b-tp2-colocated8-swe500-c192-20261010-auto-r2",
    },
    *[
        {"concurrency": concurrency, "method": method, "run_id": run_name(concurrency, method)}
        for concurrency, methods in (
            (192, ("native", "hicache")),
            (256, ("agentic", "colocated", "native", "hicache")),
        )
        for method in methods
    ],
]

SEED_RUNS = [
    (128, "agentic", "qwen38-27b-tp2-mixed2p6d-agentic-swe500-c128-20261008-r3"),
    (128, "colocated", "qwen38-27b-tp2-colocated8-swe500-c128-20261008-r1"),
    (128, "native", "qwen38-27b-tp2-mixed2p6d-native-oneway-swe500-c128-20261008-r2"),
    (128, "hicache", "qwen38-27b-tp2-mixed2p6d-hicache-mooncake-swe500-c128-20261009-r3"),
]

METHOD_LABEL = {
    "agentic": "Agentic-PD",
    "colocated": "Colocated",
    "native": "原生单向 PD",
    "hicache": "HiCache/Mooncake PD",
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{stamp}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command(argv: list[str | Path], *, timeout: int = 3600, check: bool = True) -> subprocess.CompletedProcess[str]:
    rendered = [str(item) for item in argv]
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
    """Interpolate a monotonic Prometheus counter at a wall-clock boundary."""
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
) -> tuple[float | None, float | None, str | None]:
    """Return Decode tokens and throughput from workload start through T450.

    Prefer per-endpoint monotonic Prometheus counters.  T450-only runs were
    stopped before ``inference.py`` could flush ``engine_metrics.jsonl``; for
    those runs an audited sidecar records the integration of SGLang's
    per-interval Decode throughput logs and its endpoint coverage.
    """
    duration = t450_end - workload_start
    if duration <= 0:
        return None, None, None
    metrics_path = directory / "workload/engine_metrics.jsonl"
    series: dict[str, list[tuple[float, float]]] = {}
    expected_endpoints = 0
    if metrics_path.is_file():
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
                        counter = endpoint.get("metrics", {}).get(DECODE_COUNTER)
                        if counter is None:
                            continue
                        series.setdefault(str(endpoint["endpoint"]), []).append(
                            (float(row["ts"]), float(counter))
                        )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            series = {}
        if series and len(series) >= expected_endpoints:
            total = 0.0
            for points in series.values():
                points.sort()
                if len(points) < 2:
                    return None, None, None
                total += max(
                    0.0,
                    _counter_at_boundary(points, t450_end)
                    - _counter_at_boundary(points, workload_start),
                )
            return total, total / duration, "prometheus_counter"

    sidecar = directory / "workload/t450_decode_metrics.json"
    if not sidecar.is_file():
        return None, None, None
    try:
        payload = json.loads(sidecar.read_text())
        if abs(float(payload["t450_seconds"]) - duration) > 5.0:
            return None, None, None
        tokens = float(payload["decode_tokens"])
        throughput = float(payload["decode_tokens_per_second"])
        source = str(payload["source"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None, None
    return tokens, throughput, source


def completed_result(concurrency: int, method: str, directory: Path) -> dict[str, Any] | None:
    summary_path = directory / "workload/summary.json"
    progress = load_progress(directory)
    if len(progress) != 500 or not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    requests = int(summary.get("requests", -1))
    completed = int(summary.get("completed", -1))
    failed = int(summary.get("failed", -1))
    # A terminal verifier/infrastructure failure is still a finished workload
    # item.  Requiring 500 successful items leaves an otherwise complete run
    # stuck forever when the accepted small failure budget is non-zero.
    if requests != 500 or completed < 0 or failed < 0 or completed + failed != requests:
        return None
    timestamps = sorted(
        float(row["completed_at"]) for row in progress if row.get("completed_at") is not None
    )
    elapsed = float(summary["measurement_elapsed_seconds"])
    t450 = None
    t450_decode_tokens = None
    t450_decode_tps = None
    t450_decode_source = None
    if len(timestamps) >= 450:
        marker = directory / "workload/measurement_start.json"
        if marker.is_file():
            measurement_start = float(
                json.loads(marker.read_text())["measurement_start_wall"]
            )
        else:
            measurement_start = timestamps[-1] - elapsed
        t450 = timestamps[449] - measurement_start
        (
            t450_decode_tokens,
            t450_decode_tps,
            t450_decode_source,
        ) = t450_decode_metrics(directory, measurement_start, timestamps[449])
    environment_errors = sum(bool(row.get("environment_error")) for row in progress)
    return {
        "concurrency": concurrency,
        "method": method,
        "run_id": directory.name,
        "directory": str(directory),
        "completed": completed,
        "failed": failed,
        "solved": sum(float(row.get("reward") or 0) > 0 for row in progress),
        "environment_errors": environment_errors,
        "t450_seconds": t450,
        "t450_decode_tokens": t450_decode_tokens,
        "t450_decode_tps": t450_decode_tps,
        "t450_decode_source": t450_decode_source,
        "t500_seconds": elapsed,
        "prefill_prompt_tps": summary.get("prefill_prompt_tokens_per_second"),
        "prefill_uncached_tps": summary.get("prefill_uncached_tokens_per_second"),
        "scheduler_prefill_tps": summary.get("scheduler_prefill_compute_tokens_per_second"),
        "decode_tps": summary.get("decode_generation_tokens_per_second"),
        "scheduler_decode_tps": summary.get("scheduler_decode_tokens_per_second"),
        "max_queue": summary.get("max_engine_queue_requests"),
        "result_scope": "full-500",
    }


def t450_result(concurrency: int, method: str, directory: Path) -> dict[str, Any] | None:
    """Return a T450-only result even when the final verifier tail is unfinished."""
    progress = load_progress(directory)
    if len(progress) < 450:
        return None
    timestamps = sorted(
        float(row["completed_at"]) for row in progress if row.get("completed_at") is not None
    )
    if len(timestamps) < 450:
        return None
    marker = directory / "workload/measurement_start.json"
    if marker.is_file():
        measurement_start = float(
            json.loads(marker.read_text())["measurement_start_wall"]
        )
        start_source = "measurement_start.json"
    else:
        config = directory / "workload/config.json"
        if not config.is_file():
            return None
        # Runs launched before measurement_start.json was introduced show a
        # stable 3.0--4.2 second gap between config creation and the first
        # sampler timestamp. Four seconds is the conservative calibrated
        # fallback and is explicitly identified in the report record.
        measurement_start = config.stat().st_mtime + 4.0
        start_source = "config_mtime_plus_4s"
    environment_errors = sum(bool(row.get("environment_error")) for row in progress)
    (
        t450_decode_tokens,
        t450_decode_tps,
        t450_decode_source,
    ) = t450_decode_metrics(directory, measurement_start, timestamps[449])
    return {
        "concurrency": concurrency,
        "method": method,
        "run_id": directory.name,
        "directory": str(directory),
        "completed": len(progress),
        "failed": environment_errors,
        "solved": sum(float(row.get("reward") or 0) > 0 for row in progress),
        "environment_errors": environment_errors,
        "t450_seconds": timestamps[449] - measurement_start,
        "t450_decode_tokens": t450_decode_tokens,
        "t450_decode_tps": t450_decode_tps,
        "t450_decode_source": t450_decode_source,
        "t500_seconds": None,
        "prefill_prompt_tps": None,
        "prefill_uncached_tps": None,
        "scheduler_prefill_tps": None,
        "decode_tps": None,
        "scheduler_decode_tps": None,
        "max_queue": None,
        "result_scope": "T450-only",
        "t450_start_source": start_source,
    }


def all_results() -> list[dict[str, Any]]:
    entries = list(SEED_RUNS)
    entries.extend(
        (item["concurrency"], item["method"], item["run_id"]) for item in EXPERIMENTS
    )
    results = []
    seen = set()
    for concurrency, method, name in entries:
        key = (concurrency, method)
        directory = RUNS / name
        result = completed_result(concurrency, method, directory) or t450_result(
            concurrency, method, directory
        )
        if result and key not in seen:
            results.append(result)
            seen.add(key)
    return sorted(results, key=lambda row: (row["concurrency"], ("agentic", "colocated", "native", "hicache").index(row["method"])))


def value(number: Any, digits: int = 2) -> str:
    return "—" if number is None else f"{float(number):,.{digits}f}"


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = all_results()
    current = EXPERIMENTS[state["index"]] if state["index"] < len(EXPERIMENTS) else None
    lines = [
        START_MARKER,
        "## 7. 自动运行进度",
        "",
        f"> 自动控制器最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}。",
        f"> 状态：`{state['status']}`；已获得可用 T450 结果 **{len(results)}/12**。",
    ]
    if current:
        lines.append(
            f"> 当前：c{current['concurrency']} {METHOD_LABEL[current['method']]}，"
            f"episode progress **{current_progress}/500**，run `{current['run_id']}`。"
        )
    elif state["status"] == "complete":
        lines.append("> 12 组实验均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend([
        "",
        "Decode 参考吞吐统一采用正式测量起点到第 450 个 terminal completion",
        "（0–T450）的全部 D/Serving endpoint（PD 为 6 个、Colocated 为 8 个）",
        "总 Decode token 增量除以 T450。",
        "它不是完整 500 题 summary 的平均吞吐。",
        "",
        "| 并发 | 方法 | 完成/失败 | Resolved | T450 (s) | 0–T450 Decode token | 0–T450 Decode tok/s | T500 (s) | 口径/异常 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for result in results:
        caveat = []
        if result.get("result_scope") == "T450-only":
            caveat.append("T450-only")
            if result.get("t450_start_source") != "measurement_start.json":
                caveat.append(str(result.get("t450_start_source")))
        if result["environment_errors"]:
            caveat.append(f"env={result['environment_errors']}")
        if result["failed"]:
            caveat.append(f"failed={result['failed']}")
        decode_source = result.get("t450_decode_source")
        if decode_source == "service_log_integral_6_of_6":
            caveat.append("Decode=6/6日志积分")
        elif decode_source == "service_log_integral_3_of_6_scaled_2x":
            caveat.append("Decode=半节点×2估算")
        lines.append(
            "| {concurrency} | {method} | {completed}/{failed} | {solved}/500 | {t450} | {decode_tokens} | {decode_tps} | {t500} | {caveat} |".format(
                concurrency=result["concurrency"],
                method=METHOD_LABEL[result["method"]],
                completed=result["completed"],
                failed=result["failed"],
                solved=result["solved"],
                t450=value(result["t450_seconds"]),
                decode_tokens=value(result.get("t450_decode_tokens"), 0),
                decode_tps=value(result.get("t450_decode_tps")),
                t500=value(result["t500_seconds"]),
                caveat=", ".join(caveat) or "—",
            )
        )
    lines.extend(
        [
            "",
            "> **c256 HiCache/Mooncake 不能仅按 T450 判为更快：**其 0–T450 只生成",
            "> 3.29M Decode token，远低于同档 Agentic-PD 的 10.12M 和 Colocated 的",
            "> 9.99M；Resolved 也只有 66/500。其 1,363.64 tok/s 明显低于两者，",
            "> 因而 2,410.56 秒主要来自轨迹异常缩短，不是 serving 吞吐优势。",
            "",
            "原始目录：",
            "",
        ]
    )
    for result in results:
        lines.append(
            f"- c{result['concurrency']} {METHOD_LABEL[result['method']]}: `{result['directory']}`"
        )
    lines.extend([END_MARKER, ""])
    block = "\n".join(lines)
    text = REPORT.read_text()
    if START_MARKER in text and END_MARKER in text:
        before, rest = text.split(START_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, REPORT)


def base_prepare(experiment: dict[str, Any]) -> list[str]:
    directory = run_dir(experiment)
    return [
        "--job-id", JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", directory,
        "--python", PD_PYTHON,
        "--model", MODEL,
        "--data-root", DATA,
        "--swe-sif-dir", SIF,
        "--workload-config", WORKLOAD,
        "--requests", "500",
        "--concurrency", str(experiment["concurrency"]),
        "--tp-size", "2",
    ]


def prepare_and_start(experiment: dict[str, Any]) -> None:
    directory = run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse existing next-run directory: {directory}")
    method = experiment["method"]
    if method == "agentic":
        common = [
            PD_PYTHON, AGENTIC_LAUNCHER,
            "--job-id", JOB_ID,
            "--prefill-node", NODE0,
            "--decode-node", NODE1,
            "--python", PD_PYTHON,
            "--sglang-root", PD_ENGINE,
            "--model", MODEL,
            "--data-root", DATA,
            "--swe-sif-dir", SIF,
            "--workload-config", WORKLOAD,
            "--run-dir", directory,
            "--run-id", experiment["run_id"],
            "--requests", "500",
            "--concurrency", str(experiment["concurrency"]),
            "--tp-size", "2",
            "--mixed-one-prefill-per-node",
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
        command([*common, "--detach", "start-decode"], timeout=600)
        command([*common, "wait-ready"], timeout=2400)
        command([*common, "--detach", "start-router"], timeout=600)
        command([*common, "--detach", "run-workload"], timeout=600)
        return
    if method == "colocated":
        prepare = [
            PD_PYTHON, COLOCATED_LAUNCHER, "prepare",
            *base_prepare(experiment),
            "--workload-python", PD_PYTHON,
            "--groups-per-node", "4",
            "--cpus-per-node", "224",
        ]
        command(prepare)
        command([PD_PYTHON, COLOCATED_LAUNCHER, "start", "--job-id", JOB_ID, "--run-dir", directory], timeout=3600)
        return
    prepare = [
        BASELINE_PYTHON, NATIVE_LAUNCHER, "prepare",
        "--job-id", JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", directory,
        "--python", BASELINE_PYTHON,
        "--workload-python", PD_PYTHON,
        "--sglang-root", BASELINE_ENGINE,
        "--model", MODEL,
        "--data-root", DATA,
        "--swe-sif-dir", SIF,
        "--workload-config", WORKLOAD,
        "--requests", "500",
        "--concurrency", str(experiment["concurrency"]),
        "--tp-size", "2",
        "--mamba-full-memory-ratio", "0.9",
        "--prefill-groups-per-node", "1",
        "--decode-groups-per-node", "3",
        "--mixed-one-prefill-per-node",
        "--launch-transport", "ssh",
    ]
    if method == "hicache":
        prepare.extend([
            "--native-mooncake",
            "--mooncake-compat", MOONCAKE_COMPAT,
            "--prefill-hicache-size-gb", "32",
            "--decode-hicache-size-gb", "32",
            # c256 can concurrently retain more recurrent states than the
            # ~17.28 GB device pool.  Keep enough host capacity to avoid the
            # stock tombstone-eviction path while preserving ratio=0.9.
            "--mamba-hicache-size-gb", "32",
            "--mooncake-store-size-gb", "512",
            "--mooncake-prefetch-policy", "wait_complete",
            "--workload-job-id", JOB_ID,
            "--workload-node", NODE1,
            "--storage-job-id", JOB_ID,
            "--storage-node", NODE0,
        ])
    command(prepare)
    command([BASELINE_PYTHON, NATIVE_LAUNCHER, "start", "--job-id", JOB_ID, "--run-dir", directory], timeout=3600)


def stop_experiment(experiment: dict[str, Any]) -> None:
    def best_effort_stop(argv: list[str | Path], *, timeout: int) -> None:
        """Request an exact, run-scoped stop without trusting RPC completion.

        A component may finish between result validation and the stop RPC, or
        an SSH/Slurm control client may outlive the remote supervisor response.
        The authoritative safety check is ``wait_clean`` below, which requires
        both zero GPU compute processes and zero processes containing this
        immutable run id before another experiment can start.
        """
        try:
            command(argv, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            log(
                "stop request timed out; continuing to authoritative "
                "run-scoped cleanup verification: "
                + " ".join(str(item) for item in argv)
            )

    directory = run_dir(experiment)
    method = experiment["method"]
    if method == "agentic":
        common = [
            PD_PYTHON, AGENTIC_LAUNCHER,
            "--job-id", JOB_ID,
            "--prefill-node", NODE0,
            "--decode-node", NODE1,
            "--python", PD_PYTHON,
            "--run-dir", directory,
            "--launch-transport", "ssh",
        ]
        for component, role in (
            ("workload", None), ("router", None), ("worker", "decode"),
            ("worker", "prefill"), ("control", None),
        ):
            args = [*common, "stop", "--component", component]
            if role:
                args.extend(["--worker-role", role])
            best_effort_stop(args, timeout=180)
    elif method == "colocated":
        best_effort_stop(
            [PD_PYTHON, COLOCATED_LAUNCHER, "stop", "--job-id", JOB_ID, "--run-dir", directory],
            timeout=300,
        )
    else:
        best_effort_stop(
            [BASELINE_PYTHON, NATIVE_LAUNCHER, "stop", "--job-id", JOB_ID, "--run-dir", directory],
            timeout=300,
        )


def gpu_process_count() -> int:
    total = 0
    for node in (NODE0, NODE1):
        result = command([
            "srun", f"--jobid={JOB_ID}", "--overlap", "--nodes=1", "--ntasks=1",
            f"--nodelist={node}", "--cpus-per-task=1", "bash", "-lc",
            "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
            "2>/dev/null | sed '/^$/d' | wc -l",
        ], timeout=30)
        total += int(result.stdout.strip().splitlines()[-1])
    return total


def scoped_process_count(run_id: str) -> int:
    """Count surviving run-scoped processes without matching the probe itself."""
    probe = (
        "import os,sys\n"
        "needle=sys.argv[1].encode()\n"
        "marker=b'dualpd-scoped-probe-v2'\n"
        "excluded={os.getpid(),os.getppid()}\n"
        "def hit(name):\n"
        "  if not name.isdigit() or int(name) in excluded: return False\n"
        "  try:\n"
        "    cmdline=open('/proc/'+name+'/cmdline','rb').read()\n"
        "    return needle in cmdline and marker not in cmdline\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError): return False\n"
        "print(sum(hit(name) for name in os.listdir('/proc')))"
    )
    total = 0
    for node in (NODE0, NODE1):
        result = command(
            [
                "srun", f"--jobid={JOB_ID}", "--overlap", "--nodes=1", "--ntasks=1",
                f"--nodelist={node}", "--cpus-per-task=1",
                "python3", "-c", probe, run_id,
            ],
            timeout=30,
        )
        total += int(result.stdout.strip().splitlines()[-1])
    return total


def terminate_scoped_non_gpu_processes(run_id: str) -> None:
    """Stop detached workload/verifier children after every GPU is idle.

    SWE verifier containers create their own sessions, so stopping the workload
    parent at the T450 boundary does not always reap the few tests already in
    flight.  Match only the immutable run id and exclude the cleanup probe and
    all of its ancestors.  This is deliberately called only after both nodes
    report zero GPU compute processes.
    """
    cleanup = (
        "import os,signal,sys,time\n"
        "needle=sys.argv[1].encode()\n"
        "excluded=set(); pid=os.getpid()\n"
        "while pid and pid not in excluded:\n"
        "  excluded.add(pid)\n"
        "  try: pid=int(open('/proc/'+str(pid)+'/stat').read().split()[3])\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError): break\n"
        "pgids=set()\n"
        "for name in os.listdir('/proc'):\n"
        "  if not name.isdigit() or int(name) in excluded: continue\n"
        "  try:\n"
        "    cmdline=open('/proc/'+name+'/cmdline','rb').read()\n"
        "    if needle in cmdline:\n"
        "      pgid=os.getpgid(int(name))\n"
        "      if pgid > 1 and pgid not in excluded: pgids.add(pgid)\n"
        "  except (FileNotFoundError,PermissionError,ProcessLookupError): pass\n"
        "for pgid in pgids:\n"
        "  try: os.killpg(pgid,signal.SIGTERM)\n"
        "  except ProcessLookupError: pass\n"
        "time.sleep(5)\n"
        "for pgid in pgids:\n"
        "  try: os.killpg(pgid,signal.SIGKILL)\n"
        "  except ProcessLookupError: pass\n"
        "print(len(pgids))\n"
    )
    for node in (NODE0, NODE1):
        command(
            [
                "srun", f"--jobid={JOB_ID}", "--overlap", "--nodes=1",
                "--ntasks=1", f"--nodelist={node}", "--cpus-per-task=1",
                "python3", "-c", cleanup, run_id,
            ],
            timeout=30,
            check=False,
        )


def wait_clean(run_id: str) -> None:
    deadline = time.monotonic() + 300
    cleaned_detached = False
    while time.monotonic() < deadline:
        try:
            gpu_count = gpu_process_count()
            scoped_count = scoped_process_count(run_id)
        except subprocess.TimeoutExpired as error:
            log(f"transient cleanup probe timeout: {error}")
            time.sleep(15)
            continue
        if gpu_count == 0 and scoped_count == 0:
            time.sleep(10)
            if gpu_process_count() == 0 and scoped_process_count(run_id) == 0:
                return
        if gpu_count == 0 and scoped_count > 0 and not cleaned_detached:
            log(
                "GPU compute is idle; terminating detached run-scoped "
                f"workload/verifier processes={scoped_count}"
            )
            terminate_scoped_non_gpu_processes(run_id)
            cleaned_detached = True
        log(
            "waiting for exact stop; "
            f"GPU compute processes={gpu_count}, run-scoped processes={scoped_count}"
        )
        time.sleep(15)
    raise RuntimeError(
        "completed run did not release every GPU and run-scoped process within 300 seconds"
    )


def job_state() -> tuple[str, int]:
    result = command(
        ["scontrol", "show", "job", "-o", JOB_ID], timeout=30
    ).stdout.strip()
    fields = dict(token.split("=", 1) for token in result.split() if "=" in token)
    state = fields.get("JobState", "UNKNOWN")
    end = fields.get("EndTime", "Unknown")
    if end in {"Unknown", "N/A", "None"}:
        return state, 0
    end_epoch = int(
        subprocess.check_output(["date", "-u", "-d", end, "+%s"], text=True).strip()
    )
    return state, end_epoch - int(time.time())


def poll(state: dict[str, Any]) -> bool:
    if state["index"] >= len(EXPERIMENTS):
        state["status"] = "complete"
        state["error"] = ""
        update_report(state, 500)
        atomic_json(STATE_PATH, state)
        return True
    experiment = EXPERIMENTS[state["index"]]
    directory = run_dir(experiment)
    # A previous allocation may have completed the preceding experiment but
    # refused to launch this one because less than five hours remained.  A
    # successor allocation resumes from that durable state and owns creation
    # of the still-absent immutable run directory.
    if not directory.exists():
        slurm_state, seconds_left = job_state()
        if slurm_state != "RUNNING":
            raise RuntimeError(f"Slurm job {JOB_ID} is {slurm_state}")
        if seconds_left < MIN_START_SECONDS:
            raise RuntimeError(
                f"Slurm job has only {seconds_left / 3600:.2f}h left; "
                "refusing to start a long run"
            )
        log(f"starting pending experiment {experiment['run_id']}")
        prepare_and_start(experiment)
        state.update({
            "status": "running",
            "current_run_id": experiment["run_id"],
            "progress": 0,
            "last_progress_value": 0,
            "last_progress_epoch": time.time(),
            "last_started_epoch": time.time(),
            "error": "",
        })
        atomic_json(STATE_PATH, state)
        update_report(state, 0)
        return False
    progress = load_progress(directory)
    result = completed_result(experiment["concurrency"], experiment["method"], directory)
    t450_only = t450_result(
        experiment["concurrency"], experiment["method"], directory
    )
    now = time.time()
    previous_progress = int(state.get("last_progress_value", -1))
    if len(progress) != previous_progress:
        state["last_progress_value"] = len(progress)
        state["last_progress_epoch"] = now
    elif (
        len(progress) < 500
        and t450_only is None
        and state.get("last_progress_epoch")
        and now - float(state["last_progress_epoch"]) > 3 * 3600
    ):
        raise RuntimeError(
            f"no episode progress for more than three hours at {len(progress)}/500"
        )
    state.update({
        "status": "running",
        "current_run_id": experiment["run_id"],
        "progress": len(progress),
        "last_poll_epoch": now,
        "error": "",
    })
    update_report(state, len(progress))
    atomic_json(STATE_PATH, state)
    if result is None and t450_only is None:
        log(
            f"{experiment['run_id']}: {len(progress)}/500; waiting for complete progress and summary"
        )
        return False
    result = result or t450_only
    assert result is not None
    if result["failed"] > 2 or result["environment_errors"] > 2:
        raise RuntimeError(
            f"completed artifact has failed={result['failed']} "
            f"environment_errors={result['environment_errors']}"
        )
    log(
        f"validated {experiment['run_id']}: T450={result['t450_seconds']:.2f}s "
        f"at {len(progress)}/500; updating report and stopping"
    )
    stop_experiment(experiment)
    wait_clean(experiment["run_id"])
    state["index"] += 1
    state["last_completed_run_id"] = experiment["run_id"]
    state["progress"] = 500
    if state["index"] >= len(EXPERIMENTS):
        state["status"] = "complete"
        atomic_json(STATE_PATH, state)
        update_report(state, 500)
        log("all matrix experiments complete")
        return True
    slurm_state, seconds_left = job_state()
    if slurm_state != "RUNNING":
        raise RuntimeError(f"Slurm job {JOB_ID} is {slurm_state}")
    if seconds_left < MIN_START_SECONDS:
        raise RuntimeError(
            f"Slurm job has only {seconds_left / 3600:.2f}h left; refusing to start a long run"
        )
    next_experiment = EXPERIMENTS[state["index"]]
    log(f"starting next experiment {next_experiment['run_id']}")
    prepare_and_start(next_experiment)
    state.update({
        "status": "running",
        "current_run_id": next_experiment["run_id"],
        "progress": 0,
        "last_started_epoch": time.time(),
    })
    atomic_json(STATE_PATH, state)
    update_report(state, 0)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another matrix autopilot already holds the controller lock")
    state = (
        json.loads(STATE_PATH.read_text())
        if STATE_PATH.is_file()
        else {"schema_version": 1, "index": 0, "status": "running", "error": ""}
    )
    while True:
        try:
            finished = poll(state)
        except BaseException as error:
            state.update({
                "status": "paused",
                "error": f"{type(error).__name__}: {error}",
                "last_poll_epoch": time.time(),
            })
            atomic_json(STATE_PATH, state)
            current_progress = int(state.get("progress", 0))
            update_report(state, current_progress)
            log(f"PAUSED: {state['error']}")
            return 1
        if finished or args.once:
            return 0
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
