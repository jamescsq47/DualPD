#!/usr/bin/env python3
"""Run the Qwen3.5-35B-A3B TP2 single-node 1P:3D SWE500 matrix."""

from __future__ import annotations

import bisect
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from . import qwen38_27b_single_node_1p3d_matrix_autopilot as base
else:  # Direct script execution.
    import qwen38_27b_single_node_1p3d_matrix_autopilot as base


CONTROL_DIR = (
    base.RUNS
    / "qwen35-35b-a3b-tp2-single-node-1p3d-c64c128c192-growth128k-20261014"
)
REPORT = (
    base.ROOT
    / "runs-host/"
    "SWEBENCH_QWEN35_35B_A3B_TP2_SINGLE_NODE_1P3D_H100.md"
)
MODEL = base.RUNS.parent / "models/Qwen3.5-35B-A3B"
JOB_ID = "25734937"
NODE = "node2800"
CONCURRENCIES = (64, 128, 192)
METHODS = ("agentic", "colocated", "native", "hicache")
TP_SIZE = 2
_RESOURCE_CACHE: dict[str, dict[str, float | None]] = {}


def _interval_value(
    rows: list[tuple[float, dict[str, float]]],
    keys: tuple[str, ...],
    begin: float,
    end: float,
    *,
    counter: bool,
) -> float | None:
    """Interpolate one counter/gauge over the T450 wall-clock window."""
    points: list[tuple[float, float]] = []
    for timestamp, metrics in rows:
        value = next((metrics[key] for key in keys if key in metrics), None)
        if value is None or not math.isfinite(float(value)):
            continue
        points.append((float(timestamp), float(value)))
    if not points or end <= begin:
        return None
    points.sort()
    if counter and any(y < x for (_, x), (_, y) in zip(points, points[1:])):
        return None
    timestamps = [point[0] for point in points]

    def at(boundary: float) -> float:
        index = bisect.bisect_right(timestamps, boundary) - 1
        if index < 0:
            return points[0][1]
        if index >= len(points) - 1:
            return points[-1][1]
        left, right = points[index], points[index + 1]
        if right[0] == left[0]:
            return right[1]
        fraction = (boundary - left[0]) / (right[0] - left[0])
        return left[1] + fraction * (right[1] - left[1])

    if counter:
        return max(0.0, at(end) - at(begin)) / (end - begin)
    clipped = [(begin, at(begin))]
    clipped.extend(point for point in points if begin < point[0] < end)
    clipped.append((end, at(end)))
    return sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2
        for left, right in zip(clipped, clipped[1:])
    ) / (end - begin)


def _resource_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    """Derive T450 P/D utilization from immutable per-endpoint scrapes."""
    run_id = str(result["run_id"])
    if run_id in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[run_id]
    directory = Path(result["directory"])
    progress = base.core.load_progress(directory)
    terminal_times = sorted(
        float(row["completed_at"])
        for row in progress
        if row.get("completed_at") is not None
    )
    metrics_path = directory / "workload/engine_metrics.jsonl"
    if len(terminal_times) < 450 or not metrics_path.is_file():
        return {}
    end = terminal_times[449]
    begin = end - float(result["t450_seconds"])
    series: dict[tuple[str, str], list[tuple[float, dict[str, float]]]] = (
        defaultdict(list)
    )
    try:
        with metrics_path.open(errors="replace") as stream:
            for line in stream:
                row = json.loads(line)
                timestamp = float(row["ts"])
                # Keep one scrape on either side for boundary interpolation.
                if timestamp < begin - 30 or timestamp > end + 30:
                    continue
                role = str(row.get("role"))
                for endpoint in row.get("endpoint_metrics", []):
                    series[(role, str(endpoint["endpoint"]))].append(
                        (timestamp, endpoint.get("metrics", {}))
                    )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {}

    prefill = [
        rows for (role, _), rows in series.items() if role == "prefill"
    ]
    decode_role = "prefill" if result["method"] == "colocated" else "decode"
    decode = [
        rows for (role, _), rows in series.items() if role == decode_role
    ]

    def per_engine_mean(
        groups: list[list[tuple[float, dict[str, float]]]],
        keys: tuple[str, ...],
        *,
        counter: bool,
    ) -> float | None:
        values = [
            _interval_value(rows, keys, begin, end, counter=counter)
            for rows in groups
        ]
        if not values or any(value is None for value in values):
            return None
        return sum(float(value) for value in values) / len(values)

    prefill_compute = [
        _interval_value(
            rows,
            ("sglang_realtime_tokens_total|mode=prefill_compute",),
            begin,
            end,
            counter=True,
        )
        for rows in prefill
    ]
    prefill_cache = [
        _interval_value(
            rows,
            ("sglang_realtime_tokens_total|mode=prefill_cache",),
            begin,
            end,
            counter=True,
        )
        for rows in prefill
    ]
    compute_rate = (
        sum(float(value) for value in prefill_compute if value is not None)
        if prefill_compute and all(value is not None for value in prefill_compute)
        else None
    )
    # A missing prefill_cache series means that this engine recorded no cache
    # hits (not missing scrapes); prefill_compute remains the liveness guard.
    cache_rate = (
        sum(float(value or 0.0) for value in prefill_cache)
        if compute_rate is not None
        else None
    )
    reuse = (
        cache_rate / (cache_rate + compute_rate)
        if cache_rate is not None and compute_rate is not None
        and cache_rate + compute_rate > 0
        else None
    )
    metrics = {
        "p_forward": (
            lambda value: None if value is None else value / TP_SIZE
        )(
            per_engine_mean(
                prefill,
                (
                    "sglang_gpu_execution_seconds_total|category=forward_extend",
                    "sglang_forward_execution_seconds_total|category=extend",
                ),
                counter=True,
            )
        ),
        "p_kv": per_engine_mean(
            prefill,
            ("sglang_full_token_usage", "sglang_token_usage"),
            counter=False,
        ),
        "d_forward": (
            lambda value: None if value is None else value / TP_SIZE
        )(
            per_engine_mean(
                decode,
                (
                    "sglang_gpu_execution_seconds_total|category=forward_decode",
                    "sglang_forward_execution_seconds_total|category=decode",
                ),
                counter=True,
            )
        ),
        "d_kv": per_engine_mean(
            decode,
            ("sglang_full_token_usage", "sglang_token_usage"),
            counter=False,
        ),
        "prefix_reuse": reuse,
    }
    _RESOURCE_CACHE[run_id] = metrics
    return metrics


def resource_report_lines(results: list[dict[str, Any]]) -> list[str]:
    """Append the requested P/D Forward, KV, and reuse table."""
    by_run = {str(result["run_id"]): result for result in results}

    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.2f}%"

    lines = [
        "",
        "### T450窗口 P/D Forward、KV与复用",
        "",
        "| 并发 | 方法 | P Forward/卡 | P Attention KV/引擎 | D Forward/卡 | D/共享 Attention KV/引擎 | Prefix复用率 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        result = by_run.get(experiment["run_id"])
        if result is None:
            lines.append(
                f"| {experiment['concurrency']} | "
                f"{base.METHOD_LABEL[experiment['method']]} | — | — | — | — | — |"
            )
            continue
        metrics = _resource_metrics(result)
        p_kv = None if experiment["method"] == "colocated" else metrics.get("p_kv")
        lines.append(
            f"| {experiment['concurrency']} | "
            f"{base.METHOD_LABEL[experiment['method']]} | "
            f"{percent(metrics.get('p_forward'))} | {percent(p_kv)} | "
            f"{percent(metrics.get('d_forward'))} | "
            f"{percent(metrics.get('d_kv'))} | "
            f"{percent(metrics.get('prefix_reuse'))} |"
        )
    lines.extend([
        "",
        "Forward为从workload起点到T450的物理GPU平均device-timer占比；",
        "Attention KV为同窗口内`sglang_full_token_usage`的时间加权均值，按逻辑TP=2引擎平均。",
        "Colocated共享P/D KV pool，因此P KV留空，共享值记在D/共享列。",
        "Prefix复用率为P侧GPU token计数器`prefill_cache / "
        "(prefill_cache + prefill_compute)`；这是实际prefix命中口径，不冒充仅统计"
        "page-aligned parent-KV的专用回传成功率。",
    ])
    return lines


def run_name(concurrency: int, method: str) -> str:
    names = {
        "agentic": "agentic",
        "colocated": "colocated4",
        "native": "native-oneway",
        "hicache": "hicache-mooncake",
    }
    if method in {"colocated", "native"}:
        revision = "correctness-r1"
    elif method == "hicache":
        # r1 failed before loading the model because this older baseline uses
        # --mamba-scheduler-strategy rather than the radix-cache spelling.
        revision = "r2"
    else:
        revision = "r1"
    return (
        f"q35-35b-a3b-h100-1n-1p3d-{names[method]}-"
        f"swe500-c{concurrency}-20261014-{revision}"
    )


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_name(concurrency, method),
    }
    for concurrency in CONCURRENCIES
    for method in METHODS
]


def rebuild_allocation(reason: str) -> None:
    """Replace a dirty allocation, but remain strictly on node2800."""
    old_job_id = base.JOB_ID
    base.core.log(
        f"hard gate failed ({reason}); cancelling node2800 allocation {old_job_id}"
    )
    base.core.command(["scancel", old_job_id], timeout=60, check=False)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        queued = base.core.command(
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

    submitted = base.core.command(
        [
            "sbatch",
            "--parsable",
            "--job-name=qwen35-35b-a3b-swe-h100-1n",
            "--partition=pi_songhan",
            "--account=mit_general",
            "--nodes=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=224",
            "--gres=gpu:h100:8",
            "--mem=1800G",
            "--exclusive",
            "--time=7-00:00:00",
            "--nodelist=node2800",
            f"--output={base.WORKSPACE}/slurm-qwen35-35b-a3b-%j.out",
            '--wrap=trap "exit 0" TERM INT; while true; do sleep 300; done',
        ],
        timeout=60,
    )
    new_job_id = submitted.stdout.strip().splitlines()[-1].split(";", 1)[0]
    base.JOB_ID = new_job_id
    base.core.JOB_ID = new_job_id
    base.NODE = ""
    base.core.NODE = ""
    base.core.save_allocation("pending")
    new_node = base.core.wait_for_allocation(new_job_id)
    if new_node != "node2800":
        raise RuntimeError(f"replacement landed on forbidden node {new_node}")
    base.NODE = new_node
    base.core.NODE = new_node
    base.core.save_allocation("running")
    base.core.command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            new_node,
            f"grep -q job_{new_job_id} /proc/self/cgroup",
        ],
        timeout=30,
    )
    base.core.assert_clean_for_start()
    base.core.log(
        f"replacement allocation {new_job_id} on node2800 passed the hard zero gate"
    )


def configure() -> None:
    base.CONTROL_DIR = CONTROL_DIR
    base.STATE_PATH = CONTROL_DIR / "state.json"
    base.LOCK_PATH = CONTROL_DIR / "controller.lock"
    base.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
    base.BASELINE_VALIDATION_PATH = (
        CONTROL_DIR / "baseline-correctness-validation.json"
    )
    base.REPORT = REPORT
    base.MODEL = MODEL
    base.JOB_ID = JOB_ID
    base.NODE = NODE
    base.CONCURRENCIES = CONCURRENCIES
    base.METHODS = METHODS
    base.RESULT_EXPERIMENTS = EXPERIMENTS
    base.EXPERIMENTS = EXPERIMENTS
    base.HICACHE_MAMBA_CACHE_STRATEGY_FLAG = "scheduler"
    base.REPORT_EXTRA_SECTIONS = resource_report_lines
    base.rebuild_allocation = rebuild_allocation


def main() -> int:
    configure()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
