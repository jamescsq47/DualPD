#!/usr/bin/env python3
"""Summarize one closed-loop agentic steady-state run.

Only requests completed inside the recorded measurement window contribute to
request statistics. Engine counters and gauges are interpolated at the exact
window boundaries so different methods use the same 300+1200 second basis.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def at(points: list[tuple[float, float]], timestamp: float) -> float:
    times = [point[0] for point in points]
    index = bisect.bisect_right(times, timestamp) - 1
    if index < 0:
        return points[0][1]
    if index >= len(points) - 1:
        return points[-1][1]
    left_t, left_v = points[index]
    right_t, right_v = points[index + 1]
    return left_v + (right_v - left_v) * (timestamp - left_t) / (right_t - left_t)


def interval_value(
    rows: list[dict], key: str, start: float, end: float, *, counter: bool
) -> float | None:
    points = sorted(
        (float(row["ts"]), float(row["metrics"][key]))
        for row in rows
        if row.get("metrics")
        and key in row["metrics"]
        and math.isfinite(float(row["metrics"][key]))
    )
    if not points:
        return None
    if counter:
        if any(right[1] < left[1] for left, right in zip(points, points[1:])):
            raise ValueError(f"counter reset in {key}")
        return (at(points, end) - at(points, start)) / (end - start)
    clipped = [(start, at(points, start))]
    clipped.extend(point for point in points if start < point[0] < end)
    clipped.append((end, at(points, end)))
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2
        for left, right in zip(clipped, clipped[1:])
    )
    return area / (end - start)


def engine_summary(rows: list[dict], role: str, start: float, end: float) -> dict:
    selected = [row for row in rows if row.get("role") == role and row.get("metrics")]
    if not selected:
        raise ValueError(f"no engine metrics for role={role}")
    engine_count = max(int(row.get("engine_count", 1)) for row in selected)
    mode = "prefill_compute" if role == "prefill" else "decode"
    category = "forward_extend" if role == "prefill" else "forward_decode"
    tps = interval_value(
        selected, f"sglang_realtime_tokens_total|mode={mode}", start, end, counter=True
    )
    forward_total = interval_value(
        selected,
        f"sglang_gpu_execution_seconds_total|category={category}",
        start,
        end,
        counter=True,
    )
    result = {
        "engine_count": engine_count,
        "token_s_total": tps,
        "token_s_per_gpu": tps / engine_count if tps is not None else None,
        "forward_fraction_total": forward_total,
        "forward_fraction_per_gpu": (
            forward_total / engine_count if forward_total is not None else None
        ),
    }
    for label, key in (
        ("token_usage", "sglang_token_usage"),
        ("full_token_usage", "sglang_full_token_usage"),
        ("running", "sglang_num_running_reqs"),
        ("queue", "sglang_num_queue_reqs"),
        ("transfer_queue", "sglang_num_decode_transfer_queue_reqs"),
    ):
        mean_total = interval_value(selected, key, start, end, counter=False)
        window_values = [
            float(row["metrics"][key])
            for row in selected
            if start <= float(row["ts"]) <= end and key in row["metrics"]
        ]
        result[f"{label}_mean_per_gpu"] = (
            mean_total / engine_count if mean_total is not None else None
        )
        result[f"{label}_peak_per_gpu"] = (
            max(window_values) / engine_count if window_values else None
        )
    return result


def summarize(run_dir: Path) -> dict:
    boundary = json.loads((run_dir / "closed_loop_boundaries.json").read_text())
    start = float(boundary["measurement_start_wall"])
    end = float(boundary["measurement_end_wall"])
    duration = end - start
    requests = read_jsonl(run_dir / "requests.jsonl")
    completed = [row for row in requests if start <= float(row["finished_ts"]) <= end]
    expected = (
        int(boundary["state_at_measurement_end"]["successes"])
        - int(boundary["state_at_measurement_start"]["successes"])
    )
    if len(completed) != expected:
        raise ValueError(f"measurement request mismatch: rows={len(completed)} state={expected}")

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in completed]

    def distribution(key: str) -> dict:
        data = values(key)
        return {
            "mean": statistics.fmean(data),
            "median": quantile(data, 0.5),
            "p90": quantile(data, 0.9),
            "max": max(data),
        }

    stop_reasons = Counter(row.get("metadata", {}).get("stop_reason") for row in completed)
    statuses = Counter(row.get("status") for row in completed)
    engine_rows = read_jsonl(run_dir / "engine_metrics.jsonl")
    return {
        "run_dir": str(run_dir.resolve()),
        "status": boundary["status"],
        "warmup_seconds": float(boundary["warmup_seconds"]),
        "measurement_seconds": duration,
        "concurrency": int(boundary["state_at_measurement_start"]["active"]),
        "completed_agents": len(completed),
        "agent_s": len(completed) / duration,
        "failed_agents": (
            int(boundary["state_at_measurement_end"]["failures"])
            - int(boundary["state_at_measurement_start"]["failures"])
        ),
        "statuses": dict(statuses),
        "stop_reasons": {str(key): value for key, value in stop_reasons.items()},
        "one_turn_fraction": sum(row["generation_turns"] == 1 for row in completed)
        / len(completed),
        "turns": distribution("generation_turns"),
        "model_prompt_tokens": distribution("model_prompt_tokens"),
        "model_completion_tokens": distribution("model_completion_tokens"),
        "tool_time_seconds": distribution("tool_time_seconds"),
        "search_calls": sum(int(row.get("search_calls", 0)) for row in completed),
        "code_calls": sum(int(row.get("code_calls", 0)) for row in completed),
        "shell_calls": sum(
            int(row.get("metadata", {}).get("shell_call_count", 0) or 0)
            for row in completed
        ),
        "prefill": engine_summary(engine_rows, "prefill", start, end),
        "decode": engine_summary(engine_rows, "decode", start, end),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    output = args.run_dir / "agentic_steady_summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
