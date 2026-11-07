#!/usr/bin/env python3
"""Summarize durable TP8 MiniMax Agentic-PD metrics over T50--T450."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from summarize_native_swe_colocated import interval_value


def read_jsonl(path: Path):
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A live writer can leave one incomplete trailing record.
                continue


def key(samples, *candidates: str) -> str:
    for candidate in candidates:
        if any(candidate in metrics for _, metrics in samples):
            return candidate
    raise ValueError(f"missing metric: {candidates}")


def gauge_stats(samples, metric: str, begin: float, end: float) -> dict[str, float | None]:
    values = [metrics[metric] for ts, metrics in samples if begin <= ts <= end and metric in metrics]
    if not values:
        return {"mean": None, "p95": None, "peak": None}
    ordered = sorted(values)
    return {
        "mean": interval_value(samples, metric, begin, end, False),
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "peak": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()

    completed = sorted(
        float(row["completed_at"])
        for row in read_jsonl(args.run / "episode_progress.jsonl")
        if row.get("completed_at") is not None
    )
    if len(completed) < 450:
        raise ValueError(f"only {len(completed)} terminal episodes")
    begin, end = completed[49], completed[449]

    series: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    malformed_or_error_rows = 0
    for row in read_jsonl(args.run / "workload/engine_metrics.jsonl"):
        role = row.get("role")
        if role not in {"prefill", "decode"}:
            continue
        endpoints = row.get("endpoint_metrics", [])
        if len(endpoints) != 1 or row.get("errors"):
            malformed_or_error_rows += 1
        for endpoint in endpoints:
            series[role].append((float(row["ts"]), endpoint.get("metrics", {})))
    for samples in series.values():
        samples.sort()

    output: dict[str, Any] = {
        "begin": begin,
        "end": end,
        "seconds": end - begin,
        "metrics_error_rows": malformed_or_error_rows,
        "roles": {},
    }
    for role in ("prefill", "decode"):
        samples = series[role]
        if not samples or samples[0][0] > begin or samples[-1][0] < end:
            raise ValueError(f"{role} metrics do not cover T50--T450")
        category = "extend" if role == "prefill" else "decode"
        forward_key = key(
            samples,
            f"sglang_forward_execution_seconds_total|category={category}",
            f"sglang_gpu_execution_seconds_total|category=forward_{category}",
        )
        token_mode = "prefill_compute" if role == "prefill" else "decode"
        token_key = key(samples, f"sglang_realtime_tokens_total|mode={token_mode}")
        gauges = {}
        for name in (
            "sglang_full_token_usage",
            "sglang_num_running_reqs",
            "sglang_num_queue_reqs",
            "sglang_num_prefill_inflight_queue_reqs",
            "sglang_num_decode_prealloc_queue_reqs",
            "sglang_num_decode_transfer_queue_reqs",
        ):
            gauges[name] = gauge_stats(samples, name, begin, end)
        role_result = {
            "token_s": interval_value(samples, token_key, begin, end, True),
            "forward_fraction_per_gpu": interval_value(
                samples, forward_key, begin, end, True
            ) / args.tp,
            "gauges": gauges,
            "max_scrape_gap_seconds": max(
                right[0] - left[0]
                for left, right in zip(samples, samples[1:])
                if left[0] < end and right[0] > begin
            ),
        }
        if role == "prefill":
            cache_metric = "sglang_realtime_tokens_total|mode=prefill_cache"
            # Native one-way PD intentionally disables D->P restoration and
            # older stock SGLang does not publish a zero-valued prefill_cache
            # counter in that mode.  Its absence means every prompt token was
            # recomputed, not that the run is missing metrics.
            cache_s = (
                interval_value(samples, cache_metric, begin, end, True)
                if any(cache_metric in metrics for _, metrics in samples)
                else 0.0
            )
            role_result["cache_token_s"] = cache_s
            total = role_result["token_s"] + cache_s
            role_result["prompt_token_s"] = total
            role_result["cache_reuse"] = cache_s / total if total else None
        output["roles"][role] = role_result

    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
