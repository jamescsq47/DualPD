#!/usr/bin/env python3
"""Summarize one stopped MiniMax service log over an absolute UTC window.

The T450-only runner did not persist Prometheus samples.  This reader therefore
uses only values that remain auditable in the scheduler/service log.  In
particular, ``active_second_fraction`` is a coarse log-activity proxy and must
not be presented as the CUDA device-timer Forward fraction.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
import re
from zoneinfo import ZoneInfo


STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: TP(\d+) EP\d+)?\]")
PREFILL = re.compile(
    r"Prefill batch.*?#new-seq: (\d+), #new-token: (\d+), "
    r"#cached-token: (\d+), token usage: ([0-9.]+), #running-req: (\d+), "
    r"#queue-req: (\d+), #prealloc-req: (\d+), #inflight-req: (\d+)"
)
DECODE = re.compile(
    r"Decode batch.*?#running-req: (\d+), #token: (\d+), token usage: ([0-9.]+), "
    r"pre-allocated usage: ([0-9.]+), #prealloc-req: (\d+), "
    r"#transfer-req: (\d+), #retracted-req: (\d+).*?"
    r"gen throughput \(token/s\): ([0-9.]+), #queue-req: (\d+)"
)
HANDOFF = re.compile(
    r"slow ownership handoff: key=([^ ]+) path=([^ ]+) operation=([^ ]+) "
    r"tokens=(\d+) total_s=([0-9.]+).*?dma_s=([0-9.]+)"
)
POLICY = re.compile(r"Agentic V2 policy progress (\{.*\})$")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    left = math.floor(index)
    right = math.ceil(index)
    if left == right:
        return ordered[left]
    return ordered[left] * (right - index) + ordered[right] * (index - left)


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--begin", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--log-timezone", default="America/New_York")
    args = parser.parse_args()

    zone = ZoneInfo(args.log_timezone)
    prefill: dict[str, list[float]] = {
        key: [] for key in ("new_seq", "new", "cached", "kv", "running", "queue", "prealloc", "inflight")
    }
    decode: dict[str, list[float]] = {
        key: [] for key in ("running", "tokens", "kv", "prealloc_usage", "prealloc", "transfer", "retracted", "gen_tps", "queue")
    }
    active_seconds: set[int] = set()
    handoffs: Counter[str] = Counter()
    handoff_tokens: Counter[str] = Counter()
    handoff_total_s: dict[str, list[float]] = {}
    handoff_dma_s: dict[str, list[float]] = {}
    policy: dict[str, list[float]] = {
        key: [] for key in ("d2p_host_entries", "d2p_host_gib", "d2p_host_evictions")
    }

    with open(args.log, errors="replace") as stream:
        for line in stream:
            stamp = STAMP.search(line)
            if not stamp:
                continue
            when = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=zone).timestamp()
            if when < args.begin or when > args.end:
                continue
            rank = stamp.group(2)
            # Scheduler batch and coordinator lines are emitted by rank zero.
            if rank not in (None, "0"):
                continue

            match = PREFILL.search(line)
            if match:
                values = list(map(float, match.groups()))
                for key, value in zip(prefill, values):
                    prefill[key].append(value)
                active_seconds.add(int(when))
                continue

            match = DECODE.search(line)
            if match:
                values = list(map(float, match.groups()))
                for key, value in zip(decode, values):
                    decode[key].append(value)
                active_seconds.add(int(when))
                continue

            match = HANDOFF.search(line)
            if match:
                _key, path, operation, tokens, total_s, dma_s = match.groups()
                label = f"{path}:{operation}"
                handoffs[label] += 1
                handoff_tokens[label] += int(tokens)
                handoff_total_s.setdefault(label, []).append(float(total_s))
                handoff_dma_s.setdefault(label, []).append(float(dma_s))
                continue

            match = POLICY.search(line)
            if match:
                try:
                    payload = ast.literal_eval(match.group(1))
                except (SyntaxError, ValueError):
                    continue
                for key in policy:
                    if key in payload:
                        policy[key].append(float(payload[key]))

    duration = args.end - args.begin
    result = {
        "window_seconds": duration,
        "active_seconds": len(active_seconds),
        "active_second_fraction": len(active_seconds) / duration if duration > 0 else None,
        "prefill": {
            "batches": len(prefill["new"]),
            "new_tokens": sum(prefill["new"]),
            "cached_tokens": sum(prefill["cached"]),
            "cache_reuse": (
                sum(prefill["cached"]) / (sum(prefill["new"]) + sum(prefill["cached"]))
                if sum(prefill["new"]) + sum(prefill["cached"]) else None
            ),
            **{key: stats(values) for key, values in prefill.items() if key not in ("new", "cached")},
        },
        "decode": {
            "samples": len(decode["running"]),
            **{key: stats(values) for key, values in decode.items()},
        },
        "handoffs": {
            key: {
                "count": count,
                "tokens": handoff_tokens[key],
                "total_s": stats(handoff_total_s[key]),
                "dma_s": stats(handoff_dma_s[key]),
            }
            for key, count in sorted(handoffs.items())
        },
        "policy": {key: stats(values) for key, values in policy.items()},
        "policy_eviction_delta": (
            policy["d2p_host_evictions"][-1] - policy["d2p_host_evictions"][0]
            if len(policy["d2p_host_evictions"]) >= 2 else None
        ),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
