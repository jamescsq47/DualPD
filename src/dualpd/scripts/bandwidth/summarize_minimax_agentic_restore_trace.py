#!/usr/bin/env python3
"""Summarize c448 Agentic ownership traces without changing runtime state."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


PATTERN = re.compile(
    r"path=(?P<path>\S+) operation=(?P<operation>\S+) tokens=(?P<tokens>\d+) "
    r"total_s=(?P<total>[\d.]+) prepare_s=(?P<prepare>[\d.]+) "
    r"dma_s=(?P<dma>[\d.]+) bind_s=(?P<bind>[\d.]+) "
    r"release_s=(?P<release>[\d.]+)"
)


def stats(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    pick = lambda q: values[round((len(values) - 1) * q)]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": pick(0.9),
        "max": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--kv-bytes-per-token-per-rank", type=int, default=31744)
    args = parser.parse_args()
    groups: dict[str, list[dict[str, float]]] = {}
    for line in args.log.read_text(errors="replace").splitlines():
        match = PATTERN.search(line)
        if not match:
            continue
        row = {key: float(value) for key, value in match.groupdict().items() if key not in {"path", "operation"}}
        row["ownership_non_dma_s"] = row["prepare"] + row["bind"] + row["release"]
        row["aggregate_gib"] = (
            row["tokens"] * args.kv_bytes_per_token_per_rank * args.tp_size / (1 << 30)
        )
        row["effective_transfer_phase_gib_s"] = (
            row["aggregate_gib"] / row["dma"] if row["dma"] >= 0.001 else None
        )
        groups.setdefault(f"{match['path']}:{match['operation']}", []).append(row)
    result = {
        "trace": str(args.log),
        "censoring": "runtime emits only ownership handoffs whose total_s >= 1.0; all summaries are slow-tail conditional statistics",
        "phase_semantics": (
            "dma is coordinator PREPARED-to-all-rank-DMA_DONE and includes command/scheduling overhead; "
            "ownership_non_dma is PREPARE+BIND+RELEASE only and is a lower bound, not scheduler-ready-to-Forward wait"
        ),
        "tp_size": args.tp_size,
        "kv_bytes_per_token_per_rank": args.kv_bytes_per_token_per_rank,
        "groups": {},
    }
    if not groups:
        raise SystemExit("no ownership handoff records matched")
    for key, rows in groups.items():
        result["groups"][key] = {
            name: stats([row[name] for row in rows if row[name] is not None])
            for name in (
                "tokens",
                "total",
                "prepare",
                "dma",
                "bind",
                "release",
                "ownership_non_dma_s",
                "effective_transfer_phase_gib_s",
            )
        }
        result["groups"][key]["matched_records"] = len(rows)
        result["groups"][key]["bandwidth_records_excluding_sub_ms"] = sum(
            row["effective_transfer_phase_gib_s"] is not None for row in rows
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
