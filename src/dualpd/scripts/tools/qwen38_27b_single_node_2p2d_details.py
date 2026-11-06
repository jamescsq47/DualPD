#!/usr/bin/env python3
"""Render T50--T450 details for the single-node Qwen3.8-27B 2P:2D matrix."""

from pathlib import Path

import qwen38_27b_single_node_1p3d_details as details


details.REPORT = (
    Path(__file__).resolve().parents[2]
    / "runs-host/SWEBENCH_QWEN38_27B_TP2_SINGLE_NODE_2P2D_H100.md"
)
details.CONCURRENCIES = (96, 128, 192)
details.METHOD_LABEL = {
    "agentic": "当前新方法（2P:2D）",
    "colocated": "Colocated（4×TP2）",
    "native": "No-reverse PD（2P:2D）",
    "hicache": "原生 HiCache/Mooncake（2P:2D）",
}
details.COLOCATED_GROUPS = 4
details.PD_PREFILL_GROUPS = 2
details.PD_DECODE_GROUPS = 2


def run_name(concurrency: int, method: str) -> str:
    stem = {
        "agentic": "agentic",
        "colocated": "colocated4",
        "native": "native-oneway",
        "hicache": "hicache-mooncake",
    }[method]
    revision = (
        "correctness-r1" if method in {"colocated", "native"}
        else ("r2" if method == "agentic" else "r1")
    )
    return (
        f"q38-27b-h100-1n-2p2d-{stem}-swe500-c{concurrency}-"
        f"20261016-{revision}"
    )


details.RUNS = tuple(
    (concurrency, method, run_name(concurrency, method))
    for concurrency in details.CONCURRENCIES
    for method in details.METHODS
)


if __name__ == "__main__":
    details.main()
