#!/usr/bin/env python3
"""Render the incrementally completed Qwen3-8B agentic matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = (
    ("new", "当前新方法"),
    ("no-reverse", "No-reverse PD"),
    ("hicache", "原生 Mooncake"),
    ("colocated", "Colocated baseline"),
)


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[384, 512, 640])
    args = parser.parse_args()

    rows = []
    completed = []
    for concurrency in args.concurrencies:
        for method, label in METHODS:
            run_dir = args.run_root / f"{args.prefix}-{method}-c{concurrency}-w300-m1200-h100"
            summary_path = run_dir / "agentic_steady_summary.json"
            if not summary_path.exists():
                continue
            data = json.loads(summary_path.read_text())
            completed.append(run_dir.name)
            count = data["completed_agents"]
            shell_per_agent = data.get("shell_calls", 0) / count if count else 0.0
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(concurrency),
                        label,
                        number(data["agent_s"], 4),
                        f'{count:,} / {data["failed_agents"]}',
                        f'{number(data["turns"]["mean"])} / {number(data["turns"]["median"], 0)} / {number(data["turns"]["p90"], 0)}',
                        number(shell_per_agent),
                        number(data["tool_time_seconds"]["median"]),
                        percent(data["prefill"]["forward_fraction_per_gpu"]),
                        percent(data["decode"]["forward_fraction_per_gpu"]),
                        percent(data["prefill"]["full_token_usage_peak_per_gpu"]),
                        percent(data["decode"]["full_token_usage_peak_per_gpu"]),
                    ]
                )
                + " |"
            )

    lines = [
        "# Terminal-Bench 2.1 pilot + Qwen3-8B（H100）",
        "",
        "节点：`node2700`；GPU：8×H100；数据为固定顺序循环的 16 道 Docker 任务，",
        "seed 2026、temperature 0、top_p 1、top_k -1、context 40960、最多 32 轮，",
        "每组使用 300 秒业务预热 + 1200 秒正式测量。",
        "",
        "PD 配置为 4P:4D、TP=1，P=0/2/4/6、D=1/3/5/7。新方法统一使用",
        "D→P Host 总量 512 GiB（128 GiB/P）和 P→D Host 总量 256 GiB（64 GiB/P）；",
        "原生 Mooncake 保留 stock SGLang 的共享 segment/HiCache 机制。",
        "",
        "| 并发 | 方法 | Agent/s | 完成/失败 | 轮次均值 / 中位 / P90 | Shell/Agent | 工具时间中位(s) | P Forward/卡 | D Forward/卡 | P KV峰值 | D KV峰值 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"已完成 **{len(rows)}/12** 组；每一行均来自对应目录的正式测量窗口。",
        "",
        "有效结果目录：",
        "",
        *[f"- `{name}`" for name in completed],
        "",
    ]
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
