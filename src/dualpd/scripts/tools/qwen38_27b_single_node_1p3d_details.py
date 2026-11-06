#!/usr/bin/env python3
"""Render T50--T450 details for the single-node Qwen3.8-27B matrix.

This is an offline reader.  It only includes runs that have at least 450
terminal episode records and complete engine-metric coverage for the window.
It never probes, restarts, or otherwise changes a live experiment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from summarize_native_swe_colocated import interval_value


RUN_ROOT = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
REPORT = Path(__file__).resolve().parents[2] / "runs-host/SWEBENCH_QWEN38_27B_TP2_SINGLE_NODE_1P3D_H100.md"
START_MARKER = "<!-- T50_T450_DETAILS_START -->"
END_MARKER = "<!-- T50_T450_DETAILS_END -->"
TP = 2
COLOCATED_GROUPS = 4
PD_PREFILL_GROUPS = 1
PD_DECODE_GROUPS = 3

METHODS = ("agentic", "colocated", "native", "hicache")
CONCURRENCIES = (64, 128, 192)
METHOD_LABEL = {
    "agentic": "当前新方法（1P:3D）",
    "colocated": "Colocated（4×TP2）",
    "native": "No-reverse PD（1P:3D）",
    "hicache": "原生 HiCache/Mooncake（1P:3D）",
}


def run_name(concurrency: int, method: str) -> str:
    stem = {
        "agentic": "agentic",
        "colocated": "colocated4",
        "native": "native-oneway",
        "hicache": "hicache-mooncake",
    }[method]
    revision = "correctness-r3" if method == "colocated" else (
        "correctness-r4" if method == "native" else "r2"
    )
    return (
        f"q38-27b-h100-1n-1p3d-{stem}-swe500-c{concurrency}-"
        f"20261014-{revision}"
    )


RUNS = tuple(
    (concurrency, method, run_name(concurrency, method))
    for concurrency in CONCURRENCIES
    for method in METHODS
)

COUNTER_KEYS = {
    "sglang_realtime_tokens_total|mode=prefill_compute",
    "sglang_realtime_tokens_total|mode=prefill_cache",
    "sglang_realtime_tokens_total|mode=decode",
    "sglang_forward_execution_seconds_total|category=extend",
    "sglang_forward_execution_seconds_total|category=decode",
    "sglang_gpu_execution_seconds_total|category=forward_extend",
    "sglang_gpu_execution_seconds_total|category=forward_decode",
}
GAUGE_KEYS = {
    "sglang_full_token_usage",
    "sglang_mamba_usage",
    "sglang_num_running_reqs",
    "sglang_num_queue_reqs",
    "sglang_num_prefill_inflight_queue_reqs",
    "sglang_num_decode_prealloc_queue_reqs",
    "sglang_num_decode_transfer_queue_reqs",
}
KEEP_KEYS = COUNTER_KEYS | GAUGE_KEYS


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(errors="replace") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_series(
    metrics_path: Path, colocated: bool
) -> dict[str, list[list[tuple[float, dict[str, float]]]]]:
    endpoints: dict[tuple[str, str], list[tuple[float, dict[str, float]]]] = defaultdict(list)
    with metrics_path.open(errors="replace") as stream:
        for line in stream:
            row = json.loads(line)
            role = row.get("role")
            if colocated and role != "prefill":
                continue
            for endpoint in row.get("endpoint_metrics", []):
                metrics = {
                    key: float(value)
                    for key, value in endpoint.get("metrics", {}).items()
                    if key in KEEP_KEYS and isinstance(value, (int, float))
                }
                if metrics:
                    endpoints[(role, endpoint["endpoint"])].append(
                        (float(row["ts"]), metrics)
                    )
    for samples in endpoints.values():
        samples.sort(key=lambda item: item[0])
    if colocated:
        shared = [
            samples
            for (role, _), samples in endpoints.items()
            if role == "prefill"
        ]
        if len(shared) != COLOCATED_GROUPS:
            raise ValueError(
                f"expected {COLOCATED_GROUPS} colocated TP2 groups, found {len(shared)}"
            )
        return {"prefill": shared, "decode": shared}
    result = {
        role: [
            samples
            for (sample_role, _), samples in endpoints.items()
            if sample_role == role
        ]
        for role in ("prefill", "decode")
    }
    if (
        len(result["prefill"]) != PD_PREFILL_GROUPS
        or len(result["decode"]) != PD_DECODE_GROUPS
    ):
        raise ValueError(
            f"expected {PD_PREFILL_GROUPS}P:{PD_DECODE_GROUPS}D TP2 groups, found "
            f"{len(result['prefill'])}P:{len(result['decode'])}D"
        )
    return result


def metric_key(
    groups: list[list[tuple[float, dict[str, float]]]], *candidates: str
) -> str:
    for candidate in candidates:
        if any(candidate in metrics for samples in groups for _, metrics in samples):
            return candidate
    raise ValueError(f"none of the metric candidates were present: {candidates}")


def counter_total(
    groups: list[list[tuple[float, dict[str, float]]]],
    key: str,
    begin: float,
    end: float,
) -> float:
    values = [interval_value(samples, key, begin, end, True) for samples in groups]
    if any(value is None for value in values):
        raise ValueError(f"missing counter {key}")
    return sum(value for value in values if value is not None)


def gauge_mean(
    groups: list[list[tuple[float, dict[str, float]]]],
    key: str,
    begin: float,
    end: float,
) -> float | None:
    values = [interval_value(samples, key, begin, end, False) for samples in groups]
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def role_metrics(
    groups: list[list[tuple[float, dict[str, float]]]],
    role: str,
    begin: float,
    end: float,
) -> dict[str, float | None]:
    token_key = (
        "sglang_realtime_tokens_total|mode=prefill_compute"
        if role == "prefill"
        else "sglang_realtime_tokens_total|mode=decode"
    )
    category = "extend" if role == "prefill" else "decode"
    forward_key = metric_key(
        groups,
        f"sglang_forward_execution_seconds_total|category={category}",
        f"sglang_gpu_execution_seconds_total|category=forward_{category}",
    )
    forward_values = [
        interval_value(samples, forward_key, begin, end, True) for samples in groups
    ]
    if any(value is None for value in forward_values):
        raise ValueError(f"missing forward timer for {role}")
    # Token counters are rank-0 logical counters.  Forward timers contain both
    # TP ranks, so divide their per-group mean by TP=2 for a physical-GPU share.
    forward_per_gpu = (
        sum(value for value in forward_values if value is not None)
        / len(forward_values)
        / TP
    )
    gaps = [
        right[0] - left[0]
        for samples in groups
        for left, right in zip(samples, samples[1:])
        if left[0] < end and right[0] > begin
    ]
    return {
        "token_s": counter_total(groups, token_key, begin, end),
        "forward_per_gpu": forward_per_gpu,
        "kv": gauge_mean(groups, "sglang_full_token_usage", begin, end),
        "mamba": gauge_mean(groups, "sglang_mamba_usage", begin, end),
        "running": gauge_mean(groups, "sglang_num_running_reqs", begin, end),
        "queue": gauge_mean(groups, "sglang_num_queue_reqs", begin, end),
        "inflight": gauge_mean(
            groups, "sglang_num_prefill_inflight_queue_reqs", begin, end
        ),
        "prealloc": gauge_mean(
            groups, "sglang_num_decode_prealloc_queue_reqs", begin, end
        ),
        "transfer": gauge_mean(
            groups, "sglang_num_decode_transfer_queue_reqs", begin, end
        ),
        "coverage_first": min(samples[0][0] for samples in groups),
        "coverage_last": max(samples[-1][0] for samples in groups),
        "max_gap": max(gaps, default=0.0),
    }


def summarize(concurrency: int, method: str, run_id: str) -> dict[str, Any]:
    run = RUN_ROOT / run_id
    progress = jsonl(run / "episode_progress.jsonl")
    completed_at = sorted(
        float(row["completed_at"])
        for row in progress
        if row.get("completed_at") is not None
    )
    if len(completed_at) < 450:
        raise ValueError(f"only {len(completed_at)}/450 terminal records")
    begin, end = completed_at[49], completed_at[449]
    series = build_series(
        run / "workload/engine_metrics.jsonl", method == "colocated"
    )
    roles = {
        role: role_metrics(groups, role, begin, end)
        for role, groups in series.items()
    }
    for role, values in roles.items():
        if values["coverage_first"] > begin or values["coverage_last"] < end:
            raise ValueError(f"{role} metrics do not cover T50--T450")
    prefill_groups = series["prefill"]
    compute_s = float(roles["prefill"]["token_s"])
    # No-reverse deliberately disables radix/prefix reuse and its baseline
    # therefore does not export the optional prefill_cache counter at all.
    # Treat that method-specific absence as an exact zero; require the counter
    # for every cache-enabled method so exporter regressions cannot look like
    # a measured 0% reuse rate.
    cache_s = 0.0 if method == "native" else counter_total(
        prefill_groups,
        "sglang_realtime_tokens_total|mode=prefill_cache",
        begin,
        end,
    )
    logical_s = compute_s + cache_s
    reuse = cache_s / logical_s if logical_s else None
    return {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_id,
        "seconds": end - begin,
        "agent_s": 400.0 / (end - begin),
        "reuse": reuse,
        "roles": roles,
    }


def number(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def percent(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{100 * value:.2f}%"


def render(results: list[dict[str, Any]], skipped: dict[str, str]) -> str:
    by_key = {(row["concurrency"], row["method"]): row for row in results}
    lines = [
        "## T50–T450 P/D Forward 与吞吐",
        "",
        "下表统一取第50条至第450条terminal result之间的主体窗口。Prefill/Decode tok/s为scheduler实际计算量。",
        "",
        "Forward计数器包含两个TP rank，因此先按TP组求均值，再除以TP=2，得到每张物理GPU的墙钟占比。PD的P/D位于不同GPU，二者不能相加；Colocated的Extend/Decode位于同一批GPU，可以相加理解计算时间构成。",
        "",
        "| 并发 | 方法 | 窗口(s) | Agent/s | P compute tok/s | D tok/s | P/Extend Forward/GPU | D/Decode Forward/GPU |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, method, run_id in RUNS:
        result = by_key.get((concurrency, method))
        if result is None:
            lines.append(
                f"| {concurrency} | {METHOD_LABEL[method]} | — | — | — | — | — | — |"
            )
            continue
        p = result["roles"]["prefill"]
        d = result["roles"]["decode"]
        lines.append(
            "| {c} | {method} | {seconds} | {agent} | {ptps} | {dtps} | "
            "{pfwd} | {dfwd} |".format(
                c=concurrency,
                method=METHOD_LABEL[method],
                seconds=number(result["seconds"]),
                agent=number(result["agent_s"], 5),
                ptps=number(p["token_s"]),
                dtps=number(d["token_s"]),
                pfwd=percent(p["forward_per_gpu"]),
                dfwd=percent(d["forward_per_gpu"]),
            )
        )
    lines.extend([
        "",
        "## T50–T450 复用率与缓存/队列",
        "",
        "有效复用率按 `P cache / (P cache + P compute)` 计算，等价于 `1 - 实际Prefill计算量 / 逻辑Prompt量`。No-reverse关闭复用且不导出 `prefill_cache` counter，因此按精确的0%记录。",
        "",
        "| 并发 | 方法 | 有效复用率 | P queue | D running | D prealloc / transfer | Attention KV P/D | Mamba P/D |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for concurrency, method, run_id in RUNS:
        result = by_key.get((concurrency, method))
        if result is None:
            lines.append(
                f"| {concurrency} | {METHOD_LABEL[method]} | — | — | — | — | — | — |"
            )
            continue
        p = result["roles"]["prefill"]
        d = result["roles"]["decode"]
        if method == "colocated":
            kv = f"{percent(d['kv'])}（共享）"
            mamba = f"{percent(d['mamba'])}（共享）"
        else:
            kv = f"{percent(p['kv'])} / {percent(d['kv'])}"
            mamba = f"{percent(p['mamba'])} / {percent(d['mamba'])}"
        lines.append(
            "| {c} | {method} | {reuse} | {pqueue} | {drunning} | "
            "{dprealloc} / {dtransfer} | {kv} | {mamba} |".format(
                c=concurrency,
                method=METHOD_LABEL[method],
                reuse=percent(result["reuse"]),
                pqueue=number(p["queue"]),
                drunning=number(d["running"]),
                dprealloc=number(d["prealloc"]),
                dtransfer=number(d["transfer"]),
                kv=kv,
                mamba=mamba,
            )
        )
    lines.extend([
        "",
        "Attention KV/Mamba为各TP组活跃、不可驱逐缓存池的时间均值，不是整卡HBM占比；queue、running、prealloc、transfer均为每个TP组的时间均值。空白行表示该run尚未达到T450或时序metrics尚不完整。",
        "",
    ])
    return "\n".join(lines)


def update_report(markdown: str) -> None:
    text = REPORT.read_text()
    block = f"{START_MARKER}\n{markdown.rstrip()}\n{END_MARKER}\n"
    if START_MARKER in text and END_MARKER in text:
        before, tail = text.split(START_MARKER, 1)
        _, after = tail.split(END_MARKER, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    REPORT.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-report", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for concurrency, method, run_id in RUNS:
        try:
            results.append(summarize(concurrency, method, run_id))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            skipped[run_id] = str(error)
    if args.json:
        args.json.write_text(
            json.dumps({"results": results, "skipped": skipped}, indent=2) + "\n"
        )
    markdown = render(results, skipped)
    if args.update_report:
        update_report(markdown)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
