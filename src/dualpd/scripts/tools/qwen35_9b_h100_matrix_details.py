"""Recompute T50--T450 engine details for the Qwen3.5-9B H100 matrix.

This is an offline reader.  It never probes or changes a live experiment.  A
run is included only after it has 500 unique request records, 500 terminal
episode records, no environment errors, and a complete summary.
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
RUNS = (
    (256, "agentic", "qwen35-9b-tp1-single-h100-agentic-2p6d-swe500-c256-20261009-r1"),
    (256, "colocated", "qwen35-9b-tp1-single-h100-colocated8-swe500-c256-20261009-r3"),
    (256, "native", "qwen35-9b-tp1-single-h100-native-oneway-2p6d-swe500-c256-20261009-r2"),
    (256, "hicache", "qwen35-9b-tp1-single-h100-hicache-mooncake-2p6d-swe500-c256-20261009-r9"),
    (394, "agentic", "qwen35-9b-tp1-single-h100-agentic-2p6d-swe500-c394-20261009-r1"),
    (394, "colocated", "qwen35-9b-tp1-single-h100-colocated8-swe500-c394-20261009-r1"),
    (394, "native", "qwen35-9b-tp1-single-h100-native-oneway-2p6d-swe500-c394-20261009-r1"),
    (394, "hicache", "qwen35-9b-tp1-single-h100-hicache-mooncake-2p6d-swe500-c394-20261009-r2"),
    (500, "agentic", "qwen35-9b-tp1-single-h100-agentic-2p6d-swe500-c500-20261009-r2"),
    (500, "colocated", "qwen35-9b-tp1-single-h100-colocated8-swe500-c500-20261009-r1"),
    (500, "native", "qwen35-9b-tp1-single-h100-native-oneway-2p6d-swe500-c500-20261009-r1"),
    (500, "hicache", "qwen35-9b-tp1-single-h100-hicache-mooncake-2p6d-swe500-c500-20261009-r1"),
)

METHOD_LABEL = {
    "agentic": "当前新方法",
    "colocated": "Colocated",
    "native": "No-reverse PD",
    "hicache": "原生 HiCache/Mooncake",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(errors="replace") as stream:
        for line in stream:
            records.append(json.loads(line))
    return records


def validate(run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workload = run / "workload"
    requests = jsonl(workload / "requests.jsonl")
    progress = jsonl(run / "episode_progress.jsonl")
    summary = json.loads((workload / "summary.json").read_text())
    instance_ids = [row["metadata"]["instance_id"] for row in requests]
    if len(requests) != 500 or len(set(instance_ids)) != 500:
        raise ValueError(f"{run.name}: requests are not 500 unique instances")
    if len(progress) != 500:
        raise ValueError(f"{run.name}: episode progress is {len(progress)}/500")
    progress_ids = [row["instance_id"] for row in progress]
    if len(set(progress_ids)) != 500:
        raise ValueError(f"{run.name}: duplicate episode progress records")
    if any(row.get("environment_error") for row in progress):
        raise ValueError(f"{run.name}: contains environment errors")
    if summary.get("completed") != 500 or summary.get("failed") != 0:
        raise ValueError(f"{run.name}: summary is not a clean 500/0")
    return requests, progress


def mean(values: list[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None) / len(values)


def metric_key(samples: list[tuple[float, dict[str, float]]], *candidates: str) -> str:
    for candidate in candidates:
        if any(candidate in metrics for _, metrics in samples):
            return candidate
    raise ValueError(f"none of the metric candidates were present: {candidates}")


def build_series(metrics_path: Path, colocated: bool) -> dict[str, list[list[tuple[float, dict[str, float]]]]]:
    endpoints: dict[tuple[str, str], list[tuple[float, dict[str, float]]]] = defaultdict(list)
    for row in jsonl(metrics_path):
        if colocated and row.get("role") != "prefill":
            continue
        for endpoint in row.get("endpoint_metrics", []):
            endpoints[(row["role"], endpoint["endpoint"])].append(
                (float(row["ts"]), endpoint.get("metrics", {}))
            )
    for samples in endpoints.values():
        samples.sort(key=lambda item: item[0])
    if colocated:
        shared = [samples for (role, _), samples in endpoints.items() if role == "prefill"]
        if len(shared) != 8:
            raise ValueError(f"expected 8 colocated endpoints, found {len(shared)}")
        return {"prefill": shared, "decode": shared}
    result = {
        role: [samples for (sample_role, _), samples in endpoints.items() if sample_role == role]
        for role in ("prefill", "decode")
    }
    if len(result["prefill"]) != 2 or len(result["decode"]) != 6:
        raise ValueError(
            f"expected 2P:6D endpoints, found {len(result['prefill'])}P:{len(result['decode'])}D"
        )
    return result


def counter_total(
    groups: list[list[tuple[float, dict[str, float]]]], key: str, begin: float, end: float
) -> float:
    values = [interval_value(samples, key, begin, end, True) for samples in groups]
    if any(value is None for value in values):
        raise ValueError(f"missing counter {key}")
    return sum(value for value in values if value is not None)


def gauge_mean(
    groups: list[list[tuple[float, dict[str, float]]]], key: str, begin: float, end: float
) -> float | None:
    return mean([interval_value(samples, key, begin, end, False) for samples in groups])


def role_metrics(
    groups: list[list[tuple[float, dict[str, float]]]], role: str, begin: float, end: float
) -> dict[str, float | None]:
    representative = [item for samples in groups for item in samples]
    token_key = metric_key(
        representative,
        f"sglang_realtime_tokens_total|mode={'prefill_compute' if role == 'prefill' else 'decode'}",
    )
    category = "extend" if role == "prefill" else "decode"
    forward_key = metric_key(
        representative,
        f"sglang_forward_execution_seconds_total|category={category}",
        f"sglang_gpu_execution_seconds_total|category=forward_{category}",
    )
    forward_values = [interval_value(samples, forward_key, begin, end, True) for samples in groups]
    if any(value is None for value in forward_values):
        raise ValueError(f"missing forward timer for {role}")
    forward = mean(forward_values)
    token_s = counter_total(groups, token_key, begin, end)
    active_tps = token_s / len(groups) / forward if forward else None
    first = min(t for samples in groups for t, metrics in samples if metrics)
    last = max(t for samples in groups for t, metrics in samples if metrics)
    gaps = [
        right[0] - left[0]
        for samples in groups
        for left, right in zip(samples, samples[1:])
        if left[0] < end and right[0] > begin
    ]
    return {
        "token_s": token_s,
        "forward": forward,
        "active_tps_per_gpu": active_tps,
        "coverage_first": first,
        "coverage_last": last,
        "max_gap": max(gaps, default=0.0),
        "kv": gauge_mean(groups, "sglang_full_token_usage", begin, end),
        "mamba": gauge_mean(groups, "sglang_mamba_usage", begin, end),
        "running": gauge_mean(groups, "sglang_num_running_reqs", begin, end),
        "queue": gauge_mean(groups, "sglang_num_queue_reqs", begin, end),
        "inflight": gauge_mean(groups, "sglang_num_prefill_inflight_queue_reqs", begin, end),
        "prealloc": gauge_mean(groups, "sglang_num_decode_prealloc_queue_reqs", begin, end),
        "transfer": gauge_mean(groups, "sglang_num_decode_transfer_queue_reqs", begin, end),
    }


def summarize(concurrency: int, method: str, run_id: str) -> dict[str, Any]:
    run = RUN_ROOT / run_id
    requests, progress = validate(run)
    completed = sorted(float(row["completed_at"]) for row in progress)
    begin, end = completed[49], completed[449]
    series = build_series(run / "workload" / "engine_metrics.jsonl", method == "colocated")
    roles = {
        role: role_metrics(groups, role, begin, end)
        for role, groups in series.items()
    }
    for role, values in roles.items():
        if values["coverage_first"] > begin or values["coverage_last"] < end:
            raise ValueError(
                f"{run_id}: {role} metrics do not cover T50--T450 "
                f"({values['coverage_first']}, {begin}, {end}, {values['coverage_last']})"
            )
    turns = [float(row.get("generation_turns", 0)) for row in requests]
    return {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_id,
        "seconds": end - begin,
        "agent_s": 400.0 / (end - begin),
        "mean_turns": sum(turns) / len(turns),
        "roles": roles,
    }


def number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:,.{digits}f}"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def render(results: list[dict[str, Any]]) -> str:
    lines = [
        "## T50–T450 P/D Forward 与窗口吞吐",
        "",
        "下表按每组第50条至第450条episode回调的绝对时间边界，直接对scheduler累计计数器做差。",
        "`token/GPU-forward-s` 为角色总 token/s 除以该角色GPU数和每卡Forward占比，表示每个GPU实际处于对应Forward期间的等效吞吐；它不是墙钟吞吐。",
        "",
        "| 并发 | 方法 | 窗口(s) | Agent/s | P token/s | P Forward/卡 | P token/GPU-forward-s | D token/s | D Forward/卡 | D token/GPU-forward-s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        prefill, decode = result["roles"]["prefill"], result["roles"]["decode"]
        lines.append(
            "| {c} | {method} | {seconds} | {agent} | {ptps} | {pfwd} | {pactive} | {dtps} | {dfwd} | {dactive} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                seconds=number(result["seconds"]), agent=number(result["agent_s"], 5),
                ptps=number(prefill["token_s"]), pfwd=percent(prefill["forward"]),
                pactive=number(prefill["active_tps_per_gpu"]),
                dtps=number(decode["token_s"]), dfwd=percent(decode["forward"]),
                dactive=number(decode["active_tps_per_gpu"]),
            )
        )
    lines.extend([
        "",
        "Colocated的P/D数据来自同一组8个引擎；其他三种方法的P和D分别来自独立的2张P卡与6张D卡。",
        "因此Colocated的P/D Forward可直接相加理解单卡计算时间构成；PD方法则应分别看P卡和D卡，不能跨角色相加。",
        "",
        "## T50–T450 资源均值",
        "",
        "| 并发 | 方法 | P KV | P Mamba | P queue | P inflight | D KV | D Mamba | D running | D queue | D prealloc | D transfer | 最大采样间隔(s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in results:
        p, d = result["roles"]["prefill"], result["roles"]["decode"]
        shared = result["method"] == "colocated"
        lines.append(
            "| {c} | {method} | {pkv} | {pmamba} | {pqueue} | {pinflight} | {dkv} | {dmamba} | {drunning} | {dqueue} | {dprealloc} | {dtransfer} | {gap} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                pkv="—" if shared else percent(p["kv"]),
                pmamba="—" if shared else percent(p["mamba"]),
                pqueue="—" if shared else number(p["queue"]),
                pinflight="—" if shared else number(p["inflight"]),
                dkv=percent(d["kv"]), dmamba=percent(d["mamba"]),
                drunning=number(d["running"]), dqueue=number(d["queue"]),
                dprealloc="—" if shared else number(d["prealloc"]),
                dtransfer="—" if shared else number(d["transfer"]),
                gap=number(max(p["max_gap"], d["max_gap"])),
            )
        )
    lines.extend([
        "",
        "KV和Mamba为各引擎设备池的不可驱逐使用率均值；queue、inflight、running、prealloc和transfer均为每引擎均值。",
        "Colocated共享同一套KV池和调度队列，故只在D侧列一次，P侧共享项记为“—”。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    results = []
    for concurrency, method, run_id in RUNS:
        try:
            results.append(summarize(concurrency, method, run_id))
        except (FileNotFoundError, ValueError) as error:
            if not args.allow_partial:
                raise
            print(f"<!-- skipped {run_id}: {error} -->")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    print(render(results))


if __name__ == "__main__":
    main()
