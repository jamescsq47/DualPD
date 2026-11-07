#!/usr/bin/env python3
"""One-case P->D dual-path ablation for Qwen3-8B BrowseComp c576.

This keeps the completed 4P:4D H100 new-method configuration fixed, except
that P->D late binding and Host staging are jointly disabled.  The router
reserves a Decode worker before Prefill and the completed KV is delivered to
that fixed worker through the native Direct transfer path.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / "."
LOWC = ROOT / "scripts/cluster/browsecomp_qwen3_8b_lowc_h100_matrix.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-prealloc-c576-20261027"
)
REPORT = PD / "runs-host/BROWSECOMP_QWEN3_8B_H100_P2D_PREALLOC_ABLATION.md"
RUN_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-prealloc-direct-c576-w300-m1200-20261027-r1"
)
FULL_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1"
)
COLOCATED_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-colocated-c576-w300-m1200-20261006-r1"
)
CASE = ("browsecomp8b4p4d", "p2d_prealloc_direct", 576)
CASES = [CASE]
FULL_RESOURCE = {
    "p_forward": 0.7322, "p_kv": 0.7740, "p_queue": 51.08,
    "p_inflight": 17.30, "d_forward": 0.9748, "d_kv": 0.8507,
    "d_running": 52.33, "d_queue": 0.00, "d_prealloc": 1.88,
    "d_transfer": 0.36,
}
COLOCATED_RESOURCE = {
    "p_forward": 0.6324, "d_forward": 0.3662, "d_kv": 0.8675,
    "d_running": 66.64, "d_queue": 4.23,
}


spec = importlib.util.spec_from_file_location("browsecomp8b_prealloc_base", LOWC)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {LOWC}")
lowc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lowc
spec.loader.exec_module(lowc)
base = lowc.base
base.CONTROL = CONTROL
base.CASES = CASES


def case_key(_case: tuple[str, str, int]) -> str:
    return "browsecomp8b-4p4d-p2d-prealloc-direct-c576"


def run_dir(_case: tuple[str, str, int]) -> Path:
    return RUN_DIR


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    # Start from the exact successful c576 4P:4D new-method path.  These two
    # switches are the only behavioral changes in the experiment.
    env, command = lowc.modified_environment(
        (case[0], "new_method", case[2]), directory
    )
    env.update(
        SGLANG_PD_ABLATION_P2D_PREBIND="true",
        P2D_HOST_STAGING="false",
        SGLANG_AGENTIC_KV_P2D_HOST_STAGING="false",
        ABLATION_NAME="p2d-prealloc-direct-only",
    )
    return env, command


def validate_inputs() -> None:
    lowc.validate_inputs()
    required = [
        FULL_DIR / "offload_analysis_summary.json",
        COLOCATED_DIR / "offload_analysis_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing comparison inputs:\n" + "\n".join(missing))


def read_result(directory: Path) -> dict[str, Any]:
    for name in ("h100_result.json", "offload_analysis_summary.json"):
        path = directory / name
        if path.exists():
            value = json.loads(path.read_text())
            if isinstance(value, dict):
                return value
    return {}


def metric(result: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if result.get(key) is not None:
            return result[key]
    return None


def number(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def percent(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def normalized(result: dict[str, Any]) -> dict[str, Any]:
    reuse = result.get("completion_weighted_reuse", {})
    if not isinstance(reuse, dict):
        reuse = {}
    return {
        "decode_tps": metric(result, "decode_tps"),
        "agent_s": metric(result, "agent_s", "completion_rps"),
        "prefill_tps": metric(result, "prefill_tps", "prefill_compute_tps"),
        "completed": metric(result, "completed_agents"),
        "reuse": metric(result, "parent_kv_reuse")
        if result.get("parent_kv_reuse") is not None
        else reuse.get("page_aligned_parent_prefix_reuse_fraction"),
        "p_forward": metric(result, "prefill_forward_per_gpu"),
        "d_forward": metric(result, "decode_forward_per_gpu"),
        "p_kv": metric(result, "p_kv"),
        "p_queue": metric(result, "p_queue"),
        "p_inflight": metric(result, "p_inflight"),
        "d_kv": metric(result, "d_kv"),
        "d_running": metric(result, "d_running"),
        "d_queue": metric(result, "d_queue"),
        "d_prealloc": metric(result, "d_prealloc"),
        "d_transfer": metric(result, "d_transfer"),
    }


def render_report(states: dict[str, Any]) -> None:
    full = normalized(read_result(FULL_DIR))
    colocated = normalized(read_result(COLOCATED_DIR))
    full.update(FULL_RESOURCE)
    colocated.update(COLOCATED_RESOURCE)
    item = states.get(case_key(CASE), {})
    ablation = normalized(item.get("result", {}) if isinstance(item, dict) else {})
    status = {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（等待修复）",
    }.get(item.get("status") if isinstance(item, dict) else None, "排队/待运行")

    def row(label: str, status_text: str, result: dict[str, Any]) -> str:
        completed = result.get("completed")
        return (
            f"| {label} | {status_text} | {number(result.get('decode_tps'))} | "
            f"{number(result.get('agent_s'), 4)} | {number(result.get('prefill_tps'))} | "
            f"{percent(result.get('reuse'))} | "
            f"{'—' if completed is None else f'{int(completed):,}'} |"
        )

    full_tps = full.get("decode_tps")
    colocated_tps = colocated.get("decode_tps")
    ablation_tps = ablation.get("decode_tps")
    speedup = (
        float(ablation_tps) / float(colocated_tps)
        if ablation_tps is not None and colocated_tps
        else None
    )
    dual_gain = (
        float(full_tps) / float(ablation_tps) - 1.0
        if full_tps is not None and ablation_tps
        else None
    )
    lines = [
        "# BrowseComp Qwen3-8B H100：P→D 双路径消融",
        "",
        "> 配置：单节点 H100×8，TP=1，4P:4D，c576，300 秒业务预热 + 1200 秒稳态测量。",
        "> 消融仅设置 `SGLANG_PD_ABLATION_P2D_PREBIND=true` 与 `P2D_HOST_STAGING=false`：",
        "> Router 在 Prefill 前预留固定 Decode，P→D 只走 Direct；D→P 机制及其余超参保持完整方法不变。",
        "",
        "## 核心结果",
        "",
        "| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成数 |",
        "|---|---|---:|---:|---:|---:|---:|",
        row("Colocated baseline（复用）", "完成", colocated),
        row("完整 P→D Direct/Host 双路径（复用）", "完成", full),
        row("P→D preallocate + Direct-only", status, ablation),
        "",
        "## 对比",
        "",
        f"- 完整双路径相对 Colocated：`{number(float(full_tps) / float(colocated_tps), 3) if full_tps and colocated_tps else '—'}×`。",
        f"- 消融相对 Colocated：`{number(speedup, 3) if speedup is not None else '—'}×`。",
        f"- 完整双路径相对消融的吞吐提升：`{percent(dual_gain) if dual_gain is not None else '—'}`。",
        "",
        "## 稳态资源细化",
        "",
        "| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in (
        ("Colocated baseline", colocated),
        ("完整双路径", full),
        ("Preallocate + Direct-only", ablation),
    ):
        lines.append(
            f"| {label} | {percent(result.get('p_forward'))} | {percent(result.get('p_kv'))} | "
            f"{number(result.get('p_queue'), 2)} | {number(result.get('p_inflight'), 2)} | "
            f"{percent(result.get('d_forward'))} | {percent(result.get('d_kv'))} | "
            f"{number(result.get('d_running'), 2)} | {number(result.get('d_queue'), 2)} | "
            f"{number(result.get('d_prealloc'), 2)} | {number(result.get('d_transfer'), 2)} |"
        )
    lines.extend([
        "",
        "## 固定配置与原始记录",
        "",
        "- GPU 映射：P=`0,2,4,6`，D=`1,3,5,7`；各引擎 TP=1。",
        "- Host 容量：D→P 每个 P 128 GiB（总 512 GiB）；P→D 每个 P 64 GiB（总 256 GiB，但本消融禁用 P→D Host）。",
        "- 阈值：fast-tool=1 秒，Direct handshake=1 秒；Slow 拥堵 high/low=`32/8`。",
        f"- Colocated：`{COLOCATED_DIR}`",
        f"- 完整双路径：`{FULL_DIR}`",
        f"- 本消融：`{RUN_DIR}`",
        f"- 控制状态：`{CONTROL}`",
        "",
    ])
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(REPORT)


def cleanup_gate(states: dict[str, Any]) -> None:
    previous_path = CONTROL / "state.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    key = case_key(CASE)
    newly_complete = (
        states.get(key, {}).get("status") == "complete"
        and previous.get(key, {}).get("status") != "complete"
    )
    if not newly_complete:
        return
    env, _ = case_environment(CASE, RUN_DIR)
    base.wait_ports_free(env, timeout=300)
    occupied = base.gpu_processes()
    if occupied:
        raise RuntimeError(f"post-case GPU cleanup gate failed: {occupied}")
    helper_names = {"apptainer", "squashfuse_ll", "fuse-overlayfs"}
    helpers: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text().strip()
            if name in helper_names:
                helpers.append((int(entry.name), name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    if helpers:
        raise RuntimeError(f"post-case Apptainer/FUSE cleanup gate failed: {helpers}")


def atomic_json(path: Path, value: object) -> None:
    if path == CONTROL / "state.json" and isinstance(value, dict):
        cleanup_gate(value)
    lowc.original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        render_report(value)


base.case_key = case_key
base.run_dir = run_dir
base.case_environment = case_environment
base.validate_inputs = validate_inputs
base.atomic_json = atomic_json


if __name__ == "__main__":
    existing = (
        json.loads((CONTROL / "state.json").read_text())
        if (CONTROL / "state.json").exists()
        else {}
    )
    render_report(existing)
    base.main()
