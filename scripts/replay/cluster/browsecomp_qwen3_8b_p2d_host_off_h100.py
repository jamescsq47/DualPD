#!/usr/bin/env python3
"""H100 c576 ablation disabling only P->D Host staging.

D->P Direct/Host recovery, congestion feedback, P->D late binding, workload,
topology, and the 300+1200 second measurement protocol remain identical to
the completed H100 full-method run.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PD = ROOT / "."
SOURCE = ROOT / "scripts/replay/cluster/browsecomp_qwen3_8b_lowc_h100_matrix.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-host-off-c576-20261028-r1"
)
RUN_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-host-off-c576-w300-m1200-20261028-r1"
)
REPORT = PD / "runs-host/BROWSECOMP_QWEN3_8B_H100_P2D_HOST_ABLATION.md"
CASE = ("browsecomp8b4p4d", "new_method", 576)
CASES = [CASE]


spec = importlib.util.spec_from_file_location("p2d_host_off_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
lowc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lowc
spec.loader.exec_module(lowc)
base = lowc.base

lowc.CONTROL = CONTROL
lowc.REPORT = REPORT
lowc.CASES = CASES
base.CONTROL = CONTROL
base.CASES = CASES


def case_key(_case: tuple[str, str, int]) -> str:
    return "browsecomp8b-4p4d-p2d-host-off-c576"


def run_dir(_case: tuple[str, str, int]) -> Path:
    return RUN_DIR


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = lowc.modified_environment(case, directory)
    env.update(
        # This is the only functional change from the H100 full method.
        P2D_HOST_STAGING="false",
        SGLANG_AGENTIC_KV_P2D_HOST_STAGING="false",
        # Preserve P->D late binding and do not enable any preallocation
        # ablation.  D->P Direct/Host recovery remains unchanged.
        SGLANG_PD_ABLATION_P2D_PREBIND="false",
        SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE="false",
        ABLATION_NAME="p2d-host-off-late-binding-on",
    )
    return env, command


def fmt(value: object, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def pct(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def render_report(states: dict[str, Any]) -> None:
    item = states.get(case_key(CASE), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}
    status = {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（等待修复）",
    }.get(str(item.get("status")), "排队/待运行")
    full_tps = 10778.5
    ablation_tps = result.get("decode_tps")
    relative = (
        float(ablation_tps) / full_tps if ablation_tps is not None else None
    )
    lines = [
        "# BrowseComp Qwen3-8B H100：仅关闭 P→D Host 消融",
        "",
        "> 单节点 H100×8，TP=1，4P:4D，c576，300 秒业务预热 + 1200 秒稳态。",
        "> D→P Direct/Host、Slow 拥堵重算与 P→D late binding 均保持不变；",
        "> 唯一变化为关闭 P→D Host staging。未启用 P→D prebind/native-prealloc。",
        "",
        "## 核心结果",
        "",
        "| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| 完整 P→D Direct/Host 双路径（复用） | 完成 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 / 0 |",
        f"| 仅关闭 P→D Host | {status} | {fmt(result.get('decode_tps'), 1)} | {fmt(result.get('agent_s'), 4)} | {fmt(result.get('prefill_tps'), 1)} | {pct(result.get('parent_kv_reuse'))} | "
        + ("— |" if result.get("completed_agents") is None else f"{int(result['completed_agents']):,} / {int(result.get('failures', 0)):,} |"),
        "",
        f"- Direct-only / 完整双路径吞吐比：`{fmt(relative, 3)}×`。",
        "",
        "## 稳态资源",
        "",
        "| 方法 | P Forward/卡 | P KV | P queue | P inflight | D Forward/卡 | D KV | D running | D prealloc | D transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| 完整双路径 | 73.22% | 77.40% | 51.08 | 17.30 | 97.48% | 85.07% | 52.33 | 1.88 | 0.36 |",
        f"| 仅关闭 P→D Host | {pct(result.get('prefill_forward_per_gpu'))} | {pct(result.get('p_kv'))} | {fmt(result.get('p_queue'))} | {fmt(result.get('p_inflight'))} | {pct(result.get('decode_forward_per_gpu'))} | {pct(result.get('d_kv'))} | {fmt(result.get('d_running'))} | {fmt(result.get('d_prealloc'))} | {fmt(result.get('d_transfer'))} |",
        "",
        "## 原始记录",
        "",
        f"- 控制目录：`{CONTROL}`",
        f"- 本轮结果：`{RUN_DIR}`",
        "- 完整方法：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`",
        "",
    ]
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(REPORT)


lowc.case_key = case_key
lowc.run_dir = run_dir
lowc.case_environment = case_environment
lowc.render_report = render_report
base.case_key = case_key
base.run_dir = run_dir
base.case_environment = case_environment
base.validate_inputs = lowc.validate_inputs
base.atomic_json = lowc.atomic_json


if __name__ == "__main__":
    CONTROL.mkdir(parents=True, exist_ok=True)
    existing = json.loads((CONTROL / "state.json").read_text()) if (CONTROL / "state.json").exists() else {}
    render_report(existing)
    base.main()
