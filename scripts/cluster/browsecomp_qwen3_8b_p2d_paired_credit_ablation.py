#!/usr/bin/env python3
"""c576 native P->D preallocation paired with capacity-backed P credit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / "."
SOURCE = ROOT / "scripts/cluster/browsecomp_qwen3_8b_p2d_native_prealloc_ablation.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-paired-credit-c576-20261028-r10"
)
RUN_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-paired-credit-direct-c576-"
    "w300-m1200-20261028-r10"
)
REPORT = PD / "runs-host/BROWSECOMP_QWEN3_8B_H100_P2D_PAIRED_CREDIT_ABLATION.md"
CASE = ("browsecomp8b4p4d", "p2d_paired_credit_direct", 576)


spec = importlib.util.spec_from_file_location("p2d_paired_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)
base = prior.base
metrics = prior.legacy
PREVIOUS_R9_DIR = prior.RUN_DIR
PRIOR_CASE_ENVIRONMENT = prior.case_environment

prior.CONTROL = CONTROL
prior.RUN_DIR = RUN_DIR
prior.REPORT = REPORT
prior.CASE = CASE
prior.CASES = [CASE]
metrics.CONTROL = CONTROL
metrics.CASE = CASE
metrics.CASES = [CASE]
base.CONTROL = CONTROL
base.CASES = [CASE]


def case_key(_case: tuple[str, str, int]) -> str:
    return "browsecomp8b-4p4d-p2d-paired-credit-direct-c576-r10"


def run_dir(_case: tuple[str, str, int]) -> Path:
    return RUN_DIR


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = PRIOR_CASE_ENVIRONMENT(case, directory)
    env.update(
        SGLANG_PD_ABLATION_P2D_PAIRED_CREDIT="true",
        ABLATION_NAME="p2d-paired-credit-native-prealloc-direct-only",
    )
    return env, command


def _row(label: str, status: str, result: dict[str, Any]) -> str:
    completed = result.get("completed")
    return (
        f"| {label} | {status} | {metrics.number(result.get('decode_tps'))} | "
        f"{metrics.number(result.get('agent_s'), 4)} | "
        f"{metrics.number(result.get('prefill_tps'))} | "
        f"{metrics.percent(result.get('reuse'))} | "
        f"{'—' if completed is None else f'{int(completed):,}'} |"
    )


def render_report(states: dict[str, Any]) -> None:
    full = metrics.normalized(metrics.read_result(metrics.FULL_DIR))
    full.update(metrics.FULL_RESOURCE)
    colocated = metrics.normalized(metrics.read_result(metrics.COLOCATED_DIR))
    colocated.update(metrics.COLOCATED_RESOURCE)
    previous = metrics.normalized(metrics.read_result(PREVIOUS_R9_DIR))
    item = states.get(case_key(CASE), {})
    paired = metrics.normalized(
        item.get("result", {}) if isinstance(item, dict) else {}
    )
    status = {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（watchdog 恢复）",
    }.get(item.get("status") if isinstance(item, dict) else None, "排队/待运行")
    lines = [
        "# BrowseComp Qwen3-8B H100：配对 P-credit 的原生 P→D 预留消融",
        "",
        "> 单节点 H100×8，TP=1，4P:4D，c576；300 秒业务预热 + 1200 秒稳态。",
        "> 保留 D 原生无界 P→D preallocation；每个请求只有在取得可容纳完整工作集的 P 逻辑 credit 后才提交其 D 预留。",
        "> parent arrival 立即通知旧 D：1 秒内目标 D 与 P credit 配对成功可走 Direct，否则先 D→Host 释放旧 D；目标 D 预留完成后才 Host→P、P Forward、P→D。",
        "",
        "## 核心结果",
        "",
        "| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成数 |",
        "|---|---|---:|---:|---:|---:|---:|",
        _row("Colocated baseline（复用）", "完成", colocated),
        _row("完整 P→D Direct/Host 双路径（复用）", "完成", full),
        _row("r9 D-first、无容量 P credit（诊断）", "完成但尾段饥饿", previous),
        _row("r10 配对 P credit + 原生 D preallocation", status, paired),
        "",
        "## 稳态资源细化",
        "",
        "| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in (
        ("完整双路径", full),
        ("r9 无容量 credit", previous),
        ("r10 配对 P credit", paired),
    ):
        lines.append(
            f"| {label} | {metrics.percent(result.get('p_forward'))} | "
            f"{metrics.percent(result.get('p_kv'))} | {metrics.number(result.get('p_queue'), 2)} | "
            f"{metrics.number(result.get('p_inflight'), 2)} | {metrics.percent(result.get('d_forward'))} | "
            f"{metrics.percent(result.get('d_kv'))} | {metrics.number(result.get('d_running'), 2)} | "
            f"{metrics.number(result.get('d_queue'), 2)} | {metrics.number(result.get('d_prealloc'), 2)} | "
            f"{metrics.number(result.get('d_transfer'), 2)} |"
        )
    lines.extend([
        "",
        "## 配置与原始记录",
        "",
        "- GPU：P=`0,2,4,6`，D=`1,3,5,7`；各 TP=1。",
        "- D→P Host：每个 P 128 GiB，总 512 GiB；P→D Host 禁用。",
        "- D preallocation 不设置 request/transfer 人工上限；P credit 由实时 HBM used + Router 未提交 workset 原子记账。",
        f"- 完整方法：`{metrics.FULL_DIR}`",
        f"- r9：`{PREVIOUS_R9_DIR}`",
        f"- r10：`{RUN_DIR}`",
        f"- 控制目录：`{CONTROL}`",
        "",
    ])
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(REPORT)


def atomic_json(path: Path, value: object) -> None:
    if path == CONTROL / "state.json" and isinstance(value, dict):
        metrics.cleanup_gate(value)
    metrics.lowc.original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        render_report(value)


base.case_key = case_key
base.run_dir = run_dir
base.case_environment = case_environment
base.validate_inputs = metrics.validate_inputs
base.atomic_json = atomic_json
prior.case_key = case_key
prior.run_dir = run_dir
prior.case_environment = case_environment


if __name__ == "__main__":
    existing = (
        json.loads((CONTROL / "state.json").read_text())
        if (CONTROL / "state.json").exists()
        else {}
    )
    render_report(existing)
    base.main()
