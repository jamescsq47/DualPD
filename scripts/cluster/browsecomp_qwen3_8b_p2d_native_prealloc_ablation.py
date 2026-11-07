#!/usr/bin/env python3
"""Unbounded c576 P->D ablation with strict D-reserved-before-P semantics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / "."
SOURCE = ROOT / "scripts/cluster/browsecomp_qwen3_8b_p2d_prealloc_ablation.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-p2d-dfirst-recovery-after-d-c576-20261028-r9"
)
RUN_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-dfirst-recovery-after-d-direct-c576-"
    "w300-m1200-20261028-r9"
)
FAILED_R8_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-dfirst-queued-direct-c576-"
    "w300-m1200-20261028-r8"
)
FAILED_R7_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-dfirst-targeted-direct-c576-"
    "w300-m1200-20261028-r7"
)
INVALID_R6_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-dfirst-unbounded-direct-c576-"
    "w300-m1200-20261028-r6"
)
PREVIOUS_R5_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-settled-prealloc4-direct-c576-"
    "w300-m1200-20261028-r5"
)
MISPLACED_R4_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-settled-prealloc4-direct-c576-"
    "w300-m1200-20261028-r4"
)
BOUNDED8_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-bounded-prealloc8-direct-c576-"
    "w300-m1200-20261028-r3"
)
UNBOUNDED_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-native-prealloc-direct-c576-"
    "w300-m1200-20261027-r2"
)
INVALID_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-prealloc-direct-c576-"
    "w300-m1200-20261027-r1"
)
REPORT = PD / "runs-host/BROWSECOMP_QWEN3_8B_H100_P2D_PREALLOC_ABLATION.md"
CASE = ("browsecomp8b4p4d", "p2d_settled_prealloc_direct", 576)


spec = importlib.util.spec_from_file_location("p2d_native_base", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
base = legacy.base

# The legacy module contains the proven launch, result extraction, and cleanup
# gate.  Redirect its globals to a fresh run so the invalid experiment remains
# immutable and auditable.
legacy.CONTROL = CONTROL
legacy.RUN_DIR = RUN_DIR
legacy.REPORT = REPORT
legacy.CASE = CASE
legacy.CASES = [CASE]
base.CONTROL = CONTROL
base.CASES = [CASE]


def case_key(_case: tuple[str, str, int]) -> str:
    return "browsecomp8b-4p4d-p2d-dfirst-recovery-after-d-direct-c576-r9"


def run_dir(_case: tuple[str, str, int]) -> Path:
    return RUN_DIR


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = legacy.lowc.modified_environment(
        (case[0], "new_method", case[2]), directory
    )
    env.update(
        SGLANG_PD_ABLATION_P2D_PREBIND="false",
        SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC="true",
        SGLANG_PD_ABLATION_P2D_ROUTE_SETTLED="true",
        P2D_HOST_STAGING="false",
        SGLANG_AGENTIC_KV_P2D_HOST_STAGING="false",
        # Zero is upstream/unbounded semantics: physical D request slots, KV
        # capacity and decode safety headroom are the only admission limits.
        SGLANG_PD_MAX_TRANSFER_INFLIGHT="0",
        PD_MAX_TRANSFER_INFLIGHT="0",
        SGLANG_PD_P_READY_BACKPRESSURE_MODE="disabled",
        ABLATION_NAME="p2d-dfirst-unbounded-direct-only",
    )
    return env, command


def render_report(states: dict[str, Any]) -> None:
    full = legacy.normalized(legacy.read_result(legacy.FULL_DIR))
    full.update(legacy.FULL_RESOURCE)
    colocated = legacy.normalized(legacy.read_result(legacy.COLOCATED_DIR))
    colocated.update(legacy.COLOCATED_RESOURCE)
    invalid = legacy.normalized(legacy.read_result(INVALID_DIR))
    unbounded = legacy.normalized(legacy.read_result(UNBOUNDED_DIR))
    bounded8 = legacy.normalized(legacy.read_result(BOUNDED8_DIR))
    misplaced_r4 = legacy.normalized(legacy.read_result(MISPLACED_R4_DIR))
    previous_r5 = legacy.normalized(legacy.read_result(PREVIOUS_R5_DIR))
    invalid_r6 = legacy.normalized(legacy.read_result(INVALID_R6_DIR))
    failed_r7 = legacy.normalized(legacy.read_result(FAILED_R7_DIR))
    failed_r8 = legacy.normalized(legacy.read_result(FAILED_R8_DIR))
    item = states.get(case_key(CASE), {})
    raw_corrected = item.get("result", {}) if isinstance(item, dict) else {}
    corrected = legacy.normalized(raw_corrected)
    corrected["tail_completion_stall"] = bool(
        raw_corrected.get("tail_completion_stall")
        if isinstance(raw_corrected, dict)
        else False
    )
    corrected["tail_decode_tokens"] = (
        raw_corrected.get("tail_120s_decode_tokens")
        if isinstance(raw_corrected, dict)
        else None
    )
    status = {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（控制器将恢复）",
    }.get(item.get("status") if isinstance(item, dict) else None, "排队/待运行")
    if corrected.get("tail_completion_stall"):
        status = "完成（尾段饥饿）"
    elif (
        status == "完成"
        and corrected.get("tail_decode_tokens") is not None
        and float(corrected["tail_decode_tokens"]) < 1000
    ):
        status = "完成（尾段近停滞）"

    def row(label: str, status_text: str, result: dict[str, Any]) -> str:
        completed = result.get("completed")
        return (
            f"| {label} | {status_text} | {legacy.number(result.get('decode_tps'))} | "
            f"{legacy.number(result.get('agent_s'), 4)} | "
            f"{legacy.number(result.get('prefill_tps'))} | "
            f"{legacy.percent(result.get('reuse'))} | "
            f"{'—' if completed is None else f'{int(completed):,}'} |"
        )

    full_tps = full.get("decode_tps")
    colocated_tps = colocated.get("decode_tps")
    corrected_tps = corrected.get("decode_tps")
    corrected_speedup = (
        float(corrected_tps) / float(colocated_tps)
        if corrected_tps is not None and colocated_tps
        else None
    )
    dual_gain = (
        float(full_tps) / float(corrected_tps) - 1.0
        if full_tps is not None and corrected_tps
        else None
    )
    lines = [
        "# BrowseComp Qwen3-8B H100：P→D 双路径消融",
        "",
        "> 配置：单节点 H100×8，TP=1，4P:4D，c576，300 秒业务预热 + 1200 秒稳态测量。",
        "> 正式消融先让 D 实际占用 request slot 和物理目标 KV；随后才向选定 P 发布 targeted arrival，完成既有 D→P Direct/Host 恢复并提交 P forward。P→D 只走 Direct。",
        "> 不设置请求数或 transfer inflight 上限（两个上限参数均为 0）；仅由 D 原生 KV allocator、request slots 与 decode safety headroom 限制。",
        "> 旧的 Router 中心预留实验改变了原生调度语义并在高并发下停滞，只保留审计，不作为性能结论。",
        "",
        "## 核心结果",
        "",
        "| 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成数 |",
        "|---|---|---:|---:|---:|---:|---:|",
        row("Colocated baseline（复用）", "完成", colocated),
        row("完整 P→D Direct/Host 双路径（复用）", "完成", full),
        row("D 实际预留后放行 P + Direct-only（无人工上限）", status, corrected),
        row("r8 P 提前 bootstrap、恢复早于 D 预留（闭环，无效）", "无效", failed_r8),
        row("r7 D 预留后才提交 P（超时，无效）", "无效", failed_r7),
        row("r6 targeted-arrival 顺序错误（无效）", "无效", invalid_r6),
        row("旧 4/D 路由稳定运行（诊断）", "完成（尾段近停滞）", previous_r5),
        row("4/D 修复未生效运行（诊断）", "无效", misplaced_r4),
        row("有界原生 prealloc + Direct-only（8/D，诊断）", "失败（重定向残留）", bounded8),
        row("无界原生 prealloc + Direct-only（诊断）", "完成（资源闭环拥堵）", unbounded),
        row("旧中心预留实现（无效，仅审计）", "无效", invalid),
        "",
        "## 正式对比",
        "",
        f"- 完整双路径相对 Colocated：`{legacy.number(float(full_tps) / float(colocated_tps), 3) if full_tps and colocated_tps else '—'}×`。",
        f"- D-first 无界消融相对 Colocated：`{legacy.number(corrected_speedup, 3) if corrected_speedup is not None else '—'}×`。",
        f"- 完整双路径相对 D-first 无界消融的吞吐提升：`{legacy.percent(dual_gain) if dual_gain is not None else '—'}`（即 `{legacy.number(float(full_tps) / float(corrected_tps), 3) if full_tps and corrected_tps else '—'}×`）。",
        f"- 完整窗口数据可复核，但正式消融最后 120 秒仅增长 `{legacy.number(corrected.get('tail_decode_tokens'), 0)}` 个 Decode tokens，属于尾段近停滞；该数值不能当作健康原生 SGLang baseline。" if corrected.get("tail_decode_tokens") is not None and float(corrected["tail_decode_tokens"]) < 1000 else "",
        "- 有界原生消融完成了完整测量窗口，但最后 120 秒无请求完成 token 增长；吞吐按完整 1200 秒窗口统计。" if corrected.get("tail_completion_stall") else "",
        "- 8/D 诊断运行虽限制了 transfer，但动态 P 重定向仍触发大量 receiver abort，不能代表合理的经典预分配实现。",
        "- 无界原生窗口的诊断值为 403.7 decode token/s；其 P/D HBM 被在途 transfer 同时钉住，不能代表合理的经典预分配实现。",
        "",
        "## 稳态资源细化",
        "",
        "| 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, result in (
        ("Colocated baseline", colocated),
        ("完整双路径", full),
        ("D 实际预留后放行 P + Direct-only（无人工上限）", corrected),
        ("r8 P 提前 bootstrap、恢复早于 D 预留（闭环，无效）", failed_r8),
        ("r7 D 预留后才提交 P（超时，无效）", failed_r7),
        ("r6 targeted-arrival 顺序错误（无效）", invalid_r6),
        ("旧 4/D 路由稳定运行", previous_r5),
        ("4/D 修复未生效运行", misplaced_r4),
        ("有界原生 prealloc + Direct-only（8/D）", bounded8),
        ("无界原生 prealloc + Direct-only", unbounded),
    ):
        lines.append(
            f"| {label} | {legacy.percent(result.get('p_forward'))} | "
            f"{legacy.percent(result.get('p_kv'))} | {legacy.number(result.get('p_queue'), 2)} | "
            f"{legacy.number(result.get('p_inflight'), 2)} | {legacy.percent(result.get('d_forward'))} | "
            f"{legacy.percent(result.get('d_kv'))} | {legacy.number(result.get('d_running'), 2)} | "
            f"{legacy.number(result.get('d_queue'), 2)} | {legacy.number(result.get('d_prealloc'), 2)} | "
            f"{legacy.number(result.get('d_transfer'), 2)} |"
        )
    lines.extend(
        [
            "",
            "## 固定配置与原始记录",
            "",
            "- GPU 映射：P=`0,2,4,6`，D=`1,3,5,7`；各引擎 TP=1。",
            "- Host 容量：D→P 每个 P 128 GiB（总 512 GiB）；正式消融禁用 P→D Host。",
            "- 调度语义：Router 先 POST D，等待 D worker 完成真实物理 KV 预留并写出逐请求证明；随后才向目标 P 发布 targeted arrival，等待 D→P 恢复确定后提交 P forward。",
            "- 上限语义：`SGLANG_PD_MAX_TRANSFER_INFLIGHT=0`、`PD_MAX_TRANSFER_INFLIGHT=0`，无人工请求数上限。",
            f"- Colocated：`{legacy.COLOCATED_DIR}`",
            f"- 完整双路径：`{legacy.FULL_DIR}`",
            f"- 正式 D-first 无界消融：`{RUN_DIR}`",
            f"- r8 恢复早于 D 预留的资源闭环运行：`{FAILED_R8_DIR}`",
            f"- r7 预留后才提交 P 的超时运行：`{FAILED_R7_DIR}`",
            f"- r6 顺序错误无效运行：`{INVALID_R6_DIR}`",
            f"- 旧 4/D 路由稳定诊断：`{PREVIOUS_R5_DIR}`",
            f"- r4 修复未生效诊断：`{MISPLACED_R4_DIR}`",
            f"- 8/D 重定向残留诊断：`{BOUNDED8_DIR}`",
            f"- 无界原生诊断：`{UNBOUNDED_DIR}`",
            f"- 无效旧实现：`{INVALID_DIR}`",
            f"- 控制状态：`{CONTROL}`",
            "",
        ]
    )
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(REPORT)


def atomic_json(path: Path, value: object) -> None:
    if path == CONTROL / "state.json" and isinstance(value, dict):
        legacy.cleanup_gate(value)
    legacy.lowc.original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        render_report(value)


base.case_key = case_key
base.run_dir = run_dir
base.case_environment = case_environment
base.validate_inputs = legacy.validate_inputs
base.atomic_json = atomic_json
legacy.case_key = case_key
legacy.run_dir = run_dir
legacy.case_environment = case_environment


if __name__ == "__main__":
    existing = (
        json.loads((CONTROL / "state.json").read_text())
        if (CONTROL / "state.json").exists()
        else {}
    )
    render_report(existing)
    base.main()
