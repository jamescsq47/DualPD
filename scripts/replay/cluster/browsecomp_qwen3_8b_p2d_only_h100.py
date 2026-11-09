#!/usr/bin/env python3
"""H100 c576 direction ablation: change P->D only, keep D->P recompute.

The No-reverse baseline uses native P->D preallocation and recomputes every
later-turn Prefill.  This case keeps the latter behavior while replacing only
P->D with the full method's late-binding Direct/Host dual path.
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
    "h100-browsecomp-qwen3-8b-p2d-only-c576-20261030-r1"
)
RUN_DIR = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "browsecomp-qwen3-8b-h100-p2d-only-c576-w300-m1200-20261030-r1"
)
REPORT = PD / "runs-host/BROWSECOMP_QWEN3_8B_H100_DIRECTION_ABLATION.md"
CASE = ("browsecomp8b4p4d", "new_method", 576)
CASES = [CASE]


spec = importlib.util.spec_from_file_location("p2d_only_base", SOURCE)
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
    return "browsecomp8b-4p4d-p2d-only-c576"


def run_dir(_case: tuple[str, str, int]) -> Path:
    return RUN_DIR


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = lowc.modified_environment(case, directory)
    env.update(
        # The one functional direction change relative to No-reverse PD:
        # retain late binding plus P->D Direct/Host staging.
        P2D_HOST_STAGING="true",
        SGLANG_AGENTIC_KV_P2D_HOST_STAGING="true",
        SGLANG_PD_ABLATION_P2D_PREBIND="false",
        SGLANG_PD_ABLATION_P2D_NATIVE_PREALLOC="false",
        SGLANG_PD_ABLATION_P2D_ROUTE_SETTLED="false",
        SGLANG_PD_ABLATION_P2D_PAIRED_CREDIT="false",
        # Reserve half of each D's native transfer window for durable Host
        # recovery.  Router may use Direct only while the other four slots
        # are genuinely admissible; otherwise the P-ready generation spills
        # instead of pinning P HBM behind D's prealloc queue.
        SGLANG_PD_LATE_BIND_MAX_DIRECT_TRANSFER_INFLIGHT="4",
        # Keep the baseline D->P semantics: no parent generation is exported,
        # no reverse Direct/Host state is created, and later turns recompute.
        SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE="true",
        ABLATION_NAME="p2d-late-binding-dual-path-only",
    )
    return env, command


def fmt(value: object, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def pct(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.2f}%"


def render_report(states: dict[str, Any]) -> None:
    item = states.get(case_key(CASE), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}
    status = {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（等待 watchdog 修复）",
    }.get(str(item.get("status")), "排队/待运行")
    completed = result.get("completed_agents")
    completion = (
        "—"
        if completed is None
        else f"{int(completed):,} / {int(result.get('failures', 0)):,}"
    )
    lines = [
        "# BrowseComp Qwen3-8B H100：D→P / P→D 方向消融",
        "",
        "> 单节点 H100×8，TP=1，4P:4D，c576；固定 source-order n680，",
        "> 300 秒业务预热 + 1200 秒稳态测量。新实验只改变 P→D：从原生",
        "> preallocation 换为 late-binding Direct/Host 双路径；D→P 仍完全重算。",
        "",
        "## 2×2 核心结果",
        "",
        "| D→P parent 恢复 | P→D 路径 | 角色 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
        "| 原生（无反向恢复） | 原生 preallocate | Baseline：都不改 | 完成 | 5,297.8 | 2.6199 | 113,299.2 | 0.00% | 3,144 / 0 |",
        "| Direct/Host | 原生 preallocate + paired P credit | 仅改 D→P | 完成 | 4,601.7 | 2.2842 | 95,951.8 | 23.55% | 2,741 / 0 |",
        f"| 原生（无反向恢复） | late-binding Direct/Host | **仅改 P→D（本轮）** | {status} | {fmt(result.get('decode_tps'))} | {fmt(result.get('agent_s'), 4)} | {fmt(result.get('prefill_tps'))} | {pct(result.get('parent_kv_reuse'))} | {completion} |",
        "| Direct/Host | late-binding Direct/Host | 两边都改 | 完成 | 10,778.5 | 5.2750 | 79,293.4 | 95.52% | 6,330 / 0 |",
        "",
        "## 2x2 稳态资源与 KV cache hit rate",
        "",
        "| D→P parent 恢复 | P→D late-binding | KV cache hit rate | Direct-return fraction | D KV | D running | D prealloc | D transfer | D 侧未就绪合计 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        "| 关 | 关 | 0.00% | N/A | 91.43% | 14.28 | 112.46 | 17.06 | 129.52 |",
        "| 开 | 关 | 23.55% | 31.19% | 71.38% | 11.44 | 23.66 | 9.49 | 33.15 |",
        f"| 关 | 开 | {pct(result.get('parent_kv_reuse'))} | N/A | {pct(result.get('d_kv'))} | {fmt(result.get('d_running'), 2)} | {fmt(result.get('d_prealloc'), 2)} | {fmt(result.get('d_transfer'), 2)} | {fmt(float(result.get('d_prealloc', 0)) + float(result.get('d_transfer', 0)), 2)} |",
        "| 开 | 开 | 95.52% | 78.91% | 85.07% | 52.33 | 1.88 | 0.36 | 2.24 |",
        "",
        "> `KV cache hit rate` 采用 completion-weighted、page-aligned parent-prefix reuse 口径。",
        "> `D prealloc` 和 `D transfer` 是每个 D 实例的稳态时间平均请求数；",
        "> 两者之和表示已进入 D 侧但尚不能 Decode 的请求数。仅 P→D 开启时的",
        "> 5.23% 为本地/公共 Radix parent-prefix 命中，不是 D→P 反向恢复。",
        "> `Direct-return fraction` 只在启用 D→P 时定义：在稳态窗口内对唯一 snapshot",
        "> 去重，以 fast-arrival 且已终结的回传结果为分母。仅 D→P 为",
        "> 2,290 Direct / 7,342 outcomes = 31.19%；两边都开为",
        "> 12,072 Direct / 15,298 outcomes = 78.91%。",
        "",
        "### 证据边界",
        "",
        "该消融直接证明了两个方向的协同收益，日志也显示仅 D→P 配置与更低的",
        "Direct-return fraction 相关。但这些数据尚不能将差异唯一归因于额外的 P/D",
        "admission 延迟，因为配置同时改变了请求长度分布、队列驻留时间和 HBM 状态。",
        "同样，仅 D→P 将 D 侧未就绪请求从 129.52 降至 33.15，但 D running",
        "也从 14.28 降至 11.44；因此 prealloc/transfer 请求数只表示调度状态深度，",
        "不能单独代表 HBM 压力或解释全部吞吐差异。",
        "",
        "### 仅 P→D late-binding 的稳态资源细节",
        "",
        "| P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D prealloc | D transfer |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {pct(result.get('prefill_forward_per_gpu'))} | {pct(result.get('p_kv'))} | {fmt(result.get('p_queue'), 2)} | {fmt(result.get('p_inflight'), 2)} | {pct(result.get('decode_forward_per_gpu'))} | {pct(result.get('d_kv'))} | {fmt(result.get('d_running'), 2)} | {fmt(result.get('d_prealloc'), 2)} | {fmt(result.get('d_transfer'), 2)} |",
        "",
        "## 语义门禁",
        "",
        "- `SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE=true`：后续轮不携带 parent generation，D→P Direct/Host 均不可达。",
        "- `SGLANG_PD_ABLATION_P2D_* = false`：不启用 prebind/native-prealloc/paired-credit 消融。",
        "- `P2D_HOST_STAGING=true`：保留完整方法的 P→D Direct/Host 双路径与 late binding。",
        "- `SGLANG_PD_LATE_BIND_MAX_DIRECT_TRANSFER_INFLIGHT=4`：Router 只在 D 的 Direct admission window 确实有空位时绑定；D 的原生总 transfer 上限仍为 8，剩余窗口供已经持久化的 P→D Host 恢复。",
        "- 其余模型、拓扑、显存比例、Host 容量、lane、chunk、数据顺序和测量窗口与完整 c576 方法一致。",
        "",
        "## 修复与运行审计",
        "",
        "- r1 的 151.2 Decode token/s 是无效诊断结果：Router 在 D 已 `transfer=8/8` 时仍继续绑定 Direct，导致 P KV 长期被卡住。r3 已修复该 admission 不一致。",
        "- r3 的 P→D Host 事件闭合：offer 774、bind 774、D2H complete 774、H2D complete 774；没有遗留 Host snapshot。",
        "- r3 的 Direct/控制异常均为 0：NIXL sender/receiver exception 0、waiting timeout 0、reservation timeout 0、OOM 0、Traceback 0。",
        "- D 日志观察到的最大 transfer 为 3，未越过 Router Direct 上限 4；稳态 D prealloc 0.07、D transfer 0.09。",
        "- `Parent KV 复用=5.23%` 是 D→P 关闭后的本地/公共 Radix 前缀命中，不代表反向恢复；分析器记录的 D→P reverse reuse lower bound 仅 0.02%。",
        "- 完成后的 node2700 清理门禁：GPU 进程 0、实验端口 0、Apptainer/FUSE helper 0、实验服务进程 0；7 天 Slurm allocation 由 holder 继续保留。",
        "",
        "## 原始记录",
        "",
        f"- 控制目录：`{CONTROL}`",
        f"- 本轮结果：`{RUN_DIR}`",
        "- Baseline：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-no-reverse-c576-w300-m1200-20261006-r1`",
        "- 仅 D→P：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-p2d-paired-credit-direct-c576-w300-m1200-20261028-r10`",
        "- 两边都改：`/orcd/scratch/orcd/003/shangy/dualpd/runs/browsecomp-qwen3-8b-h100-new-method-c576-w300-m1200-20261006-r1`",
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
    state_path = CONTROL / "state.json"
    existing = json.loads(state_path.read_text()) if state_path.exists() else {}
    render_report(existing)
    base.main()
