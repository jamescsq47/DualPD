#!/usr/bin/env python3
"""Run one Qwen3-30B-A3B BrowseComp c320 P:D-ratio matrix.

Set ``PD_RATIO`` to ``2p2d`` or ``3p1d``.  Each allocation runs the three
methods serially; two allocations can therefore cover the ratio matrix
without sharing GPUs or ports.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PARENT = HERE / "qwen3_30b_a3b_browsecomp_h100_matrix.py"
RATIO = os.environ.get("PD_RATIO", "")
RATIOS = {
    "2p2d": {
        "label": "2P:2D",
        "node": "node2700",
        "p_groups": "0,4;1,5",
        "d_groups": "2,6;3,7",
        "p_ports": "31340 31344",
        "bootstrap_ports": "29340 29344",
        "d_ports": "31342 31343",
        "p_mem": "0.80 0.80",
        "d_mem": "0.80 0.60",
        # Four physical P ranks; totals are 512/256 GiB.
        "d2p_gib_per_rank": "128",
        "p2d_gib_per_rank": "64",
    },
    "3p1d": {
        "label": "3P:1D",
        "node": "node2701",
        "p_groups": "0,4;1,5;2,6",
        "d_groups": "3,7",
        "p_ports": "31340 31344 31345",
        "bootstrap_ports": "29340 29344 29345",
        "d_ports": "31343",
        "p_mem": "0.80 0.80 0.80",
        "d_mem": "0.60",
        # Six physical P ranks; totals are 512/256 GiB.
        "d2p_gib_per_rank": "85.3333333333",
        "p2d_gib_per_rank": "42.6666666667",
    },
}
if RATIO not in RATIOS:
    raise RuntimeError("PD_RATIO must be one of: 2p2d, 3p1d")

spec = importlib.util.spec_from_file_location("qwen30_ratio_parent", PARENT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {PARENT}")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
base = parent.base

MODES = ("new_method", "colocated", "native_mooncake", "no_reverse")
CASES = [("browsecomp32b", mode, 320) for mode in MODES]
CONTROL_ROOT = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
CONTROL = CONTROL_ROOT / f"h100-browsecomp-qwen3-30b-a3b-tp2-c320-{RATIO}-20261021"
CONTROLS = {
    ratio: CONTROL_ROOT / f"h100-browsecomp-qwen3-30b-a3b-tp2-c320-{ratio}-20261021"
    for ratio in RATIOS
}
OLD_CONTROL = CONTROL_ROOT / "h100-browsecomp-qwen3-30b-a3b-tp2-12-matrix-20261012"
REPORT = Path(
    "/home/shangy/dualpd/DualPD/examples/pd/runs-host/"
    "BROWSECOMP_QWEN3_30B_A3B_TP2_C320_PD_RATIO_H100.md"
)
REPORT_LOCK = REPORT.with_suffix(REPORT.suffix + ".lock")
original_case_environment = base.case_environment
original_atomic_json = base.atomic_json


def case_key(case: tuple[str, str, int]) -> str:
    return f"{RATIO}-{case[1]}-c320"


def run_dir(case: tuple[str, str, int]) -> Path:
    return CONTROL_ROOT / (
        "browsecomp-qwen3-30b-a3b-tp2-h100-"
        f"{RATIO}-{case[1].replace('_', '-')}-c320-w300-m1200-20261021-r1"
    )


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = original_case_environment(case, directory)
    mode = case[1]
    layout = RATIOS[RATIO]
    if mode == "colocated":
        # Colocated has no P/D roles.  Keep the same four TP=2 replicas in both
        # ratio-labelled runs so each ratio has an independent control repeat.
        return env, command

    env.update(
        PREFILL_GPU_GROUPS=layout["p_groups"],
        DECODE_GPU_GROUPS=layout["d_groups"],
        PREFILL_PORTS=layout["p_ports"],
        PREFILL_BOOTSTRAP_PORTS=layout["bootstrap_ports"],
        DECODE_PORTS=layout["d_ports"],
        PREFILL_MEM_FRACTION_STATICS=layout["p_mem"],
        DECODE_MEM_FRACTION_STATICS=layout["d_mem"],
    )
    if mode == "new_method":
        # The Agentic launcher names this variable BOOTSTRAP_PORTS.
        env["BOOTSTRAP_PORTS"] = env.pop("PREFILL_BOOTSTRAP_PORTS")
        env.update(
            PD_LATE_BIND_NUMA_DOMAINS="1",
            SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS="1",
            P2D_HOST_ARENA_GIB_PER_P=layout["p2d_gib_per_rank"],
            SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=layout["p2d_gib_per_rank"],
            SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB=layout["d2p_gib_per_rank"],
        )
    return env, command


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}


def get_item(ratio: str, mode: str) -> dict[str, object]:
    if ratio == "1p3d":
        state = read_json(OLD_CONTROL / "state.json")
        item = state.get(f"browsecomp30ba3b-{mode}-c320", {})
    else:
        state = read_json(CONTROLS[ratio] / "state.json")
        item = state.get(f"{ratio}-{mode}-c320", {})
    return item if isinstance(item, dict) else {}


def fmt(value: object, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def pct(value: object) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def status(item: dict[str, object]) -> str:
    return {"complete": "完成", "running": "运行中", "failed": "失败"}.get(
        str(item.get("status")), "待运行"
    )


def result(item: dict[str, object]) -> dict[str, object]:
    value = item.get("result", {})
    return value if isinstance(value, dict) else {}


def render_report() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = sum(
            get_item(ratio, mode).get("status") == "complete"
            for ratio in ("2p2d", "3p1d")
            for mode in MODES
        )
        lines = [
            "# BrowseComp Qwen3-30B-A3B TP=2 c320：P:D 比例消融",
            "",
            f"新增实验进度：`{complete}/8`。2P:2D 固定在 `node2700`，3P:1D 固定在 "
            "`node2701`；两个节点并行、每个节点上的四种方法串行。1P:3D 直接复用 "
            "`BROWSECOMP_QWEN3_30B_A3B_TP2_H100.md` 的 c320 正式结果。",
            "",
            "所有实验保持 BrowseComp source-order n680、temperature=0、seed=2026、"
            "context=40960、单 trajectory 预算 36864、并发 c320、300 秒预热 + "
            "1200 秒稳态测量，以及总计 4 个 TP=2 推理组。Agentic-PD 的工具/Direct "
            "窗口均为 1 秒，D→P Host 总量 512 GiB、P→D Host 总量 256 GiB。",
            "",
            "> Colocated 没有 P/D 角色划分；2P:2D 与 3P:1D 两行是相同四副本配置的独立重复，"
            "用于分别控制两个比例队列的节点/时间波动。",
            "",
            "## 吞吐与工作量",
            "",
            "| 逻辑 P:D | 方法 | 节点 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | Prefill/agent | Decode/agent | 完成/失败 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        method_labels = {
            "new_method": "Agentic-PD（新方法）",
            "colocated": "Colocated",
            "native_mooncake": "原生 HiCache/Mooncake PD",
            "no_reverse": "原生单向 PD（No-reverse）",
        }
        for ratio in ("1p3d", "2p2d", "3p1d"):
            node = "node2801（复用）" if ratio == "1p3d" else RATIOS[ratio]["node"]
            ratio_label = {"1p3d": "1P:3D", "2p2d": "2P:2D", "3p1d": "3P:1D"}[ratio]
            for mode in MODES:
                item = get_item(ratio, mode)
                data = result(item)
                reuse = data.get("parent_kv_reuse")
                reuse_text = "—" if reuse is None else f"{float(reuse) * 100:.2f}%"
                completed = data.get("completed_agents")
                done = "—" if completed is None else f"{int(completed):,}/{int(data.get('failures', 0)):,}"
                lines.append(
                    f"| {ratio_label} | {method_labels[mode]} | {node} | {status(item)} | "
                    f"{fmt(data.get('decode_tps'))} | {fmt(data.get('agent_s'), 3)} | "
                    f"{fmt(data.get('prefill_tps'))} | {reuse_text} | "
                    f"{fmt(data.get('prefill_per_agent'))} | {fmt(data.get('decode_per_agent'))} | {done} |"
                )
        lines += [
            "",
            "## 稳态资源",
            "",
            "| 逻辑 P:D | 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for ratio in ("1p3d", "2p2d", "3p1d"):
            ratio_label = {"1p3d": "1P:3D", "2p2d": "2P:2D", "3p1d": "3P:1D"}[ratio]
            for mode in MODES:
                data = result(get_item(ratio, mode))
                p_kv = "—" if mode == "colocated" else pct(data.get("p_kv"))
                p_queue = "—" if mode == "colocated" else fmt(data.get("p_queue"), 2)
                p_inflight = "—" if mode == "colocated" else fmt(data.get("p_inflight"), 2)
                lines.append(
                    f"| {ratio_label} | {method_labels[mode]} | {pct(data.get('prefill_forward_per_gpu'))} | "
                    f"{p_kv} | {p_queue} | {p_inflight} | {pct(data.get('decode_forward_per_gpu'))} | "
                    f"{pct(data.get('d_kv'))} | {fmt(data.get('d_running'), 2)} | "
                    f"{fmt(data.get('d_queue'), 2)} | {fmt(data.get('d_prealloc'), 2)} | "
                    f"{fmt(data.get('d_transfer'), 2)} |"
                )
        lines += [
            "",
            "## Host 与原始目录",
            "",
            "Agentic-PD 两种新比例都固定为 D→P 512 GiB、P→D 256 GiB 总容量。"
            "原生 HiCache/Mooncake 沿用此前 256 GiB 共享 Host cache。",
            "",
        ]
        for ratio in ("1p3d", "2p2d", "3p1d"):
            for mode in MODES:
                item = get_item(ratio, mode)
                directory = item.get("run_dir", "—")
                lines.append(f"- `{ratio}/{mode}`：`{directory}`")
        temporary = REPORT.with_name(f"{REPORT.name}.{os.getpid()}.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(REPORT)


def atomic_json(path: Path, value: object) -> None:
    original_atomic_json(path, value)
    if path == CONTROL / "state.json":
        render_report()


base.CONTROL = CONTROL
base.CASES = CASES
base.case_key = case_key
base.run_dir = run_dir
base.case_environment = case_environment
base.atomic_json = atomic_json


if __name__ == "__main__":
    render_report()
    base.main()
