#!/usr/bin/env python3
"""Serial BrowseComp Qwen3-30B-A3B TP=2 H100 matrix.

This intentionally reuses the proven Qwen3-32B controller.  The model,
output directories, and report are the only experiment-level overrides.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-30b-a3b-tp2-12-matrix-20261012"
)
BASE_CONTROLLER = Path(__file__).with_name("matrix_controller.py")
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-30B-A3B")
REPORT = Path(
    "/home/shangy/dualpd/DualPD/examples/pd/runs-host/"
    "BROWSECOMP_QWEN3_30B_A3B_TP2_H100.md"
)
CONCURRENCIES = (192, 256, 320)
REPORT_CONCURRENCIES = (128, 192, 256, 320, 384)
MODES = ("new_method", "colocated", "no_reverse", "native_mooncake")
CASES = [("browsecomp32b", mode, concurrency) for mode in MODES for concurrency in CONCURRENCIES]
LABELS = {
    "new_method": "Agentic-PD（当前新方法）",
    "colocated": "Colocated",
    "no_reverse": "原生单向 PD（无回传）",
    "native_mooncake": "原生 HiCache/Mooncake PD",
}


spec = importlib.util.spec_from_file_location("qwen3_30b_a3b_matrix_base", BASE_CONTROLLER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_CONTROLLER}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.CONTROL = CONTROL
base.MODEL_32B = MODEL
base.CASES = CASES
original_common_environment = base.common_environment
original_atomic_json = base.atomic_json


def case_key(case: tuple[str, str, int]) -> str:
    _, mode, concurrency = case
    return f"browsecomp30ba3b-{mode}-c{concurrency}"


def run_dir(case: tuple[str, str, int]) -> Path:
    _, mode, concurrency = case
    return base.RUNS / (
        "browsecomp-qwen3-30b-a3b-tp2-h100-"
        f"{mode.replace('_', '-')}-c{concurrency}-w300-m1200-20261012-r1"
    )


def common_environment(case: tuple[str, str, int], directory: Path) -> dict[str, str]:
    """Keep the Qwen3-32B BrowseComp settings byte-for-byte where possible."""
    env = original_common_environment(case, directory)
    env.update(
        MODEL_PATH=str(MODEL),
        # The cluster GCC is configured for x86_64-pc-linux-gnu and does not
        # automatically search Debian's multiarch directories. Qwen3 MoE
        # triggers a Triton launcher build that the dense 32B model did not.
        CPATH="/usr/include/x86_64-linux-gnu:/usr/include",
        LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu",
    )
    return env


def validate_inputs() -> None:
    required = [
        base.PD_ENV / "bin/python",
        base.BASELINE_ENV / "bin/python",
        base.MODIFIED_SGLANG / "sglang/__init__.py",
        MODEL / "model.safetensors.index.json",
        base.DATA / "browsecomp/bc_train.jsonl",
        base.PD / "configs/workloads/fixed_browsecomp_source_order_n680.json",
        base.PD / "data/browsecomp/artifacts/search/corpus_embeddings.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing inputs:\n" + "\n".join(missing))


def fmt(value: object, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}"


def status_text(item: dict[str, object]) -> str:
    return {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（已停止后续实验）",
    }.get(str(item.get("status")), "待运行")


def result_row(mode: str, concurrency: int, states: dict[str, object]) -> str:
    item = states.get(case_key(("browsecomp32b", mode, concurrency)), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}
    completed = result.get("completed_agents")
    failed = result.get("failures")
    done = "—" if completed is None else f"{int(completed):,} / {int(failed or 0):,}"
    reuse = result.get("parent_kv_reuse")
    reuse_text = "—" if reuse is None else f"{float(reuse) * 100:.2f}%"
    return (
        f"| {LABELS[mode]} | {concurrency} | {status_text(item)} | "
        f"{fmt(result.get('decode_tps'))} | {fmt(result.get('agent_s'), 3)} | "
        f"{fmt(result.get('prefill_tps'))} | {reuse_text} | {done} |"
    )


def resource_row(mode: str, concurrency: int, states: dict[str, object]) -> str:
    item = states.get(case_key(("browsecomp32b", mode, concurrency)), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}

    def pct(key: str) -> str:
        value = result.get(key)
        return "—" if value is None else f"{float(value) * 100:.1f}%"

    p_kv = "—" if mode == "colocated" else pct("p_kv")
    p_queue = "—" if mode == "colocated" else fmt(result.get("p_queue"), 2)
    p_inflight = "—" if mode == "colocated" else fmt(result.get("p_inflight"), 2)
    return (
        f"| {LABELS[mode]} | {concurrency} | {pct('prefill_forward_per_gpu')} | "
        f"{p_kv} | {p_queue} | {p_inflight} | {pct('decode_forward_per_gpu')} | "
        f"{pct('d_kv')} | {fmt(result.get('d_running'), 2)} | "
        f"{fmt(result.get('d_queue'), 2)} | {fmt(result.get('d_prealloc'), 2)} | "
        f"{fmt(result.get('d_transfer'), 2)} |"
    )


def mooncake_row(concurrency: int, states: dict[str, object]) -> str:
    item = states.get(case_key(("browsecomp32b", "native_mooncake", concurrency)), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}
    mooncake = result.get("mooncake", {}) if isinstance(result, dict) else {}
    fraction = mooncake.get("resident_fraction")
    return (
        f"| {concurrency} | {fmt(mooncake.get('capacity_gb'))} GiB | "
        f"{fmt(mooncake.get('resident_gb'), 2)} GiB | "
        f"{'—' if fraction is None else f'{float(fraction) * 100:.2f}%'} | "
        f"{fmt(mooncake.get('evicted_gb'), 2)} GiB | "
        f"{fmt(mooncake.get('evicted_keys'), 0)} | "
        f"{fmt(mooncake.get('allocation_failures'), 0)} |"
    )


def render_report(states: dict[str, object]) -> None:
    complete = sum(
        isinstance(item, dict) and item.get("status") == "complete"
        for item in states.values()
    )
    lines = [
        "# BrowseComp + Qwen3-30B-A3B TP=2（单节点 H100）",
        "",
        "测试节点：c192/c256/c320 使用 `node2801`，c128/c384 使用 `node2800`；",
        "两者均为 8×H100 80GB。此矩阵仅把原",
        "`BROWSECOMP_QWEN3_32B_TP2_H100.md` 的模型替换为 `Qwen3-30B-A3B`，",
        "其余配置保持一致。当前进度：`%d/20`。" % complete,
        "",
        "BrowseComp source-order n680，temperature=0、seed=2026、context=40960、",
        "trajectory budget=36864；每组 300 秒业务预热 + 1200 秒正式测量，",
        "并发 c128/c192/c256/c320/c384。实验严格按表格顺序串行执行，任一组非可恢复失败时",
        "控制器 fail-closed，不会静默跳到下一组。",
        "",
        "PD 方法保持 2P:6D 物理布局：TP=2 Prefill `[0,4]`，TP=2 Decode",
        "`[1,5]`、`[2,6]`、`[3,7]`。P 显存比例 0.80；D 为 0.80/0.80/0.60；",
        "GPU7 同时承载搜索服务。Agentic-PD 使用 1 秒工具窗口、1 秒 Direct 窗口、",
        "Q high/low=32/8，D→P Host 总量 512 GiB、P→D Host 总量 256 GiB。",
        "",
        "## 正式结果",
        "",
        "| 方法 | 并发 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(result_row(mode, c, states) for mode in MODES for c in REPORT_CONCURRENCIES)
    lines.extend([
        "",
        "## 稳态资源",
        "",
        "| 方法 | 并发 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(resource_row(mode, c, states) for mode in MODES for c in REPORT_CONCURRENCIES)
    lines.extend([
        "",
        "资源指标均按正式 1,200 秒窗口统计；Forward 按物理 GPU，KV 与队列按",
        "逻辑 TP=2 引擎。Colocated 的 P/D 共享 KV pool，因此独立 P 指标留空。",
        "",
        "## 原生 Mooncake Host 状态",
        "",
        "| 并发 | 共享容量 | 窗口末驻留 | 驻留率 | 累计淘汰 | 淘汰 key | 分配失败 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(mooncake_row(c, states) for c in REPORT_CONCURRENCIES)
    lines.extend(["", "## 原始结果目录", ""])
    for mode in MODES:
        for concurrency in REPORT_CONCURRENCIES:
            case = ("browsecomp32b", mode, concurrency)
            item = states.get(case_key(case), {})
            directory = item.get("run_dir") if isinstance(item, dict) else None
            lines.append(f"- `{case_key(case)}`：`{directory or run_dir(case)}`")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(REPORT)


def atomic_json(path: Path, value: object) -> None:
    original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        render_report(value)


base.case_key = case_key
base.run_dir = run_dir
base.common_environment = common_environment
base.validate_inputs = validate_inputs
base.atomic_json = atomic_json


if __name__ == "__main__":
    existing = json.loads((CONTROL / "state.json").read_text()) if (CONTROL / "state.json").exists() else {}
    render_report(existing)
    base.main()
