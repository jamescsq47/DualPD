#!/usr/bin/env python3
"""Qwen3-8B BrowseComp proportional P:D scaling ablation on one H100 node.

The experiment keeps the per-lane load fixed while scaling 4P:4D/c512 to
2P:2D/c256 and 1P:1D/c128.  It reuses the controller and launch paths that
completed the earlier Qwen3-8B H100 matrix; no SGLang source is changed here.
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
PD = ROOT / "examples/pd"
LOWC = ROOT / "tools/dualpd/browsecomp_qwen3_8b_lowc_h100_matrix.py"
CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-pd-scale-12-matrix-20261021"
)
REPORT = PD / "runs-host/BROWSECOMP_QWEN3_8B_H100_PD_SCALE_ABLATION.md"
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-8B")

MODES = ("new_method", "colocated", "no_reverse", "native_mooncake")
LABELS = {
    "new_method": "当前新方法",
    "colocated": "Colocated",
    "no_reverse": "No-reverse PD",
    "native_mooncake": "原生 HiCache/Mooncake",
}

# Placement keeps every P/D pair within a NUMA domain.  SEARCH_GPU remains the
# last Decode/Colocated GPU, matching the completed 4P:4D matrix's sharing rule.
TOPOLOGIES: dict[str, dict[str, Any]] = {
    "4p4d": {
        "label": "4P:4D", "concurrency": 512,
        "p_gpus": "0 2 4 6", "d_gpus": "1 3 5 7",
        "p_ports": "27300 27320 27400 27420",
        "bootstrap_ports": "28300 28320 28400 28420",
        "d_ports": "27301 27321 27401 27421",
        "direct_ports": "61000 61001 61002 61003",
        "p_mem": "0.80 0.80 0.80 0.80",
        "d_mem": "0.80 0.80 0.80 0.60",
        "model_gpus": "0 1 2 3 4 5 6 7",
        "model_ports": "27200 27201 27202 27203 27204 27205 27206 27207",
        "model_mem": "0.80 0.80 0.80 0.80 0.80 0.80 0.80 0.60",
        "search_gpu": "7", "d2p_per_p": "128", "p2d_per_p": "64",
        "expected_gpu_processes": 8,
    },
    "2p2d": {
        "label": "2P:2D", "concurrency": 256,
        "p_gpus": "0 4", "d_gpus": "1 5",
        "p_ports": "27300 27400", "bootstrap_ports": "28300 28400",
        "d_ports": "27301 27401", "direct_ports": "61000 61001",
        "p_mem": "0.80 0.80", "d_mem": "0.80 0.60",
        "model_gpus": "0 1 4 5", "model_ports": "27200 27201 27202 27203",
        "model_mem": "0.80 0.80 0.80 0.60",
        "search_gpu": "5", "d2p_per_p": "256", "p2d_per_p": "128",
        "expected_gpu_processes": 4,
    },
    "1p1d": {
        "label": "1P:1D", "concurrency": 128,
        "p_gpus": "0", "d_gpus": "1",
        "p_ports": "27300", "bootstrap_ports": "28300",
        "d_ports": "27301", "direct_ports": "61000",
        "p_mem": "0.80", "d_mem": "0.60",
        "model_gpus": "0 1", "model_ports": "27200 27201",
        "model_mem": "0.80 0.60",
        "search_gpu": "1", "d2p_per_p": "512", "p2d_per_p": "256",
        "expected_gpu_processes": 2,
    },
}
CASES = [
    (topology, mode, int(config["concurrency"]))
    for topology, config in TOPOLOGIES.items()
    for mode in MODES
]

REUSED_4P4D_DIRS = {
    mode: Path(
        "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
        f"browsecomp-qwen3-8b-h100-{mode.replace('_', '-')}-c512-w300-m1200-20261006-r1"
    )
    for mode in MODES
}
REUSED_RESOURCE_METRICS = {
    "new_method": dict(p_kv=.8356, p_queue=35.74, p_inflight=16.34,
                       d_kv=.8529, d_running=50.97, d_queue=0.0,
                       d_prealloc=1.54, d_transfer=.34),
    "colocated": dict(d_kv=.8131, d_running=60.68, d_queue=2.19),
    "no_reverse": dict(p_kv=.0780, p_queue=14.47, p_inflight=.71,
                       d_kv=.9160, d_running=12.90, d_queue=0.0,
                       d_prealloc=98.18, d_transfer=16.73),
    "native_mooncake": dict(p_kv=.0854, p_queue=13.34, p_inflight=.86,
                            d_kv=.9126, d_running=15.92, d_queue=0.0,
                            d_prealloc=96.00, d_transfer=15.84),
}


spec = importlib.util.spec_from_file_location("browsecomp8b_lowc_reuse", LOWC)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {LOWC}")
lowc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lowc
spec.loader.exec_module(lowc)
base = lowc.base
base.CONTROL = CONTROL
base.CASES = CASES


def case_key(case: tuple[str, str, int]) -> str:
    topology, mode, concurrency = case
    return f"browsecomp8b-scale-{topology}-{mode}-c{concurrency}"


def run_dir(case: tuple[str, str, int]) -> Path:
    topology, mode, concurrency = case
    return base.RUNS / (
        "browsecomp-qwen3-8b-h100-pd-scale-"
        f"{topology}-{mode.replace('_', '-')}-c{concurrency}-w300-m1200-20261021-r1"
    )


def common_environment(case: tuple[str, str, int], directory: Path) -> dict[str, str]:
    env = lowc.common_environment(case, directory)
    topology = TOPOLOGIES[case[0]]
    env.update(
        SEARCH_GPU=str(topology["search_gpu"]),
        SEARCH_START_AFTER_MODELS="true",
        SEARCH_PORT="8750",
        # Persist topology metadata in launch_config.json for the watchdog and
        # for later audit of the raw result directories.
        ABLATION_TOPOLOGY=str(topology["label"]),
        EXPECTED_GPU_PROCESSES=str(topology["expected_gpu_processes"]),
    )
    return env


def modified_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = lowc.modified_environment(case, directory)
    topology = TOPOLOGIES[case[0]]
    env.update(
        # lowc.modified_environment deliberately uses numeric 1; do not merge
        # common_environment again here because the baseline spelling "true"
        # would silently disable the modified launcher's closed-loop branch.
        CLOSED_LOOP="1",
        SEARCH_GPU=str(topology["search_gpu"]),
        ABLATION_TOPOLOGY=str(topology["label"]),
        EXPECTED_GPU_PROCESSES=str(topology["expected_gpu_processes"]),
        PREFILL_GPUS=str(topology["p_gpus"]),
        DECODE_GPUS=str(topology["d_gpus"]),
        PREFILL_PORTS=str(topology["p_ports"]),
        BOOTSTRAP_PORTS=str(topology["bootstrap_ports"]),
        DECODE_PORTS=str(topology["d_ports"]),
        AGENTIC_DIRECT_PORTS=str(topology["direct_ports"]),
        PREFILL_MEM_FRACTION_STATICS=str(topology["p_mem"]),
        DECODE_MEM_FRACTION_STATICS=str(topology["d_mem"]),
        # Keep D->P=512 GiB and P->D=256 GiB in aggregate at every scale.
        D2P_HOST_ARENA_GIB_PER_P=str(topology["d2p_per_p"]),
        P2D_HOST_ARENA_GIB_PER_P=str(topology["p2d_per_p"]),
        SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB=str(topology["d2p_per_p"]),
        SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB=str(topology["p2d_per_p"]),
    )
    if case[0] == "1p1d":
        # With a single P there is no NUMA-domain choice to make.  The router
        # intentionally rejects NUMA-domain routing for fewer than two P
        # workers; keep late binding/global-D enabled but use the sole P
        # directly.
        env.update(
            PD_LATE_BIND_NUMA_DOMAINS="0",
            SGLANG_PD_LATE_BIND_NUMA_DOMAINS="0",
            SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS="0",
            # At c128 a single P legitimately queues parent-turn prefills for
            # several minutes.  The 30 s multi-P safety timeout aborts those
            # requests before admission and prevents warmup from converging.
            # This only extends the guard; P inflight and benchmark load stay
            # unchanged.
            P_ACCEPT_TIMEOUT_SECONDS="600",
        )
    return env, command


def baseline_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env, command = lowc.baseline_environment(case, directory)
    topology = TOPOLOGIES[case[0]]
    env.update(common_environment(case, directory))
    if case[1] == "colocated":
        env.update(
            MODEL_GPUS=str(topology["model_gpus"]),
            MODEL_PORTS=str(topology["model_ports"]),
            MODEL_MEM_FRACTION_STATICS=str(topology["model_mem"]),
            SEARCH_PORT="8720",
        )
    else:
        env.update(
            PREFILL_GPUS=str(topology["p_gpus"]),
            DECODE_GPUS=str(topology["d_gpus"]),
            PREFILL_PORTS=str(topology["p_ports"]),
            PREFILL_BOOTSTRAP_PORTS=str(topology["bootstrap_ports"]),
            DECODE_PORTS=str(topology["d_ports"]),
            PREFILL_MEM_FRACTION_STATICS=str(topology["p_mem"]),
            DECODE_MEM_FRACTION_STATICS=str(topology["d_mem"]),
        )
    return env, command


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    return (
        modified_environment(case, directory)
        if case[1] == "new_method"
        else baseline_environment(case, directory)
    )


def validate_inputs() -> None:
    lowc.validate_inputs()
    if not MODEL.joinpath("config.json").exists():
        raise RuntimeError(f"model is missing: {MODEL}")


def fmt(value: object, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def pct(value: object, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def status_text(item: dict[str, Any]) -> str:
    if item.get("reused"):
        return "复用旧结果"
    return {"complete": "完成", "running": "运行中", "failed": "失败"}.get(
        str(item.get("status")), "待运行"
    )


def result_for(states: dict[str, Any], case: tuple[str, str, int]) -> dict[str, Any]:
    item = states.get(case_key(case), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}
    return result if isinstance(result, dict) else {}


def render_report(states: dict[str, Any]) -> None:
    complete = sum(
        isinstance(states.get(case_key(case)), dict)
        and states[case_key(case)].get("status") == "complete"
        for case in CASES
    )
    lines = [
        "# BrowseComp Qwen3-8B H100 P:D 等比例缩放消融",
        "",
        f"> 自动矩阵进度：**{complete}/12**（其中 4P:4D/c512 四组复用既有有效结果）。",
        "> 2P:2D/c256 与 1P:1D/c128 共八组在同一个 H100 allocation 内顺序重测；",
        "> 每组采用 300 秒业务预热 + 1200 秒正式稳态窗口。除 P:D 数量与并发外，",
        "> 模型、BrowseComp 请求序、上下文、采样、搜索服务和方法参数均保持一致。",
        "",
        "## 配置",
        "",
        "| 拓扑 | 并发 | P GPU | D GPU | 新方法 D→P Host | 新方法 P→D Host |",
        "|---|---:|---|---|---:|---:|",
    ]
    for name, config in TOPOLOGIES.items():
        lines.append(
            f"| {config['label']} | {config['concurrency']} | `{config['p_gpus']}` | "
            f"`{config['d_gpus']}` | 512 GiB total | 256 GiB total |"
        )
    lines.extend([
        "",
        "说明：原生 HiCache/Mooncake 保留其原生机制与每实例容量配置；搜索模型继续与最后一张 D/Colocated GPU 共卡。",
        "",
        "## 核心结果",
        "",
        "| 拓扑 | 并发 | 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for case in CASES:
        topology, mode, concurrency = case
        item = states.get(case_key(case), {})
        result = result_for(states, case)
        completed = result.get("completed_agents")
        failures = result.get("failures")
        done = "—" if completed is None else f"{int(completed):,} / {int(failures or 0):,}"
        lines.append(
            f"| {TOPOLOGIES[topology]['label']} | {concurrency} | {LABELS[mode]} | "
            f"{status_text(item)} | {fmt(result.get('decode_tps'))} | "
            f"{fmt(result.get('agent_s'), 4)} | {fmt(result.get('prefill_tps'))} | "
            f"{pct(result.get('parent_kv_reuse'))} | {done} |"
        )
    lines.extend([
        "",
        "## 稳态资源细化",
        "",
        "| 拓扑 | 并发 | 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for case in CASES:
        topology, mode, concurrency = case
        result = result_for(states, case)
        colocated = mode == "colocated"
        lines.append(
            f"| {TOPOLOGIES[topology]['label']} | {concurrency} | {LABELS[mode]} | "
            f"{pct(result.get('prefill_forward_per_gpu'))} | "
            f"{'—' if colocated else pct(result.get('p_kv'))} | "
            f"{'—' if colocated else fmt(result.get('p_queue'), 2)} | "
            f"{'—' if colocated else fmt(result.get('p_inflight'), 2)} | "
            f"{pct(result.get('decode_forward_per_gpu'))} | {pct(result.get('d_kv'))} | "
            f"{fmt(result.get('d_running'), 2)} | {fmt(result.get('d_queue'), 2)} | "
            f"{fmt(result.get('d_prealloc'), 2)} | {fmt(result.get('d_transfer'), 2)} |"
        )
    lines.extend(["", "## 原始结果目录", ""])
    for case in CASES:
        item = states.get(case_key(case), {})
        directory = item.get("run_dir") if isinstance(item, dict) else None
        lines.append(f"- `{case_key(case)}`：`{directory or run_dir(case)}`")
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(REPORT)


def seed_reused_4p4d(states: dict[str, Any]) -> dict[str, Any]:
    """Import the already validated 2026-10-06 c512 results into this matrix."""
    for index, mode in enumerate(MODES, 1):
        case = ("4p4d", mode, 512)
        directory = REUSED_4P4D_DIRS[mode]
        summary_path = directory / "offload_analysis_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"reused summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text())
        reuse = summary.get("completion_weighted_reuse", {})
        result = {
            "status": summary["status"],
            "warmup_seconds": summary["warmup_seconds"],
            "measurement_seconds": summary["measurement_seconds"],
            "completed_agents": summary["completed_agents"],
            "failures": 0,
            "agent_s": summary["completion_rps"],
            "prefill_tps": summary["prefill_compute_tps"],
            "decode_tps": summary["decode_tps"],
            "parent_kv_reuse": reuse.get("page_aligned_parent_prefix_reuse_fraction", 0.0),
            "prefill_forward_per_gpu": (
                summary["prefill_gpu_busy_fraction"] / summary["prefill_gpus"]
            ),
            "decode_forward_per_gpu": (
                summary["decode_gpu_busy_fraction"] / summary["decode_gpus"]
            ),
            "mooncake": summary.get("mooncake", {}),
            **REUSED_RESOURCE_METRICS[mode],
        }
        states[case_key(case)] = {
            "status": "complete",
            "index": index,
            "total": len(CASES),
            "reused": True,
            "source_node": "node1701",
            "source_date": "2026-10-06",
            "run_dir": str(directory),
            "result": result,
        }
    return states


def assert_post_case_gate(states: dict[str, Any]) -> None:
    state_path = CONTROL / "state.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    newly_complete = [
        case for case in CASES
        if states.get(case_key(case), {}).get("status") == "complete"
        and previous.get(case_key(case), {}).get("status") != "complete"
    ]
    for case in newly_complete:
        env, _ = case_environment(case, run_dir(case))
        base.wait_ports_free(env, timeout=300)
        occupied = base.gpu_processes()
        if occupied:
            raise RuntimeError(f"post-case GPU cleanup gate failed: {occupied}")
        helper_names = {"apptainer", "squashfuse_ll", "fuse-overlayfs"}
        helper_pids: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if (entry / "comm").read_text().strip() in helper_names:
                    helper_pids.append(int(entry.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        for pid in helper_pids:
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 30
        while helper_pids and time.monotonic() < deadline:
            helper_pids = [pid for pid in helper_pids if Path(f"/proc/{pid}").exists()]
            if helper_pids:
                time.sleep(1)
        for pid in helper_pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        helpers = subprocess.run(
            ["pgrep", "-af", "apptainer|squashfuse_ll|fuse-overlayfs"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        ).stdout.strip()
        if helpers:
            raise RuntimeError(f"post-case helper cleanup gate failed: {helpers}")


def atomic_json(path: Path, value: object) -> None:
    if path == CONTROL / "state.json" and isinstance(value, dict):
        assert_post_case_gate(value)
    lowc.original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        render_report(value)


base.case_key = case_key
base.run_dir = run_dir
base.common_environment = common_environment
base.modified_environment = modified_environment
base.baseline_environment = baseline_environment
base.case_environment = case_environment
base.validate_inputs = validate_inputs
base.atomic_json = atomic_json


if __name__ == "__main__":
    existing = json.loads((CONTROL / "state.json").read_text()) if (CONTROL / "state.json").exists() else {}
    existing = seed_reused_4p4d(existing)
    lowc.original_atomic_json(CONTROL / "state.json", existing)
    render_report(existing)
    base.main()
