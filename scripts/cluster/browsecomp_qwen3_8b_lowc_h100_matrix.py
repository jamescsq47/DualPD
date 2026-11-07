#!/usr/bin/env python3
"""Serial low-concurrency extension of the Qwen3-8B BrowseComp H100 matrix.

This is a control-plane-only wrapper around the controller that produced the
existing H100 matrices.  It does not modify either SGLang implementation.
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


CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-8b-lowc-12-matrix-20261018"
)
BASE_CONTROLLER = Path(__file__).with_name("matrix_controller.py")
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-8B")
REPORT = Path(
    "/home/shangy/dualpd/DualPD/runs-host/"
    "BROWSECOMP_QWEN3_8B_H100.md"
)
CONCURRENCIES = (256, 128, 64)
MODES = ("new_method", "colocated", "no_reverse", "native_mooncake")
# The user requested each concurrency's four methods before moving to the next.
CASES = [("browsecomp8b4p4d", mode, c) for c in CONCURRENCIES for mode in MODES]
LABELS = {
    "new_method": "当前新方法（Slow 拥堵反馈重算）",
    "colocated": "Colocated baseline",
    "no_reverse": "No-reverse PD",
    "native_mooncake": "原生 Mooncake",
}
START_MARKER = "<!-- LOW_CONCURRENCY_20261018_START -->"
END_MARKER = "<!-- LOW_CONCURRENCY_20261018_END -->"


spec = importlib.util.spec_from_file_location("browsecomp8b_lowc_base", BASE_CONTROLLER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_CONTROLLER}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

base.CONTROL = CONTROL
base.CASES = CASES
original_common_environment = base.common_environment
original_atomic_json = base.atomic_json


def case_key(case: tuple[str, str, int]) -> str:
    _, mode, concurrency = case
    return f"browsecomp8b-lowc-{mode}-c{concurrency}"


def run_dir(case: tuple[str, str, int]) -> Path:
    _, mode, concurrency = case
    return base.RUNS / (
        "browsecomp-qwen3-8b-h100-"
        f"{mode.replace('_', '-')}-c{concurrency}-w300-m1200-20261018-r1"
    )


def common_environment(case: tuple[str, str, int], directory: Path) -> dict[str, str]:
    # Reuse the proven pure-BrowseComp branch, then only replace the model and
    # the physical TP=1 layout in the method-specific functions below.
    _, mode, concurrency = case
    env = original_common_environment(("browsecomp32b", mode, concurrency), directory)
    env.update(
        MODEL_PATH=str(MODEL),
        REQUESTS="680",
        MATH_RATIO="0",
        SCHEDULE_FILE=str(
            base.PD / "configs/workloads/fixed_browsecomp_source_order_n680.json"
        ),
        PRESERVE_SOURCE_ORDER="true",
        WARMUP_SECONDS="300",
        MAX_WARMUP_SECONDS="420",
        MEASURE_SECONDS="1200",
        MODEL_CONTEXT_LENGTH="40960",
        MODEL_MAX_RESPONSE_LENGTH="36864",
        SEARCH_GPU="7",
        SEARCH_START_AFTER_MODELS="true",
        SEARCH_PORT="8750",
        MEM_FRACTION_STATIC="0.80",
        POST_ANALYZER="pd_offload",
    )
    return env


def modified_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    env = common_environment(case, directory)
    env_bin = base.PD_ENV / "bin"
    cuda_lib = env_bin.parent / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    env.update(
        PD_ENV_BIN=str(env_bin),
        PATH=f"{env_bin}:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        PYTHONPATH=f"{base.MODIFIED_SGLANG}:{base.PD}:{base.REPO}",
        LD_LIBRARY_PATH=f"{cuda_lib}:{env_bin.parent / 'lib'}",
        CPATH="/usr/include/x86_64-linux-gnu:/usr/include",
        LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu",
        PREFILL_GPUS="0 2 4 6",
        DECODE_GPUS="1 3 5 7",
        PREFILL_TP_SIZE="1",
        DECODE_TP_SIZE="1",
        PREFILL_PORTS="27300 27320 27400 27420",
        BOOTSTRAP_PORTS="28300 28320 28400 28420",
        DECODE_PORTS="27301 27321 27401 27421",
        PREFILL_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80",
        DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.60",
        ROUTER_PORT="27510",
        ROUTER_PROMETHEUS_PORT="27520",
        # The modified run_pd_servers.sh uses the numeric spelling, whereas
        # the baseline launchers use the literal string "true".
        CLOSED_LOOP="1",
        # run_pd_servers.sh derives one reverse-NIXL listener per Decode
        # endpoint.  The expanded *_PORTS value makes the generic pre/post
        # cleanup gate cover all four listeners too.
        AGENTIC_DIRECT_BASE_PORT="61000",
        AGENTIC_DIRECT_PORTS="61000 61001 61002 61003",
        FAST_TOOL_THRESHOLD_SECONDS="1",
        DIRECT_WAIT_SECONDS="1",
        D_TARGET_KV_FRACTION="1.0",
        P2D_SPILL_DELAY_SECONDS="0.5",
        P2D_HOST_STAGING="true",
        D2P_HOST_ARENA_GIB_PER_P="128",
        P2D_HOST_ARENA_GIB_PER_P="64",
        MAX_PREFILL_INFLIGHT="12",
        P_TO_D_CONSUMERS="24",
        SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY="true",
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE="true",
        SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE="false",
        SGLANG_AGENTIC_KV_FORCE_SLOW_PATH="false",
        SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE="false",
        SGLANG_PD_ABLATION_P2D_PREBIND="false",
        SGLANG_PD_ABLATION_RANDOM_ROUTING="false",
        SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND="memfd",
        SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB="128",
        SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND="memfd",
        SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB="64",
        SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA="1",
        SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA="1",
        SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER="1",
        SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS="1800",
        SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB="8",
        SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB="768",
        SGLANG_AGENTIC_KV_D2H_INFLIGHT="2",
        P_H2D_MAX_INFLIGHT="2",
        SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT="2",
        SGLANG_AGENTIC_KV_P2D_D2H_WORKERS="2",
        SGLANG_AGENTIC_KV_P2D_H2D_WORKERS="2",
        SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH="32",
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW="8",
    )
    return env, ["bash", str(base.PD / "scripts/new_method/run_4p4d_numa_case.sh")]


def baseline_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    _, mode, _ = case
    env = common_environment(case, directory)
    env_bin = base.BASELINE_ENV / "bin"
    cuda_lib = env_bin.parent / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    env.update(
        PD_ENV_BIN=str(env_bin),
        PATH=f"{env_bin}:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        PYTHONPATH="",
        LD_LIBRARY_PATH=f"{cuda_lib}:{env_bin.parent / 'lib'}",
        UCX_LOG_LEVEL="info",
    )
    if mode == "colocated":
        env.update(
            MODEL_GPUS="0 1 2 3 4 5 6 7",
            MODEL_TP_SIZE="1",
            MODEL_PORTS="27200 27201 27202 27203 27204 27205 27206 27207",
            MODEL_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80 0.80 0.80 0.80 0.60",
            ROUTER_PORT="27210",
            SEARCH_PORT="8720",
        )
        return env, ["bash", str(base.PD / "scripts/baseline/run_colocated_case.sh")]

    env.update(
        CASE_MODE=mode,
        PREFILL_GPUS="0 2 4 6",
        DECODE_GPUS="1 3 5 7",
        PREFILL_TP_SIZE="1",
        DECODE_TP_SIZE="1",
        PREFILL_PORTS="27300 27320 27400 27420",
        PREFILL_BOOTSTRAP_PORTS="28300 28320 28400 28420",
        DECODE_PORTS="27301 27321 27401 27421",
        PREFILL_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80",
        DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.60",
        ROUTER_PORT="27510",
        ROUTER_PROMETHEUS_PORT="27520",
        SEARCH_PORT="8750",
        P_HICACHE_SIZE="128",
        D_HICACHE_SIZE="56",
        P_HICACHE_WRITE_POLICY="write_through",
        MOONCAKE_MASTER_PORT="57651",
        MOONCAKE_CLIENT_PORT="57652",
        MOONCAKE_METADATA_PORT="57680",
        MOONCAKE_METRICS_PORT="57603",
        MOONCAKE_CLIENT_HTTP_PORT="57690",
        TP1_LOCAL_NUMA="true",
        SCHEDULER_WATCHDOG_TIMEOUT="1200",
    )
    return env, ["bash", str(base.PD / "scripts/baseline/run_pd_case.sh")]


def case_environment(
    case: tuple[str, str, int], directory: Path
) -> tuple[dict[str, str], list[str]]:
    return (
        modified_environment(case, directory)
        if case[1] == "new_method"
        else baseline_environment(case, directory)
    )


def validate_inputs() -> None:
    required = [
        BASE_CONTROLLER,
        base.PD_ENV / "bin/python",
        base.BASELINE_ENV / "bin/python",
        base.MODIFIED_SGLANG / "sglang/__init__.py",
        MODEL / "config.json",
        base.DATA / "browsecomp/bc_train.jsonl",
        base.PD / "configs/workloads/fixed_browsecomp_source_order_n680.json",
        Path(common_environment(CASES[0], base.CONTROL)["SEARCH_SERVER_EMBEDDING_CACHE"]),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing inputs:\n" + "\n".join(missing))


def fmt(value: object, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def pct(value: object, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def status_text(item: dict[str, Any]) -> str:
    return {
        "complete": "完成",
        "running": "运行中",
        "failed": "失败（控制器已停止）",
    }.get(str(item.get("status")), "待运行")


def result_for(states: dict[str, Any], mode: str, concurrency: int) -> dict[str, Any]:
    item = states.get(case_key(("browsecomp8b4p4d", mode, concurrency)), {})
    result = item.get("result", {}) if isinstance(item, dict) else {}
    return result if isinstance(result, dict) else {}


def result_row(states: dict[str, Any], mode: str, concurrency: int) -> str:
    key = case_key(("browsecomp8b4p4d", mode, concurrency))
    item = states.get(key, {})
    result = result_for(states, mode, concurrency)
    completed = result.get("completed_agents")
    failures = result.get("failures")
    done = "—" if completed is None else f"{int(completed):,} / {int(failures or 0):,}"
    return (
        f"| {concurrency} | {LABELS[mode]} | {status_text(item)} | "
        f"{fmt(result.get('decode_tps'))} | {fmt(result.get('agent_s'), 4)} | "
        f"{fmt(result.get('prefill_tps'))} | {pct(result.get('parent_kv_reuse'))} | {done} |"
    )


def resource_row(states: dict[str, Any], mode: str, concurrency: int) -> str:
    result = result_for(states, mode, concurrency)
    colocated = mode == "colocated"
    return (
        f"| {concurrency} | {LABELS[mode]} | {pct(result.get('prefill_forward_per_gpu'))} | "
        f"{'—' if colocated else pct(result.get('p_kv'))} | "
        f"{'—' if colocated else fmt(result.get('p_queue'), 2)} | "
        f"{'—' if colocated else fmt(result.get('p_inflight'), 2)} | "
        f"{pct(result.get('decode_forward_per_gpu'))} | {pct(result.get('d_kv'))} | "
        f"{fmt(result.get('d_running'), 2)} | {fmt(result.get('d_queue'), 2)} | "
        f"{fmt(result.get('d_prealloc'), 2)} | {fmt(result.get('d_transfer'), 2)} |"
    )


def render_report(states: dict[str, Any]) -> None:
    complete = sum(
        isinstance(states.get(case_key(case)), dict)
        and states[case_key(case)].get("status") == "complete"
        for case in CASES
    )
    lines = [
        START_MARKER,
        "## 低并发扩展：c256/c128/c64（node2800，2026-10-18）",
        "",
        f"> 自动矩阵进度：**{complete}/12**。每组均为 300 秒业务预热 + 1200 秒正式测量；",
        "> 顺序为 c256 → c128 → c64，且每个并发依次运行当前新方法、Colocated、",
        "> No-reverse、原生 Mooncake。其余模型、数据、4P:4D 布局及容量配置与上文一致。",
        "",
        "### 正式结果",
        "",
        "| 并发 | 方法 | 状态 | Decode token/s | Agent/s | Prefill token/s | Parent KV 复用 | 完成/失败 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(result_row(states, mode, c) for c in CONCURRENCIES for mode in MODES)
    lines.extend([
        "",
        "### 稳态资源",
        "",
        "| 并发 | 方法 | P Forward/卡 | P KV/引擎 | P queue | P inflight | D Forward/卡 | D KV/引擎 | D running | D queue | D prealloc | D transfer |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(resource_row(states, mode, c) for c in CONCURRENCIES for mode in MODES)
    lines.extend(["", "### 原始结果目录", ""])
    for case in CASES:
        key = case_key(case)
        item = states.get(key, {})
        directory = item.get("run_dir") if isinstance(item, dict) else None
        lines.append(f"- `{key}`：`{directory or run_dir(case)}`")
    lines.extend([END_MARKER, ""])
    block = "\n".join(lines)
    current = REPORT.read_text()
    if START_MARKER in current:
        before, tail = current.split(START_MARKER, 1)
        _, after = tail.split(END_MARKER, 1)
        updated = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        updated = current.rstrip() + "\n\n" + block
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(updated)
    temporary.replace(REPORT)


def assert_post_case_gate(states: dict[str, Any]) -> None:
    previous = json.loads((CONTROL / "state.json").read_text()) if (CONTROL / "state.json").exists() else {}
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
        # A dead Apptainer parent can leave a FUSE helper reparented to PID 1.
        # The node is exclusive to this allocation and the case launcher has
        # already exited here, so reap such helpers before enforcing zero.
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
    original_atomic_json(path, value)
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
    render_report(existing)
    base.main()
