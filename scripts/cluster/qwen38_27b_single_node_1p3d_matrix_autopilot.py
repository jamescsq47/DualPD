#!/usr/bin/env python3
"""Run the Qwen3.8-27B TP2 single-node 1P:3D SWE500 matrix serially."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

try:
    from . import qwen35_9b_h100_matrix_autopilot as core
except ImportError:  # Direct script execution.
    import qwen35_9b_h100_matrix_autopilot as core


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
CONTROL_DIR = (
    RUNS / "qwen38-27b-tp2-single-node-1p3d-c64c128c192-growth128k-20261014"
)
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
BASELINE_VALIDATION_PATH = CONTROL_DIR / "baseline-correctness-validation.json"
REPORT = (
    ROOT
    / "runs-host/"
    "SWEBENCH_QWEN38_27B_TP2_SINGLE_NODE_1P3D_H100.md"
)

JOB_ID = "25642154"
NODE = "node2700"
PD_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/"
    "pd_baseline/bin/python3.12"
)
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3.8-27B")
DATA = Path("/orcd/scratch/orcd/003/shangy/dualpd/datasets")
SIF = Path("/orcd/pool/004/shangy/dualpd/containers/swebench-verified-sif")
WORKLOAD = (
    ROOT
    / "configs/experiments/"
    "swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer.yaml"
)
PD_ENGINE = WORKSPACE / "sglang_pd"
HICACHE_ENGINE = RUNS.parent / "sglang-src/pd_baseline_b1ef6"
BASELINE_ENGINE = RUNS.parent / "sglang-src/pd_baseline_b1ef6_q38fix"
MOONCAKE_COMPAT = ROOT / "scripts/compat/qwen35_mooncake"
AGENTIC_LAUNCHER = ROOT / "scripts/cluster/qwen35_slurm.py"
COLOCATED_LAUNCHER = ROOT / "scripts/cluster/qwen35_colocated_slurm.py"
NATIVE_LAUNCHER = ROOT / "scripts/cluster/qwen35_native_pd_slurm.py"
_BASE_REBUILD_ALLOCATION = core.rebuild_allocation

METHOD_LABEL = {
    "agentic": "当前新方法（1P:3D）",
    "colocated": "Colocated（4×TP2）",
    "native": "No-reverse PD（1P:3D）",
    "hicache": "原生 HiCache/Mooncake（1P:3D）",
}
METHODS = ("agentic", "colocated", "native", "hicache")
CONCURRENCIES = (64, 128, 192)
# The HiCache engine can come from a different SGLang revision than the
# corrected no-reverse baseline.  Keep its CLI spelling configurable so a
# model-specific controller does not need to alter other matrices.
HICACHE_MAMBA_CACHE_STRATEGY_FLAG = "radix"
# Model-specific controllers may append read-only result tables without
# copying this controller's report renderer.  The callback receives the same
# completed-result dictionaries used by the core and auxiliary tables.
REPORT_EXTRA_SECTIONS = None


def run_name(concurrency: int, method: str) -> str:
    names = {
        "agentic": "agentic",
        "colocated": "colocated4",
        "native": "native-oneway",
        "hicache": "hicache-mooncake",
    }
    if method == "colocated":
        revision = "correctness-r3"
    elif method == "native":
        revision = "correctness-r4"
    else:
        revision = "r2"
    return (
        f"q38-27b-h100-1n-1p3d-{names[method]}-"
        f"swe500-c{concurrency}-20261014-{revision}"
    )


RESULT_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_name(concurrency, method),
    }
    for concurrency in CONCURRENCIES
    for method in METHODS
]

# The c64 new-method result already completed before the baseline correctness
# defect was identified.  The c64 HiCache run was in flight when the controller
# was migrated, so resume it first, then rerun both corrected c64 baselines.
# Later concurrency levels retain the ordinary four-method order.
EXPERIMENTS: list[dict[str, Any]] = [
    next(
        experiment
        for experiment in RESULT_EXPERIMENTS
        if experiment["concurrency"] == 64 and experiment["method"] == "hicache"
    ),
    *[
        experiment
        for experiment in RESULT_EXPERIMENTS
        if experiment["concurrency"] == 64
        and experiment["method"] in {"colocated", "native"}
    ],
    *[
        experiment
        for experiment in RESULT_EXPERIMENTS
        if experiment["concurrency"] in {128, 192}
    ],
]


def run_dir(experiment: dict[str, Any]) -> Path:
    return RUNS / experiment["run_id"]


def common_paths(experiment: dict[str, Any]) -> list[str | Path]:
    return [
        "--job-id", JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", run_dir(experiment),
        "--model", MODEL,
        "--data-root", DATA,
        "--swe-sif-dir", SIF,
        "--workload-config", WORKLOAD,
        "--requests", "500",
        "--concurrency", str(experiment["concurrency"]),
        "--tp-size", "2",
    ]


def validate_corrected_baseline() -> None:
    """Run the two baseline-only regressions once on a clean allocated GPU."""
    if BASELINE_VALIDATION_PATH.is_file():
        return
    tests = [
        str(
            BASELINE_ENGINE
            / "test/registered/models/test_qwen35_gdn_fallback_layout.py"
        ),
        str(
            BASELINE_ENGINE
            / "test/registered/disaggregation/test_native_mamba_admission.py"
        ),
    ]
    core.command(
        [
            "srun", "--jobid", JOB_ID, "--overlap", "--exact",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "-w", NODE,
            "env",
            f"PYTHONPATH={BASELINE_ENGINE / 'python'}",
            "PD_GDN_CUDA_PROBE=1",
            BASELINE_PYTHON, "-m", "pytest", "-q", *tests,
        ],
        timeout=1200,
    )
    core.atomic_json(
        BASELINE_VALIDATION_PATH,
        {
            "base_commit": "b1ef6ef003305edf07337b9ba4ced62f807606db",
            "engine": str(BASELINE_ENGINE),
            "node": NODE,
            "job_id": JOB_ID,
            "validated_at_epoch": time.time(),
            "tests": tests,
            "result": "GO",
        },
    )


def prepare_and_start(experiment: dict[str, Any]) -> None:
    directory = run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse run directory: {directory}")
    method = experiment["method"]
    if method == "agentic":
        common = [
            PD_PYTHON, AGENTIC_LAUNCHER,
            *common_paths(experiment),
            "--prefill-node", NODE,
            "--decode-node", NODE,
            "--python", PD_PYTHON,
            "--sglang-root", PD_ENGINE,
            "--single-node-1p3d",
            "--d2p-host-gib-per-rank", "85",
            "--p2d-host-gib-per-rank", "128",
            "--d2p-direct-lanes", "2",
            "--p2d-direct-lanes", "1",
            "--host-lanes", "2",
            "--host-load-workers", "1",
            "--nixl-progress-threads", "2",
            "--forward-stream-priority", "-5",
            "--flashinfer-allreduce-max-tokens", "8192",
            "--decode-growth-tokens", "131072",
            "--launch-transport", "ssh",
        ]
        core.command([*common, "prepare"])
        core.command([*common, "--detach", "start-control"])
        core.command([*common, "control-ready"], timeout=900)
        core.command([*common, "--detach", "start-prefill"], timeout=2400)
        core.command([*common, "--detach", "start-decode"], timeout=1200)
        core.command([*common, "wait-ready"], timeout=2400)
        core.command([*common, "--detach", "start-router"], timeout=600)
        core.command([*common, "--detach", "run-workload"], timeout=600)
        return
    if method == "colocated":
        validate_corrected_baseline()
        prepare = [
            BASELINE_PYTHON, COLOCATED_LAUNCHER, "prepare",
            *common_paths(experiment),
            "--python", BASELINE_PYTHON,
            "--workload-python", PD_PYTHON,
            "--sglang-root", BASELINE_ENGINE,
            "--single-node",
            "--groups-per-node", "4",
            "--node0-groups", "4",
            "--cpus-per-node", "224",
        ]
        core.command(prepare)
        core.command(
            [BASELINE_PYTHON, COLOCATED_LAUNCHER, "start",
             "--job-id", JOB_ID, "--run-dir", directory],
            timeout=3600,
        )
        return
    if method == "native":
        validate_corrected_baseline()
    prepare = [
        BASELINE_PYTHON, NATIVE_LAUNCHER, "prepare",
        *common_paths(experiment),
        "--python", BASELINE_PYTHON,
        "--workload-python", PD_PYTHON,
        "--sglang-root", HICACHE_ENGINE if method == "hicache" else BASELINE_ENGINE,
        "--mamba-full-memory-ratio", "0.9",
        "--single-node-1p3d",
        "--prefill-groups-per-node", "1",
        "--decode-groups-per-node", "3",
        "--launch-transport", "ssh",
    ]
    if method == "hicache":
        prepare.extend([
            "--native-mooncake",
            "--mamba-cache-strategy-flag", HICACHE_MAMBA_CACHE_STRATEGY_FLAG,
            "--mooncake-compat", MOONCAKE_COMPAT,
            "--prefill-hicache-size-gb", "32",
            "--decode-hicache-size-gb", "32",
            "--mamba-hicache-size-gb", "32",
            "--mooncake-store-size-gb", "256",
            "--mooncake-prefetch-policy", "wait_complete",
            "--workload-job-id", JOB_ID,
            "--workload-node", NODE,
            "--storage-job-id", JOB_ID,
            "--storage-node", NODE,
        ])
    else:
        # The clean b1ef/v0.5.10 baseline exposes the legacy scheduler flag;
        # do not inherit the newer pd launcher's radix-cache flag default.
        prepare.extend(["--mamba-cache-strategy-flag", "scheduler"])
    core.command(prepare)
    core.command(
        [BASELINE_PYTHON, NATIVE_LAUNCHER, "start",
         "--job-id", JOB_ID, "--run-dir", directory],
        timeout=3600,
    )


def stop_experiment(experiment: dict[str, Any]) -> None:
    directory = run_dir(experiment)
    method = experiment["method"]
    if method == "agentic":
        common = [
            PD_PYTHON, AGENTIC_LAUNCHER,
            "--job-id", JOB_ID,
            "--prefill-node", NODE,
            "--decode-node", NODE,
            "--python", PD_PYTHON,
            "--run-dir", directory,
            "--single-node-1p3d",
            "--tp-size", "2",
            "--launch-transport", "ssh",
        ]
        for component, role in (
            ("workload", None), ("router", None), ("worker", "decode"),
            ("worker", "prefill"), ("control", None),
        ):
            argv = [*common, "stop", "--component", component]
            if role:
                argv.extend(["--worker-role", role])
            core.command(argv, timeout=900, check=False)
    elif method == "colocated":
        core.command(
            [BASELINE_PYTHON, COLOCATED_LAUNCHER, "stop",
             "--job-id", JOB_ID, "--run-dir", directory],
            timeout=1800, check=False,
        )
    else:
        core.command(
            [BASELINE_PYTHON, NATIVE_LAUNCHER, "stop",
             "--job-id", JOB_ID, "--run-dir", directory],
            timeout=1800, check=False,
        )


def expected_runtime_ports(experiment: dict[str, Any]) -> set[int]:
    if experiment["method"] == "agentic":
        # 23903 is the router's optional metrics listener.  It is not part of
        # the request path and may disappear while the OpenAI/router ports stay
        # healthy, so do not reject an otherwise valid run on that port alone.
        return {23900, 23902, 23904, 24000, 24010, 24020}
    if experiment["method"] == "colocated":
        # Likewise, 24003 is metrics-only; 24002 is the request router.
        return {24000, 24002, 24010, 24020, 24030}
    ports = {24200, 24300, 24310, 24320, 24400, 24401}
    if experiment["method"] == "hicache":
        ports |= {25103, 25151, 25152, 25180, 25190}
    return ports


def rebuild_allocation(reason: str) -> None:
    """Reuse the proven replacement flow and synchronize this controller."""
    global JOB_ID, NODE
    _BASE_REBUILD_ALLOCATION(reason)
    JOB_ID = core.JOB_ID
    NODE = core.NODE


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = core.all_results()
    result_by_run = {result["run_id"]: result for result in results}
    current = (
        EXPERIMENTS[state["index"]]
        if state["index"] < len(EXPERIMENTS)
        else None
    )
    current_rows = core.load_progress(run_dir(current)) if current else []
    current_records = len(current_rows) if current_rows else current_progress
    current_normal = sum(
        str(row.get("status")) == "Status.COMPLETED" for row in current_rows
    )
    current_env_errors = sum(
        bool(row.get("environment_error")) for row in current_rows
    )
    current_solved = sum(row.get("reward") == 1 for row in current_rows)
    current_label = "运行中" if state["status"] == "running" else "已终止"
    current_prefix = "当前" if state["status"] == "running" else "停止于"
    lines = [
        core.START_MARKER,
        "## 自动运行进度与结果",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}。",
        f"> 状态：`{state['status']}`；有效结果 **{len(results)}/12**。",
    ]
    if current:
        detail = (
            f"（正常完成{current_normal}，resolved {current_solved}，"
            f"environment error {current_env_errors}）"
            if current_rows else ""
        )
        lines.append(
            f"> {current_prefix}：c{current['concurrency']} {METHOD_LABEL[current['method']]}，"
            f"terminal records **{current_records}/500**{detail}，"
            f"run `{current['run_id']}`。"
        )
    elif state["status"] == "complete":
        lines.append("> 12组实验均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend([
        "",
        "### 核心指标",
        "",
        "| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for experiment in RESULT_EXPERIMENTS:
        result = result_by_run.get(experiment["run_id"])
        if result is None:
            if current and experiment["run_id"] == current["run_id"] and current_rows:
                current_failed = current_records - current_normal
                lines.append(
                    f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                    f"| {current_normal}/{current_failed}（{current_label}） | "
                    f"{current_solved}/{current_records} | — | — | — |"
                )
                continue
            lines.append(
                f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                "| — | — | — | — | — |"
            )
            continue
        tokens = result["t450_decode_tokens"]
        lines.append(
            "| {c} | {method} | {done}/{failed} | {solved}/500 | **{t450}** | "
            "**{decode}** | {tokens} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                done=result["normal_completed"], failed=result["failed"],
                solved=result["solved"], t450=core.value(result["t450_seconds"]),
                decode=core.value(result["t450_decode_tps"]),
                tokens="—" if tokens is None else f"{tokens / 1e6:.3f}M",
            )
        )
    lines.extend([
        "",
        "### 全程与辅助指标",
        "",
        "| 并发 | 方法 | T50 | T250 | T500 | T50–T450 agent/s | 全程 Prefill tok/s | 全程 Decode tok/s | Max queue | 异常 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for experiment in RESULT_EXPERIMENTS:
        result = result_by_run.get(experiment["run_id"])
        if result is None:
            if current and experiment["run_id"] == current["run_id"] and current_rows:
                lines.append(
                    f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                    f"| — | — | — | — | — | — | — | {current_label} {current_records}/500 |"
                )
                continue
            lines.append(
                f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                "| — | — | — | — | — | — | — | 待完成 |"
            )
            continue
        notes = []
        if result["environment_errors"]:
            notes.append(f"env={result['environment_errors']}")
        if result["failed"]:
            notes.append(f"failed={result['failed']}")
        lines.append(
            "| {c} | {method} | {t50} | {t250} | {t500} | {rate} | {p} | "
            "{d} | {q} | {notes} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                t50=core.value(result["t50_seconds"]),
                t250=core.value(result["t250_seconds"]),
                t500=core.value(result["t500_seconds"]),
                rate=core.value(result["t50_t450_agent_s"], 5),
                p=core.value(result["scheduler_prefill_tps"]),
                d=core.value(result["scheduler_decode_tps"]),
                q=core.value(result["max_queue"], 0),
                notes=", ".join(notes) or "—",
            )
        )
    if REPORT_EXTRA_SECTIONS is not None:
        lines.extend(REPORT_EXTRA_SECTIONS(results))
    lines.extend(["", "原始目录：", ""])
    for result in results:
        lines.append(
            f"- c{result['concurrency']} {METHOD_LABEL[result['method']]}: "
            f"`{result['directory']}`"
        )
    lines.extend([core.END_MARKER, ""])
    block = "\n".join(lines)
    text = REPORT.read_text()
    before, rest = text.split(core.START_MARKER, 1)
    _, after = rest.split(core.END_MARKER, 1)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(before.rstrip() + "\n\n" + block + after.lstrip("\n"))
    os.replace(temporary, REPORT)


def configure_core() -> None:
    assignments = {
        "RUNS": RUNS,
        "CONTROL_DIR": CONTROL_DIR,
        "STATE_PATH": STATE_PATH,
        "LOCK_PATH": LOCK_PATH,
        "ALLOCATION_PATH": ALLOCATION_PATH,
        "REPORT": REPORT,
        "JOB_ID": JOB_ID,
        "NODE": NODE,
        "PD_PYTHON": PD_PYTHON,
        "BASELINE_PYTHON": BASELINE_PYTHON,
        "MODEL": MODEL,
        "DATA": DATA,
        "SIF": SIF,
        "WORKLOAD": WORKLOAD,
        "PD_ENGINE": PD_ENGINE,
        "BASELINE_ENGINE": BASELINE_ENGINE,
        "MOONCAKE_COMPAT": MOONCAKE_COMPAT,
        "AGENTIC_LAUNCHER": AGENTIC_LAUNCHER,
        "COLOCATED_LAUNCHER": COLOCATED_LAUNCHER,
        "NATIVE_LAUNCHER": NATIVE_LAUNCHER,
        "METHOD_LABEL": METHOD_LABEL,
        "EXPERIMENTS": EXPERIMENTS,
        "ALL_EXPERIMENTS": RESULT_EXPERIMENTS,
        "SANDBOX_HELPER_NAMES": tuple(
            dict.fromkeys((*core.SANDBOX_HELPER_NAMES, "faked-sysv"))
        ),
    }
    for name, value in assignments.items():
        setattr(core, name, value)
    core.prepare_and_start = prepare_and_start
    core.stop_experiment = stop_experiment
    core.expected_runtime_ports = expected_runtime_ports
    core.update_report = update_report
    core.rebuild_allocation = rebuild_allocation


def main() -> int:
    configure_core()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
