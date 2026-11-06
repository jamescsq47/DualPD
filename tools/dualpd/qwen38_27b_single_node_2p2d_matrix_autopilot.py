#!/usr/bin/env python3
"""Run the Qwen3.8-27B TP2 single-node 2P:2D SWE500 matrix serially."""

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
CONTROL_DIR = RUNS / "qwen38-27b-tp2-single-node-2p2d-c96c128c192-20261016-r2"
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
BASELINE_VALIDATION_PATH = CONTROL_DIR / "baseline-correctness-validation.json"
REPORT = (
    ROOT
    / "examples/pd/runs-host/"
    "SWEBENCH_QWEN38_27B_TP2_SINGLE_NODE_2P2D_H100.md"
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
    / "examples/pd/configs/experiments/"
    "swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer_overlay.yaml"
)
PD_ENGINE = WORKSPACE / "sglang_pd"
HICACHE_ENGINE = RUNS.parent / "sglang-src/pd_baseline_b1ef6"
BASELINE_ENGINE = RUNS.parent / "sglang-src/pd_baseline_b1ef6_q38fix"
MOONCAKE_COMPAT = ROOT / "examples/pd/scripts/compat/qwen35_mooncake"
AGENTIC_LAUNCHER = ROOT / "tools/dualpd/qwen35_slurm.py"
COLOCATED_LAUNCHER = ROOT / "tools/dualpd/qwen35_colocated_slurm.py"
NATIVE_LAUNCHER = ROOT / "tools/dualpd/qwen35_native_pd_slurm.py"
_BASE_REBUILD_ALLOCATION = core.rebuild_allocation
_BASE_COMPLETED_RESULT = core.completed_result

METHOD_LABEL = {
    "agentic": "当前新方法（2P:2D）",
    "colocated": "Colocated（4×TP2）",
    "native": "No-reverse PD（2P:2D）",
    "hicache": "原生 HiCache/Mooncake（2P:2D）",
}
# Colocated is independent of the P:D split and is unchanged from the prior
# single-node TP2 campaign, so this campaign only schedules the three methods
# whose 2P:2D behavior must be measured.
METHODS = ("agentic", "native", "hicache")
CONCURRENCIES = (96, 128, 192)
# The isolated b1ef6 HiCache baseline uses the older, upstream CLI spelling.
# This matches the previously successful 1P:3D HiCache launch manifest.
HICACHE_MAMBA_CACHE_STRATEGY_FLAG = "scheduler"
# c128 can place more than 56 active requests on one of the two Decode groups.
# The launcher's generic 0.5 split only creates 169 Mamba slots (three slots
# are needed per active request), so reserve the same 0.9 share already used by
# the corrected native baselines.  Apply this only to this matrix's prepared
# manifest so other qwen35_slurm users keep their existing defaults.
AGENTIC_MAMBA_FULL_MEMORY_RATIO = 0.9
# The 1800 GiB job cgroup reclaims transient verifier/overlay bursts without
# OOM on this 2 TiB node.  Retain 128 GiB for emergency cleanup while avoiding
# invalidation at the recovered, non-OOM 256 GiB warning used elsewhere.
MIN_MEM_AVAILABLE_KIB = 128 * 1024 * 1024
TARGET_TERMINALS = 450
T450_RESULT_NAME = "T450_RESULT.json"
BASELINE_FAILURE_NAME = "BASELINE_FAILURE.json"


def run_name(concurrency: int, method: str) -> str:
    names = {
        "agentic": "agentic",
        "colocated": "colocated4",
        "native": "native-oneway",
        "hicache": "hicache-mooncake",
    }
    if method == "hicache":
        revision = "overlay-r3"
    elif method == "agentic" and concurrency >= 128:
        revision = "overlay-r2"
    else:
        revision = "overlay-r1"
    return (
        f"q38-27b-h100-1n-2p2d-{names[method]}-"
        f"swe500-c{concurrency}-20261016-{revision}"
    )


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_name(concurrency, method),
    }
    for concurrency in CONCURRENCIES
    for method in METHODS
]


def run_dir(experiment: dict[str, Any]) -> Path:
    return RUNS / experiment["run_id"]


def baseline_failure(experiment: dict[str, Any]) -> dict[str, Any] | None:
    path = run_dir(experiment) / BASELINE_FAILURE_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _counter_window(
    directory: Path, role: str, key: str, start: float, end: float
) -> tuple[float | None, float | None]:
    """Return a summed per-endpoint counter delta and rate for one window."""

    metrics_path = directory / "workload/engine_metrics.jsonl"
    if not metrics_path.is_file() or end <= start:
        return None, None
    series: dict[str, list[tuple[float, float]]] = {}
    expected_endpoints = 0
    try:
        with metrics_path.open(errors="replace") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("role") != role:
                    continue
                expected_endpoints = max(
                    expected_endpoints, int(row.get("engine_count", 0))
                )
                for endpoint in row.get("endpoint_metrics", []):
                    value = endpoint.get("metrics", {}).get(key)
                    if value is None:
                        continue
                    series.setdefault(str(endpoint["endpoint"]), []).append(
                        (float(row["ts"]), float(value))
                    )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None
    if not series or len(series) < expected_endpoints:
        return None, None
    total = 0.0
    for points in series.values():
        points.sort()
        if len(points) < 2:
            return None, None
        total += max(
            0.0,
            core._counter_at_boundary(points, end)
            - core._counter_at_boundary(points, start),
        )
    return total, total / (end - start)


def _max_queue_until(directory: Path, end: float) -> float | None:
    path = directory / "workload/engine_metrics.jsonl"
    if not path.is_file():
        return None
    maximum: float | None = None
    try:
        with path.open(errors="replace") as stream:
            for line in stream:
                row = json.loads(line)
                if float(row.get("ts", 0)) > end:
                    continue
                value = row.get("metrics", {}).get("sglang_num_queue_reqs")
                if value is not None:
                    maximum = max(maximum or 0.0, float(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return maximum


def completed_result(experiment: dict[str, Any]) -> dict[str, Any] | None:
    """Accept a full legacy run or seal the first 450 terminal results."""

    full = _BASE_COMPLETED_RESULT(experiment)
    if full is not None:
        full["target_terminals"] = 500
        full["stopped_at_t450"] = False
        return full
    directory = run_dir(experiment)
    marker = directory / T450_RESULT_NAME
    if marker.is_file():
        return json.loads(marker.read_text())
    if (directory / "INVALID.md").is_file():
        return None
    rows = core.load_progress(directory)
    if len(rows) < TARGET_TERMINALS:
        return None
    starts = directory / "workload/measurement_start.json"
    if not starts.is_file():
        return None
    try:
        workload_start = float(
            json.loads(starts.read_text())["measurement_start_wall"]
        )
        ordered = sorted(rows, key=lambda row: float(row["completed_at"]))
        window = ordered[:TARGET_TERMINALS]
        timestamps = [float(row["completed_at"]) for row in window]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    t450_end = timestamps[-1]
    decode_tokens, decode_tps = _counter_window(
        directory,
        "decode",
        "sglang_realtime_tokens_total|mode=decode",
        workload_start,
        t450_end,
    )
    prefill_tokens, prefill_tps = _counter_window(
        directory,
        "prefill",
        "sglang_realtime_tokens_total|mode=prefill_compute",
        workload_start,
        t450_end,
    )
    # Wait for a complete per-endpoint Decode counter window before stopping.
    if decode_tokens is None or decode_tps is None:
        return None
    normal = sum(str(row.get("status")) == "Status.COMPLETED" for row in window)
    result = {
        "concurrency": experiment["concurrency"],
        "method": experiment["method"],
        "run_id": experiment["run_id"],
        "directory": str(directory),
        "completed": TARGET_TERMINALS,
        "normal_completed": normal,
        "failed": TARGET_TERMINALS - normal,
        "environment_errors": sum(
            bool(row.get("environment_error")) for row in window
        ),
        "solved": sum(float(row.get("reward") or 0) > 0 for row in window),
        "target_terminals": TARGET_TERMINALS,
        "stopped_at_t450": True,
        "t450_seconds": t450_end - workload_start,
        "t500_seconds": None,
        "t50_seconds": timestamps[49] - workload_start,
        "t250_seconds": timestamps[249] - workload_start,
        "t50_t450_agent_s": 400.0 / (timestamps[449] - timestamps[49]),
        "t450_decode_tokens": decode_tokens,
        "t450_decode_tps": decode_tps,
        "scheduler_prefill_tps": prefill_tps,
        "scheduler_decode_tps": decode_tps,
        "t450_prefill_tokens": prefill_tokens,
        "max_queue": _max_queue_until(directory, t450_end),
        "measurement_start_wall": workload_start,
        "t450_end_wall": t450_end,
        "observed_terminal_records_when_sealed": len(rows),
    }
    core.atomic_json(marker, result)
    return result


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
    """Validate the two isolated Qwen3.8 baseline correctness patches once."""
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
            "env", f"PYTHONPATH={BASELINE_ENGINE / 'python'}",
            "PD_GDN_CUDA_PROBE=1", BASELINE_PYTHON, "-m", "pytest", "-q",
            *tests,
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
            "--single-node-2p2d",
            "--d2p-host-gib-per-rank", "128",
            "--p2d-host-gib-per-rank", "64",
            "--d2p-direct-lanes", "2",
            "--p2d-direct-lanes", "1",
            "--host-lanes", "2",
            "--host-load-workers", "1",
            "--nixl-progress-threads", "2",
            "--forward-stream-priority", "-5",
            "--flashinfer-allreduce-max-tokens", "8192",
            # Keep room for Decode growth; the scheduler's atomic Mamba-slot
            # admission queues excess ingress rather than over-allocating it.
            "--decode-growth-tokens", "131072",
            "--launch-transport", "ssh",
        ]
        core.command([*common, "prepare"])
        config_path = directory / "multinode-v3.json"
        config = json.loads(config_path.read_text())
        config["mamba_full_memory_ratio"] = AGENTIC_MAMBA_FULL_MEMORY_RATIO
        config["prefill_mamba_full_memory_ratio"] = (
            AGENTIC_MAMBA_FULL_MEMORY_RATIO
        )
        config["decode_mamba_full_memory_ratio"] = (
            AGENTIC_MAMBA_FULL_MEMORY_RATIO
        )
        core.atomic_json(config_path, config)
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
        "--single-node-2p2d",
        "--prefill-groups-per-node", "2",
        "--decode-groups-per-node", "2",
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
            "--single-node-2p2d",
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
        return {23900, 23902, 23904, 23910, 24000, 24010}
    if experiment["method"] == "colocated":
        # Likewise, 24003 is metrics-only; 24002 is the request router.
        return {24000, 24002, 24010, 24020, 24030}
    ports = {24200, 24210, 24300, 24310, 24400, 24401}
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
    lines = [
        core.START_MARKER,
        "## 自动运行进度与结果",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}。",
        f"> 状态：`{state['status']}`；有效结果 "
        f"**{len(results)}/{len(EXPERIMENTS)}**。",
    ]
    if current:
        rows = core.load_progress(run_dir(current))
        records = len(rows) if rows else current_progress
        normal = sum(str(row.get("status")) == "Status.COMPLETED" for row in rows)
        env_errors = sum(bool(row.get("environment_error")) for row in rows)
        detail = f"（正常完成{normal}，environment error {env_errors}）" if rows else ""
        lines.append(
            f"> 当前：c{current['concurrency']} {METHOD_LABEL[current['method']]}，"
            f"terminal records **{records}/{TARGET_TERMINALS}**{detail}，"
            f"run `{current['run_id']}`。"
        )
    elif state["status"] == "complete":
        lines.append("> 9组T450实验均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend([
        "",
        "### 核心指标",
        "",
        "| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for experiment in EXPERIMENTS:
        result = result_by_run.get(experiment["run_id"])
        if result is None:
            failure = baseline_failure(experiment)
            if failure is not None:
                lines.append(
                    f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                    f"| 失败（{failure['normal_completed']}/{failure['terminal_records']}） "
                    "| — | — | — | — |"
                )
                continue
            lines.append(
                f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                "| — | — | — | — | — |"
            )
            continue
        tokens = result["t450_decode_tokens"]
        lines.append(
            "| {c} | {method} | {done}/{failed} | {solved}/{target} | **{t450}** | "
            "**{decode}** | {tokens} |".format(
                c=result["concurrency"], method=METHOD_LABEL[result["method"]],
                done=result["normal_completed"], failed=result["failed"],
                solved=result["solved"],
                target=result.get("target_terminals", 500),
                t450=core.value(result["t450_seconds"]),
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
    for experiment in EXPERIMENTS:
        result = result_by_run.get(experiment["run_id"])
        if result is None:
            failure = baseline_failure(experiment)
            if failure is not None:
                lines.append(
                    f"| {experiment['concurrency']} | {METHOD_LABEL[experiment['method']]} "
                    f"| — | — | — | — | — | — | — | {failure['summary']} |"
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
    lines.extend(["", "原始目录：", ""])
    for result in results:
        lines.append(
            f"- c{result['concurrency']} {METHOD_LABEL[result['method']]}: "
            f"`{result['directory']}`"
        )
    for experiment in EXPERIMENTS:
        failure = baseline_failure(experiment)
        if failure is not None:
            lines.append(
                f"- c{experiment['concurrency']} {METHOD_LABEL[experiment['method']]} "
                f"（失败审计）: `{run_dir(experiment)}`"
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
        "MIN_MEM_AVAILABLE_KIB": MIN_MEM_AVAILABLE_KIB,
        "MOONCAKE_COMPAT": MOONCAKE_COMPAT,
        "AGENTIC_LAUNCHER": AGENTIC_LAUNCHER,
        "COLOCATED_LAUNCHER": COLOCATED_LAUNCHER,
        "NATIVE_LAUNCHER": NATIVE_LAUNCHER,
        "METHOD_LABEL": METHOD_LABEL,
        "EXPERIMENTS": EXPERIMENTS,
        "ALL_EXPERIMENTS": EXPERIMENTS,
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
    core.completed_result = completed_result


def main() -> int:
    configure_core()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
