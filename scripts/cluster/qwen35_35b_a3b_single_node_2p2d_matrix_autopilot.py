#!/usr/bin/env python3
"""Run the Qwen3.5-35B-A3B TP2 single-node 2P:2D SWE500 matrix.

The c64/c128/c192 runs schedule the three PD-separated methods and reuse the
matching colocated measurements.  The c256 extension schedules all four
methods, including a fresh colocated run requested for that concurrency.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

try:
    from . import qwen38_27b_single_node_2p2d_matrix_autopilot as base
    from . import qwen35_35b_a3b_single_node_1p3d_matrix_autopilot as metrics
except ImportError:  # Direct script execution.
    import qwen38_27b_single_node_2p2d_matrix_autopilot as base
    import qwen35_35b_a3b_single_node_1p3d_matrix_autopilot as metrics


CONTROL_DIR = (
    base.RUNS
    / "qwen35-35b-a3b-tp2-single-node-2p2d-c64c128c192-tuned-20261016"
)
REPORT = (
    base.ROOT
    / "runs-host/"
    "SWEBENCH_QWEN35_35B_A3B_TP2_SINGLE_NODE_2P2D_H100.md"
)
MODEL = base.RUNS.parent / "models/Qwen3.5-35B-A3B"
NODE = "node2800"
CONCURRENCIES = (64, 128, 192, 256)
METHODS = ("agentic", "native", "hicache")

METHOD_LABEL = {
    "agentic": "当前新方法（2P:2D，tuned）",
    "colocated": "Colocated（4×TP2）",
    "native": "No-reverse PD（2P:2D）",
    "hicache": "原生 HiCache/Mooncake（2P:2D）",
}


def run_name(concurrency: int, method: str) -> str:
    names = {
        "agentic": "a-ft30-l4l3",
        "colocated": "col",
        "native": "nr",
        "hicache": "hc",
    }
    # c256 agentic r1/r2 are retained as immutable failed artifacts: they
    # exposed missing request-row and Mamba-Radix checkpoint admission.  A
    # corrected retry needs a new run id so node-local launch-once records
    # cannot be mistaken for live services.
    revision = "r3" if concurrency == 256 and method == "agentic" else "r1"
    # multinode.NAME intentionally caps identifiers at 64 characters.
    return (
        f"q35a3b-2p2d-{names[method]}-swe500-c{concurrency}-"
        f"20261016-{revision}"
    )


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "concurrency": concurrency,
        "method": method,
        "run_id": run_name(concurrency, method),
    }
    for concurrency in (64, 128, 192)
    for method in METHODS
]
# Keep the existing c256 PD order stable so an in-flight controller can be
# restarted without changing index meanings, then append the requested fresh
# Colocated measurement as the final run.
EXPERIMENTS.extend(
    {
        "concurrency": 256,
        "method": method,
        "run_id": run_name(256, method),
    }
    for method in ("agentic", "hicache", "native", "colocated")
)

# Immutable colocated results from the matching Qwen3.5-35B-A3B 4xTP2 run.
# Colocated has no P:D split, so rerunning it for a 2P:2D comparison would be
# identical work rather than a new configuration.
REUSED_COLOCATED = {
    64: {
        "completed": 500, "failed": 0, "resolved": 273,
        "t450": 2668.89, "decode_tps": 2050.97, "decode_tokens": 5.474,
        "p_forward": 0.2817, "d_forward": 0.7057,
        "d_kv": 0.1168, "prefix_reuse": 0.9609,
    },
    128: {
        "completed": 500, "failed": 0, "resolved": 281,
        "t450": 2206.11, "decode_tps": 2476.64, "decode_tokens": 5.464,
        "p_forward": 0.4409, "d_forward": 0.5510,
        "d_kv": 0.2282, "prefix_reuse": 0.8845,
    },
    192: {
        "completed": 500, "failed": 0, "resolved": 282,
        "t450": 2217.44, "decode_tps": 2425.71, "decode_tokens": 5.379,
        "p_forward": 0.5393, "d_forward": 0.4473,
        "d_kv": 0.3239, "prefix_reuse": 0.8280,
    },
}

_BASE_PREPARE_AND_START = base.prepare_and_start
_BASE_UPDATE_REPORT = base.update_report


def prepare_and_start(experiment: dict[str, Any]) -> None:
    """Keep every tuned knob fixed and change only the P:D topology."""
    if experiment["method"] != "agentic":
        _BASE_PREPARE_AND_START(experiment)
        return
    directory = base.run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse run directory: {directory}")
    common = [
        base.PD_PYTHON, base.AGENTIC_LAUNCHER,
        *base.common_paths(experiment),
        "--prefill-node", base.NODE,
        "--decode-node", base.NODE,
        "--python", base.PD_PYTHON,
        "--sglang-root", base.PD_ENGINE,
        "--single-node-2p2d",
        # Four TP ranks per direction preserve the campaign-wide host totals:
        # D-to-P 4*128=512 GiB; P-to-D 4*64=256 GiB.
        "--d2p-host-gib-per-rank", "128",
        "--p2d-host-gib-per-rank", "64",
        "--d2p-direct-lanes", "4",
        "--p2d-direct-lanes", "3",
        "--host-lanes", "2",
        "--host-load-workers", "1",
        "--fast-tool-seconds", "30",
        "--nixl-progress-threads", "2",
        "--forward-stream-priority", "-5",
        "--flashinfer-allreduce-max-tokens", "8192",
        "--decode-growth-tokens", "131072",
        "--launch-transport", "ssh",
    ]
    base.core.command([*common, "prepare"])
    base.core.command([*common, "--detach", "start-control"])
    base.core.command([*common, "control-ready"], timeout=900)
    base.core.command([*common, "--detach", "start-prefill"], timeout=2400)
    base.core.command([*common, "--detach", "start-decode"], timeout=1200)
    base.core.command([*common, "wait-ready"], timeout=2400)
    base.core.command([*common, "--detach", "start-router"], timeout=600)
    base.core.command([*common, "--detach", "run-workload"], timeout=600)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _extra_report_lines(results: list[dict[str, Any]]) -> list[str]:
    by_key = {
        (int(result["concurrency"]), str(result["method"])): result
        for result in results
    }
    lines = [
        "",
        "### 复用的 Colocated 对照",
        "",
        "| 并发 | 方法 | 完成/失败 | Resolved | **T450 (s)** | **T450前 Decode tok/s** | T450前 Decode token |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for concurrency in sorted(REUSED_COLOCATED):
        row = REUSED_COLOCATED[concurrency]
        lines.append(
            f"| {concurrency} | {METHOD_LABEL['colocated']} | "
            f"{row['completed']}/{row['failed']} | {row['resolved']}/500 | "
            f"**{row['t450']:.2f}** | **{row['decode_tps']:.2f}** | "
            f"{row['decode_tokens']:.3f}M |"
        )
    lines.extend([
        "",
        "### T450窗口 P/D Forward、KV与复用",
        "",
        "| 并发 | 方法 | P Forward/卡 | P Attention KV/引擎 | D Forward/卡 | D/共享 Attention KV/引擎 | Prefix复用率 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for concurrency in CONCURRENCIES:
        methods = (
            ("agentic", "hicache", "native", "colocated")
            if concurrency == 256
            else ("agentic", "colocated", "native", "hicache")
        )
        for method in methods:
            if method == "colocated" and concurrency in REUSED_COLOCATED:
                row = REUSED_COLOCATED[concurrency]
                lines.append(
                    f"| {concurrency} | {METHOD_LABEL[method]} | "
                    f"{_percent(row['p_forward'])} | — | "
                    f"{_percent(row['d_forward'])} | {_percent(row['d_kv'])} | "
                    f"{_percent(row['prefix_reuse'])} |"
                )
                continue
            result = by_key.get((concurrency, method))
            if result is None:
                lines.append(
                    f"| {concurrency} | {METHOD_LABEL[method]} | — | — | — | — | — |"
                )
                continue
            resource = metrics._resource_metrics(result)
            p_kv = None if method == "colocated" else resource.get("p_kv")
            lines.append(
                f"| {concurrency} | {METHOD_LABEL[method]} | "
                f"{_percent(resource.get('p_forward'))} | "
                f"{_percent(p_kv)} | "
                f"{_percent(resource.get('d_forward'))} | "
                f"{_percent(resource.get('d_kv'))} | "
                f"{_percent(resource.get('prefix_reuse'))} |"
            )
    lines.extend([
        "",
        "Forward 为 workload 起点到 T450 的物理 GPU 平均 device-timer 占比；"
        "Attention KV 为同窗口内 `sglang_full_token_usage` 的时间加权均值。",
        "Prefix 复用率为 P 侧 `prefill_cache / (prefill_cache + prefill_compute)`。",
    ])
    return lines


def update_report(state: dict[str, Any], current_progress: int) -> None:
    """Render the live nine-run block, then add immutable comparisons."""
    _BASE_UPDATE_REPORT(state, current_progress)
    results = base.core.all_results()
    text = REPORT.read_text()
    text = text.replace("> 9组T450实验均已完成。", "> 13组新实验均已完成。")
    insertion = "\n".join(_extra_report_lines(results)) + "\n"
    text = text.replace(base.core.END_MARKER, insertion + base.core.END_MARKER, 1)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, REPORT)


def rebuild_allocation(reason: str) -> None:
    """Replace residue only on node2800 and synchronize all controller layers."""
    old_job_id = str(base.JOB_ID)
    base.core.log(
        f"hard gate failed ({reason}); cancelling node2800 allocation {old_job_id}"
    )
    base.core.command(["scancel", old_job_id], timeout=60, check=False)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        queued = base.core.command(
            ["squeue", "-h", "-j", old_job_id, "-o", "%T"],
            timeout=30, check=False,
        ).stdout.strip()
        if not queued:
            break
        time.sleep(5)
    else:
        raise RuntimeError(
            f"dirty allocation {old_job_id} did not terminate within 900s"
        )
    submitted = base.core.command(
        [
            "sbatch", "--parsable",
            "--job-name=q35-35b-a3b-2p2d-matrix",
            "--partition=pi_songhan", "--account=mit_general",
            "--nodes=1", "--ntasks-per-node=1", "--cpus-per-task=224",
            "--gres=gpu:h100:8", "--mem=0", "--exclusive",
            "--time=3-00:00:00", "--nodelist=node2800",
            f"--output={base.WORKSPACE}/slurm-q35-35b-a3b-2p2d-%j.out",
            '--wrap=trap "exit 0" TERM INT; while true; do sleep 300; done',
        ],
        timeout=60,
    )
    new_job_id = submitted.stdout.strip().splitlines()[-1].split(";", 1)[0]
    base.JOB_ID = new_job_id
    base.core.JOB_ID = new_job_id
    base.NODE = ""
    base.core.NODE = ""
    base.core.save_allocation("pending")
    new_node = base.core.wait_for_allocation(new_job_id, timeout=1800)
    if new_node != NODE:
        raise RuntimeError(f"replacement landed on forbidden node {new_node}")
    base.NODE = new_node
    base.core.NODE = new_node
    base.core.save_allocation("running")
    base.core.command(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            new_node, f"grep -q job_{new_job_id} /proc/self/cgroup",
        ],
        timeout=30,
    )
    base.core.assert_clean_for_start()
    base.core.log(
        f"replacement allocation {new_job_id} on node2800 passed hard zero gate"
    )


def configure() -> None:
    base.CONTROL_DIR = CONTROL_DIR
    base.STATE_PATH = CONTROL_DIR / "state.json"
    base.LOCK_PATH = CONTROL_DIR / "controller.lock"
    base.ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
    base.BASELINE_VALIDATION_PATH = CONTROL_DIR / "baseline-correctness-validation.json"
    base.REPORT = REPORT
    base.MODEL = MODEL
    base.NODE = NODE
    base.CONCURRENCIES = CONCURRENCIES
    base.METHODS = METHODS
    base.METHOD_LABEL = METHOD_LABEL
    base.EXPERIMENTS = EXPERIMENTS
    base.HICACHE_MAMBA_CACHE_STRATEGY_FLAG = "scheduler"
    base.prepare_and_start = prepare_and_start
    base.update_report = update_report
    base.rebuild_allocation = rebuild_allocation
    if base.ALLOCATION_PATH.is_file():
        allocation = json.loads(base.ALLOCATION_PATH.read_text())
        base.JOB_ID = str(allocation["job_id"])
        base.NODE = str(allocation.get("node") or NODE)


def main() -> int:
    configure()
    base.configure_core()
    return base.core.main()


if __name__ == "__main__":
    raise SystemExit(main())
