#!/usr/bin/env python3
"""Run the isolated Qwen3.8-27B c128 HiCache ratio=1.2 T450 cell."""

from __future__ import annotations

import json
from pathlib import Path

import qwen35_9b_h100_matrix_autopilot as core
import qwen38_27b_single_node_2p2d_matrix_autopilot as matrix


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
RUN_ID = (
    "q38-27b-h100-1n-2p2d-hicache-mooncake-swe500-"
    "c128-ratio12-20261018-r2"
)
CONTROL_DIR = RUNS / "q38-27b-hicache-ratio12-c128-node1800-r2-control"
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
RESULT_PATH = CONTROL_DIR / "result.json"
JOB_ID = "26049949"
NODE = "node1800"
MAMBA_RATIO = 1.2
EXPERIMENT = {"concurrency": 128, "method": "hicache", "run_id": RUN_ID}
EXPERIMENTS = [EXPERIMENT]


def run_dir(experiment):
    return RUNS / experiment["run_id"]


def prepare_and_start(experiment):
    directory = run_dir(experiment)
    if directory.exists():
        raise RuntimeError(f"refusing to reuse run directory: {directory}")
    prepare = [
        matrix.BASELINE_PYTHON,
        matrix.NATIVE_LAUNCHER,
        "prepare",
        "--job-id", JOB_ID,
        "--run-id", experiment["run_id"],
        "--run-dir", directory,
        "--model", matrix.MODEL,
        "--data-root", matrix.DATA,
        "--swe-sif-dir", matrix.SIF,
        "--workload-config", matrix.WORKLOAD,
        "--requests", "500",
        "--concurrency", "128",
        "--tp-size", "2",
        "--python", matrix.BASELINE_PYTHON,
        "--workload-python", matrix.PD_PYTHON,
        "--sglang-root", matrix.HICACHE_ENGINE,
        "--mamba-full-memory-ratio", str(MAMBA_RATIO),
        "--single-node-2p2d",
        "--prefill-groups-per-node", "2",
        "--decode-groups-per-node", "2",
        # pam_slurm_adopt rejects direct SSH for allocation 26049949 because
        # it was submitted from a different login session.  Bind every
        # component to the same allocation through overlapping Slurm steps;
        # this changes orchestration only, not the measured baseline.
        "--launch-transport", "srun",
        "--native-mooncake",
        "--mamba-cache-strategy-flag", matrix.HICACHE_MAMBA_CACHE_STRATEGY_FLAG,
        "--mooncake-compat", matrix.MOONCAKE_COMPAT,
        "--prefill-hicache-size-gb", "32",
        "--decode-hicache-size-gb", "32",
        "--mamba-hicache-size-gb", "32",
        "--mooncake-store-size-gb", "256",
        "--mooncake-prefetch-policy", "wait_complete",
        "--workload-job-id", JOB_ID,
        "--workload-node", NODE,
        "--storage-job-id", JOB_ID,
        "--storage-node", NODE,
    ]
    core.command(prepare)
    manifest = json.loads((directory / "native-pd-manifest.json").read_text())
    if manifest["settings"]["mamba_full_memory_ratio"] != MAMBA_RATIO:
        raise RuntimeError(f"prepared manifest did not preserve ratio={MAMBA_RATIO}")
    if set(manifest["nodes"].values()) != {NODE}:
        raise RuntimeError(f"prepared manifest escaped {NODE}: {manifest['nodes']}")
    core.command(
        [
            matrix.BASELINE_PYTHON,
            matrix.NATIVE_LAUNCHER,
            "start",
            "--job-id", JOB_ID,
            "--run-dir", directory,
        ],
        timeout=3600,
    )


def stop_experiment(experiment):
    core.command(
        [
            matrix.BASELINE_PYTHON,
            matrix.NATIVE_LAUNCHER,
            "stop",
            "--job-id", JOB_ID,
            "--run-dir", run_dir(experiment),
        ],
        timeout=1800,
        check=False,
    )


def cleanup_experiment(experiment):
    """Clean only node1800; never cancel the shared two-node allocation."""
    directory = run_dir(experiment)
    if directory.exists():
        stop_experiment(experiment)
    if core.scoped_process_count(experiment["run_id"]):
        core.terminate_scoped_processes(experiment["run_id"])
    if core.workload_process_count(experiment["run_id"]):
        raise core.DirtyAllocationError("run-scoped workload survived stop")
    helpers = core.sandbox_helper_counts()
    if any(helpers.values()):
        core.terminate_sandbox_helpers()
    core.wait_clean(experiment["run_id"])


def completed_result(experiment):
    return matrix.completed_result(experiment)


def update_report(state, current_progress):
    result = completed_result(EXPERIMENT) if run_dir(EXPERIMENT).exists() else None
    core.atomic_json(
        RESULT_PATH,
        {
            "state": state.get("status"),
            "progress": current_progress,
            "run_id": RUN_ID,
            "job_id": JOB_ID,
            "node": NODE,
            "mamba_full_memory_ratio": MAMBA_RATIO,
            "result": result,
        },
    )


def configure():
    assignments = {
        "RUNS": RUNS,
        "CONTROL_DIR": CONTROL_DIR,
        "STATE_PATH": STATE_PATH,
        "LOCK_PATH": LOCK_PATH,
        "ALLOCATION_PATH": ALLOCATION_PATH,
        "JOB_ID": JOB_ID,
        "NODE": NODE,
        "PD_PYTHON": matrix.PD_PYTHON,
        "BASELINE_PYTHON": matrix.BASELINE_PYTHON,
        "MODEL": matrix.MODEL,
        "DATA": matrix.DATA,
        "SIF": matrix.SIF,
        "WORKLOAD": matrix.WORKLOAD,
        "PD_ENGINE": matrix.PD_ENGINE,
        "BASELINE_ENGINE": matrix.HICACHE_ENGINE,
        "MOONCAKE_COMPAT": matrix.MOONCAKE_COMPAT,
        "NATIVE_LAUNCHER": matrix.NATIVE_LAUNCHER,
        "EXPERIMENTS": EXPERIMENTS,
        "ALL_EXPERIMENTS": EXPERIMENTS,
        "MIN_MEM_AVAILABLE_KIB": 128 * 1024 * 1024,
        "SANDBOX_HELPER_NAMES": tuple(
            dict.fromkeys((*core.SANDBOX_HELPER_NAMES, "faked-sysv"))
        ),
    }
    for name, value in assignments.items():
        setattr(core, name, value)
    core.prepare_and_start = prepare_and_start
    core.stop_experiment = stop_experiment
    core.cleanup_experiment = cleanup_experiment
    core.completed_result = completed_result
    core.update_report = update_report
    core.expected_runtime_ports = matrix.expected_runtime_ports


if __name__ == "__main__":
    configure()
    raise SystemExit(core.main())
