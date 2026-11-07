#!/usr/bin/env python3
"""Control one isolated upstream-SGLang Qwen3.8-27B HiCache c192 run."""

from __future__ import annotations

import subprocess
import os
from pathlib import Path

try:
    from . import qwen35_9b_h100_matrix_autopilot as core
    from . import qwen35_native_pd_slurm as native
    from . import qwen38_27b_single_node_2p2d_matrix_autopilot as q38
except ImportError:
    import qwen35_9b_h100_matrix_autopilot as core
    import qwen35_native_pd_slurm as native
    import qwen38_27b_single_node_2p2d_matrix_autopilot as q38


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
RUNS = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs")
RUN_ID = os.environ.get(
    "Q38_UPSTREAM_RUN_ID",
    "q38-27b-h100-node2701-1n-2p2d-hicache-mooncake-"
    "upstream-main-617c8b6-swe500-c192-20261019-r3",
)
RUN_DIR = RUNS / RUN_ID
CONTROL_DIR = Path(
    os.environ.get(
        "Q38_UPSTREAM_CONTROL_DIR",
        str(RUNS / "q38-27b-hicache-upstream617c-c192-node2701-20261019-r3"),
    )
)
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
CONTROLLER_LOG = Path(
    os.environ.get(
        "Q38_UPSTREAM_CONTROLLER_LOG",
        str(WORKSPACE / "q38_27b_hicache_upstream617c_c192_node2701_r3_controller.log"),
    )
)

JOB_ID = os.environ.get("Q38_UPSTREAM_JOB_ID", "26350611")
NODE = os.environ.get("Q38_UPSTREAM_NODE", "node2701")
CONCURRENCY = int(os.environ.get("Q38_UPSTREAM_CONCURRENCY", "192"))
BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd_baseline/bin/python3.12"
)
NATIVE_LAUNCHER = ROOT / "scripts/cluster/qwen35_native_pd_slurm.py"
UPSTREAM_ENGINE = RUNS.parent / "sglang-src/upstream_617c8b67"
EXPERIMENT = {"concurrency": CONCURRENCY, "method": "hicache", "run_id": RUN_ID}
EXPERIMENTS = [EXPERIMENT]


def prepare_and_start(_experiment: dict) -> None:
    manifest = RUN_DIR / "native-pd-manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"prepared upstream manifest is absent: {manifest}")
    core.command(
        [
            BASELINE_PYTHON,
            NATIVE_LAUNCHER,
            "start",
            "--job-id",
            JOB_ID,
            "--run-dir",
            RUN_DIR,
        ],
        timeout=7200,
    )


def stop_experiment(_experiment: dict) -> None:
    core.command(
        [
            BASELINE_PYTHON,
            NATIVE_LAUNCHER,
            "stop",
            "--job-id",
            JOB_ID,
            "--run-dir",
            RUN_DIR,
        ],
        timeout=1800,
        check=False,
    )


def no_shared_allocation_rebuild(reason: str) -> None:
    raise core.DirtyAllocationError(
        f"{NODE} is in job {JOB_ID}; refusing automatic allocation cancellation: {reason}"
    )


def authoritative_job_seconds_left() -> int:
    """Use squeue TIME_LEFT because this Slurm prints EndTime in site-local time."""
    value = subprocess.check_output(
        ["squeue", "-h", "-j", JOB_ID, "-o", "%L"], text=True
    ).strip()
    if not value:
        raise RuntimeError(f"Slurm job {JOB_ID} is absent")
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    else:
        raise RuntimeError(f"unrecognized TIME_LEFT: {value!r}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def update_local_state(_state: dict, _progress: int) -> None:
    return


def configure() -> None:
    values = {
        "WORKSPACE": WORKSPACE,
        "RUNS": RUNS,
        "CONTROL_DIR": CONTROL_DIR,
        "STATE_PATH": STATE_PATH,
        "LOCK_PATH": LOCK_PATH,
        "ALLOCATION_PATH": ALLOCATION_PATH,
        "JOB_ID": JOB_ID,
        "NODE": NODE,
        "BASELINE_PYTHON": BASELINE_PYTHON,
        "BASELINE_ENGINE": UPSTREAM_ENGINE,
        "MOONCAKE_COMPAT": None,
        "NATIVE_LAUNCHER": NATIVE_LAUNCHER,
        "EXPERIMENTS": EXPERIMENTS,
        "ALL_EXPERIMENTS": EXPERIMENTS,
        "MIN_MEM_AVAILABLE_KIB": 128 * 1024 * 1024,
        "NO_PROGRESS_TIMEOUT_SECONDS": 5400,
        "SANDBOX_HELPER_NAMES": tuple(
            dict.fromkeys((*core.SANDBOX_HELPER_NAMES, "faked-sysv"))
        ),
    }
    for name, value in values.items():
        setattr(core, name, value)
    core.prepare_and_start = prepare_and_start
    core.stop_experiment = stop_experiment
    core.expected_runtime_ports = q38.expected_runtime_ports
    core.completed_result = q38.completed_result
    core.update_report = update_local_state
    core.rebuild_allocation = no_shared_allocation_rebuild
    core.job_seconds_left = authoritative_job_seconds_left


def main() -> int:
    configure()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
