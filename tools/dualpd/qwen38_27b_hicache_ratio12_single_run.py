#!/usr/bin/env python3
"""Run one isolated Qwen3.8-27B HiCache c192 ratio=1.2 experiment.

This controller intentionally owns only node2701 inside the pre-existing
two-node allocation.  It never cancels or rebuilds that shared allocation.
The baseline implementation is unchanged; the only inference-setting delta
from the established c192 HiCache cell is mamba_full_memory_ratio=1.2.
"""

from __future__ import annotations

import sys
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
RUN_ID = (
    "q38-27b-h100-node2701-1n-2p2d-hicache-mooncake-"
    "swe500-c192-mambaratio12-20261018-r1"
)
CONTROL_DIR = RUNS / "q38-27b-hicache-ratio12-c192-node2701-20261018-r1"
STATE_PATH = CONTROL_DIR / "state.json"
LOCK_PATH = CONTROL_DIR / "controller.lock"
ALLOCATION_PATH = CONTROL_DIR / "allocation.json"
CONTROLLER_LOG = WORKSPACE / "q38_27b_hicache_ratio12_c192_node2701_controller.log"

JOB_ID = "26049949"
NODE = "node2701"
BASELINE_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/"
    "pd_baseline/bin/python3.12"
)
WORKLOAD_PYTHON = Path(
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12"
)
MODEL = Path("/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3.8-27B")
DATA = Path("/orcd/scratch/orcd/003/shangy/dualpd/datasets")
SIF = Path("/orcd/pool/004/shangy/dualpd/containers/swebench-verified-sif")
WORKLOAD = (
    ROOT
    / "examples/pd/configs/experiments/"
    "swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer_overlay.yaml"
)
HICACHE_ENGINE = RUNS.parent / "sglang-src/pd_baseline_b1ef6"
MOONCAKE_COMPAT = ROOT / "examples/pd/scripts/compat/qwen35_mooncake"
NATIVE_LAUNCHER = ROOT / "tools/dualpd/qwen35_native_pd_slurm.py"

EXPERIMENT = {"concurrency": 192, "method": "hicache", "run_id": RUN_ID}
EXPERIMENTS = [EXPERIMENT]


def _with_target_node_prepare(argv: list[str]) -> None:
    """Run native.prepare while selecting node2701 from the shared job."""

    original_allocation = native.site.allocation

    def target_allocation(job_id: str, *, min_nodes: int = 1):
        fields, nodes = original_allocation(job_id, min_nodes=min_nodes)
        if str(job_id) == JOB_ID and min_nodes == 1:
            if NODE not in nodes:
                raise RuntimeError(f"{NODE} is absent from allocation {JOB_ID}")
            return fields, [NODE]
        return fields, nodes

    old_argv = sys.argv
    native.site.allocation = target_allocation
    try:
        sys.argv = [str(NATIVE_LAUNCHER), *argv]
        result = native.main()
        if result:
            raise RuntimeError(f"native prepare exited with {result}")
    finally:
        sys.argv = old_argv
        native.site.allocation = original_allocation


def prepare_and_start(experiment: dict) -> None:
    directory = RUNS / experiment["run_id"]
    if directory.exists():
        raise RuntimeError(f"refusing to reuse run directory: {directory}")
    _with_target_node_prepare(
        [
            "prepare",
            "--job-id", JOB_ID,
            "--workload-job-id", JOB_ID,
            "--workload-node", NODE,
            "--storage-job-id", JOB_ID,
            "--storage-node", NODE,
            "--run-id", experiment["run_id"],
            "--run-dir", str(directory),
            "--model", str(MODEL),
            "--data-root", str(DATA),
            "--swe-sif-dir", str(SIF),
            "--workload-config", str(WORKLOAD),
            "--requests", "500",
            "--concurrency", "192",
            "--tp-size", "2",
            "--python", str(BASELINE_PYTHON),
            "--workload-python", str(WORKLOAD_PYTHON),
            "--sglang-root", str(HICACHE_ENGINE),
            "--mamba-full-memory-ratio", "1.2",
            "--single-node-2p2d",
            "--prefill-groups-per-node", "2",
            "--decode-groups-per-node", "2",
            "--launch-transport", "ssh",
            "--native-mooncake",
            "--mamba-cache-strategy-flag", "scheduler",
            "--mooncake-compat", str(MOONCAKE_COMPAT),
            "--prefill-hicache-size-gb", "32",
            "--decode-hicache-size-gb", "32",
            "--mamba-hicache-size-gb", "32",
            "--mooncake-store-size-gb", "256",
            "--mooncake-prefetch-policy", "wait_complete",
        ]
    )
    core.command(
        [
            BASELINE_PYTHON,
            NATIVE_LAUNCHER,
            "start",
            "--job-id", JOB_ID,
            "--run-dir", directory,
        ],
        timeout=3600,
    )


def stop_experiment(experiment: dict) -> None:
    core.command(
        [
            BASELINE_PYTHON,
            NATIVE_LAUNCHER,
            "stop",
            "--job-id", JOB_ID,
            "--run-dir", RUNS / experiment["run_id"],
        ],
        timeout=1800,
        check=False,
    )


def no_shared_allocation_rebuild(reason: str) -> None:
    raise core.DirtyAllocationError(
        "node2701 failed its hard zero gate, but allocation 26049949 is shared "
        f"with node1800 and must not be cancelled: {reason}"
    )


def update_local_state(_state: dict, _progress: int) -> None:
    # The root agent owns the shared Markdown result table.
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
        "MODEL": MODEL,
        "DATA": DATA,
        "SIF": SIF,
        "WORKLOAD": WORKLOAD,
        "BASELINE_ENGINE": HICACHE_ENGINE,
        "MOONCAKE_COMPAT": MOONCAKE_COMPAT,
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


def main() -> int:
    configure()
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
