#!/usr/bin/env python3
"""Run one isolated Qwen3.5-35B-A3B D-to-P restore tuning trial.

The trial preserves the production c128 1P:3D configuration and changes only
the D-to-P admission knobs under test.  It stops as soon as the 450th terminal
episode is durable because T450 and the matching Decode counter window are the
decision metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

try:
    from . import qwen35_35b_a3b_single_node_1p3d_matrix_autopilot as matrix
except ImportError:
    import qwen35_35b_a3b_single_node_1p3d_matrix_autopilot as matrix


RUN_ID = "q35a3b-c128-d2pfix-ft30-p2dl3-d2pl4-r7"
RUN_DIR = matrix.base.RUNS / RUN_ID
CONTROL_DIR = matrix.base.RUNS / "q35-35b-a3b-d2p-restore-ft30-p2dl3-d2pl4-r7-20261016"
STATE_PATH = CONTROL_DIR / "state.json"
RESULT_PATH = CONTROL_DIR / "t450-result.json"
JOB_ID = "26071916"
TARGET_DECODE_TPS = 1861.40
POLL_SECONDS = 60
STALL_SECONDS = 5400

EXPERIMENT = {"concurrency": 128, "method": "agentic", "run_id": RUN_ID}


def save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def recycle_allocation(reason: str) -> str:
    """Use the Slurm cgroup as teardown, then recreate a clean short hold."""

    old_job_id = str(matrix.base.JOB_ID)
    matrix.base.core.log(
        f"recycling allocation {old_job_id} after {reason}"
    )
    matrix.base.core.command(["scancel", old_job_id], timeout=60, check=False)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        queued = matrix.base.core.command(
            ["squeue", "-h", "-j", old_job_id, "-o", "%T"],
            timeout=30,
            check=False,
        ).stdout.strip()
        if not queued:
            break
        time.sleep(5)
    else:
        raise RuntimeError(
            f"allocation {old_job_id} did not terminate within 900s"
        )
    submitted = matrix.base.core.command(
        [
            "sbatch",
            "--parsable",
            "--job-name=q35-d2p-after-r7",
            "--partition=pi_songhan",
            "--account=mit_general",
            "--nodes=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=224",
            "--gres=gpu:h100:8",
            "--mem=0",
            "--exclusive",
            # A longer request crosses the 2026-10-20 maintenance reservation
            # and is held until after maintenance even while node2800 is idle.
            "--time=3-00:00:00",
            "--nodelist=node2800",
            f"--output={matrix.base.WORKSPACE}/slurm-q35-d2p-after-r7-%j.out",
            '--wrap=trap "exit 0" TERM INT; while true; do sleep 300; done',
        ],
        timeout=60,
    )
    new_job_id = submitted.stdout.strip().splitlines()[-1].split(";", 1)[0]
    matrix.JOB_ID = new_job_id
    matrix.base.JOB_ID = new_job_id
    matrix.base.core.JOB_ID = new_job_id
    new_node = matrix.base.core.wait_for_allocation(new_job_id, timeout=1800)
    if new_node != matrix.NODE:
        raise RuntimeError(
            f"replacement landed on forbidden node {new_node}"
        )
    matrix.base.core.assert_clean_for_start()
    matrix.base.core.log(
        f"replacement allocation {new_job_id} on {new_node} passed hard zero gate"
    )
    return new_job_id


def common() -> list[str | Path]:
    return [
        matrix.base.PD_PYTHON,
        matrix.base.AGENTIC_LAUNCHER,
        "--job-id", matrix.base.JOB_ID,
        "--run-id", RUN_ID,
        "--run-dir", RUN_DIR,
        "--model", matrix.MODEL,
        "--data-root", matrix.base.DATA,
        "--swe-sif-dir", matrix.base.SIF,
        "--workload-config", matrix.base.WORKLOAD,
        "--requests", "500",
        "--concurrency", "128",
        "--tp-size", "2",
        "--prefill-node", matrix.NODE,
        "--decode-node", matrix.NODE,
        "--python", matrix.base.PD_PYTHON,
        "--sglang-root", matrix.base.PD_ENGINE,
        "--single-node-1p3d",
        "--d2p-host-gib-per-rank", "85",
        "--p2d-host-gib-per-rank", "128",
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


def start() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to reuse {RUN_DIR}")
    matrix.base.core.assert_clean_for_start()
    argv = common()
    matrix.base.core.command([*argv, "prepare"])
    matrix.base.core.command([*argv, "--detach", "start-control"])
    matrix.base.core.command([*argv, "control-ready"], timeout=900)
    matrix.base.core.command([*argv, "--detach", "start-prefill"], timeout=2400)
    matrix.base.core.command([*argv, "--detach", "start-decode"], timeout=1200)
    matrix.base.core.command([*argv, "wait-ready"], timeout=2400)
    matrix.base.core.command([*argv, "--detach", "start-router"], timeout=600)
    matrix.base.core.command([*argv, "--detach", "run-workload"], timeout=600)


def result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = sorted(
        float(row["completed_at"])
        for row in rows
        if row.get("completed_at") is not None
    )
    measurement = json.loads(
        (RUN_DIR / "workload/measurement_start.json").read_text()
    )
    start_wall = float(measurement["measurement_start_wall"])
    t450_end = timestamps[449]
    tokens, decode_tps = matrix.base.core.t450_decode_metrics(
        RUN_DIR, start_wall, t450_end
    )
    value = {
        "run_id": RUN_ID,
        "terminal_records_at_stop": len(rows),
        "normal_completed_at_stop": sum(
            str(row.get("status")) == "Status.COMPLETED" for row in rows
        ),
        "failed_at_stop": sum(
            str(row.get("status")) != "Status.COMPLETED" for row in rows
        ),
        "resolved_at_stop": sum(float(row.get("reward") or 0) > 0 for row in rows),
        "environment_errors_at_stop": sum(
            bool(row.get("environment_error")) for row in rows
        ),
        "t50_seconds": timestamps[49] - start_wall,
        "t250_seconds": timestamps[249] - start_wall,
        "t450_seconds": t450_end - start_wall,
        "t50_t450_agent_s": 400.0 / (timestamps[449] - timestamps[49]),
        "t450_decode_tokens": tokens,
        "t450_decode_tps": decode_tps,
        "hicache_c128_target_decode_tps": TARGET_DECODE_TPS,
        "target_ratio": (
            None if decode_tps is None else float(decode_tps) / TARGET_DECODE_TPS
        ),
        "meets_hicache_target": (
            decode_tps is not None and float(decode_tps) >= TARGET_DECODE_TPS
        ),
        "configuration": {
            "host_lanes": 2,
            "host_load_workers": 1,
            "fast_tool_seconds": 30.0,
            "d2p_direct_lanes": 4,
            "p2d_direct_lanes": 3,
        },
    }
    resource_input = {
        "run_id": RUN_ID,
        "directory": str(RUN_DIR),
        "method": "agentic",
        "t450_seconds": value["t450_seconds"],
    }
    value["resource_metrics"] = matrix._resource_metrics(resource_input)
    return value


def main() -> int:
    matrix.configure()
    # ``matrix.configure`` specializes the 35B constants on the shared 27B
    # controller module; propagate those values and hooks into the core before
    # invoking any admission, health, or cleanup operation.
    matrix.base.configure_core()
    # The previous allocation was intentionally cancelled after it became
    # impossible to fork a cleanup process under Host+sandbox commit pressure.
    # Specialize every controller layer to the clean replacement allocation.
    matrix.JOB_ID = JOB_ID
    matrix.base.JOB_ID = JOB_ID
    matrix.base.core.JOB_ID = JOB_ID
    state: dict[str, Any] = {
        "status": "starting",
        "run_id": RUN_ID,
        "progress": 0,
        "started_at_epoch": time.time(),
        "last_progress_epoch": time.time(),
        "error": "",
    }
    save(STATE_PATH, state)
    try:
        allocated_node = matrix.base.core.wait_for_allocation(
            JOB_ID, timeout=3600
        )
        if allocated_node != matrix.NODE:
            raise RuntimeError(
                f"allocation landed on forbidden node {allocated_node}"
            )
        start()
        state["status"] = "running"
        save(STATE_PATH, state)
        previous = 0
        while True:
            rows = matrix.base.core.load_progress(RUN_DIR)
            now = time.time()
            if len(rows) > previous:
                previous = len(rows)
                state["last_progress_epoch"] = now
            state.update(progress=len(rows), last_poll_epoch=now)
            save(STATE_PATH, state)
            if len(rows) >= 450:
                measured = result(rows)
                save(RESULT_PATH, measured)
                state.update(
                    status="stopping-after-t450",
                    t450_decode_tps=measured["t450_decode_tps"],
                    meets_hicache_target=measured["meets_hicache_target"],
                )
                save(STATE_PATH, state)
                # At this Host capacity the live data plane remains healthy,
                # but late-run commit pressure can prevent ssh/slurm from
                # forking the ordinary multi-command cleanup sequence.  The
                # allocation cgroup is the authoritative teardown boundary:
                # recycle it after the immutable T450 result is on shared
                # storage, then admit future work only after the replacement
                # allocation passes the normal hard-zero gate.
                replacement_job_id = recycle_allocation(
                    "c128 ft30 reached T450"
                )
                state.update(
                    status="complete",
                    stopped_at_epoch=time.time(),
                    replacement_job_id=replacement_job_id,
                )
                save(STATE_PATH, state)
                print(json.dumps(measured, indent=2, sort_keys=True), flush=True)
                return 0
            if now - float(state["last_progress_epoch"]) > STALL_SECONDS:
                raise RuntimeError(f"no terminal progress for {STALL_SECONDS}s")
            # Do not fork per-minute ssh/srun probes after startup.  The fixed
            # 768 GiB registered Host arenas plus c128 sandboxes can exhaust
            # Linux's heuristic commit budget while the already-running data
            # plane still has ample physical RAM and makes terminal progress.
            # Progress is the liveness signal; the watchdog alerts after 20m
            # without a new durable terminal, and the 90m controller bound is
            # the final fail-safe.
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] "
                f"{RUN_ID}: {len(rows)}/450",
                flush=True,
            )
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        state.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            failed_at_epoch=time.time(),
        )
        save(STATE_PATH, state)
        try:
            replacement_job_id = recycle_allocation(
                f"c128 ft30 controller failure: {type(error).__name__}: {error}"
            )
            state["replacement_job_id"] = replacement_job_id
            save(STATE_PATH, state)
        except BaseException as cleanup_error:
            state["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            save(STATE_PATH, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
