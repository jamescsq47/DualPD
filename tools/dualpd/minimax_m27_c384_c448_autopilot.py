#!/usr/bin/env python3
"""Run repaired c384 HiCache, then the requested MiniMax c448 sequence."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import minimax_m27_high_concurrency_autopilot as high


core = high.core
core.CONTROL_DIR = (
    core.RUNS / "minimax-m27-tp8-two-node-c384-c448-autopilot-r1"
)
core.STATE_PATH = core.CONTROL_DIR / "state.json"
core.LOCK_PATH = core.CONTROL_DIR / "controller.lock"


def run_name(concurrency: int, method: str, revision: str) -> str:
    label = {
        "agentic": "agentic",
        "native": "native-oneway-1p1d",
        "hicache": "hicache-mooncake-1p1d",
    }[method]
    return (
        f"minimax-m27-tp8ep8-two-node-{label}-swe500-c{concurrency}"
        f"-20261017-{revision}"
    )


# Keep this exact order: first replace the invalid c384 HiCache run, then run
# the three user-requested c448 methods.  Every directory is immutable.
core.EXPERIMENTS = [
    {
        "concurrency": 384,
        "method": "hicache",
        "run_id": run_name(384, "hicache", "r4"),
    },
    {
        "concurrency": 448,
        "method": "agentic",
        "run_id": run_name(448, "agentic", "r1"),
    },
    {
        "concurrency": 448,
        "method": "hicache",
        "run_id": run_name(448, "hicache", "r1"),
    },
    {
        "concurrency": 448,
        "method": "native",
        "run_id": run_name(448, "native", "r1"),
    },
]
core.SEED_RUNS = []


START_MARKER = "<!-- MINIMAX_M27_C384_C448_START -->"
END_MARKER = "<!-- MINIMAX_M27_C384_C448_END -->"


def update_report(state: dict[str, Any], current_progress: int) -> None:
    results = []
    for experiment in core.EXPERIMENTS:
        directory = core.run_dir(experiment)
        result = core.completed_result(
            experiment["concurrency"], experiment["method"], directory
        ) or core.t450_result(
            experiment["concurrency"], experiment["method"], directory
        )
        if result is not None:
            results.append(result)
    current = (
        core.EXPERIMENTS[state["index"]]
        if state["index"] < len(core.EXPERIMENTS)
        else None
    )
    lines = [
        START_MARKER,
        "## c384 修复重跑与 c448 自动实验",
        "",
        f"> 最后检查：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}；"
        f"状态：`{state['status']}`；正式结果 **{len(results)}/4**。",
    ]
    if current is not None:
        lines.append(
            f"> 当前：c{current['concurrency']} "
            f"{high.METHOD_LABEL[current['method']]}，episode progress "
            f"**{current_progress}/500**，run `{current['run_id']}`。"
        )
    elif state.get("status") == "complete":
        lines.append("> c384 HiCache 修复重跑及三组 c448 实验均已完成。")
    if state.get("error"):
        lines.append(f"> 暂停原因：`{state['error']}`")
    lines.extend(
        [
            "",
            "| 并发 | 方法 | 完成/失败 | Resolved | **T450** | 0–T450 Decode tok/s | 异常 |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        lines.append(
            "| {c} | {method} | {done}/{failed} | {solved}/500 | "
            "{t450} s | {decode} | env={env}, spawn={spawn} |".format(
                c=result["concurrency"],
                method=high.METHOD_LABEL[result["method"]],
                done=result["completed"],
                failed=result["failed"],
                solved=result["solved"],
                t450=core.value(result["t450_seconds"]),
                decode=core.value(result.get("t450_decode_tps")),
                env=result["environment_errors"],
                spawn=result.get("hidden_spawn_errors", 0),
            )
        )
    lines.extend([END_MARKER, ""])
    block = "\n".join(lines)
    text = core.REPORT.read_text()
    if START_MARKER in text and END_MARKER in text:
        before, rest = text.split(START_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    temporary = core.REPORT.with_suffix(core.REPORT.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, core.REPORT)


core.update_report = update_report


_wait_clean = core.wait_clean


def _cleanup_run_fuse_mounts(run_id: str) -> None:
    """Unmount only this completed run's task-owned rootfs mounts.

    The controller intentionally stops at T450, so up to 50 unfinished SWE
    sandboxes do not get their normal Python ``close()`` callback.  Their FUSE
    mounts otherwise survive the killed workload and consume the node's 1024
    mount slots in later matrix entries.
    """

    prefix = f"/var/tmp/shangy/pd-swe-apptainer/{run_id}/"
    cleanup = (
        "import subprocess,sys\n"
        "prefix=sys.argv[1]\n"
        "targets=[]\n"
        "for line in open('/proc/self/mountinfo'):\n"
        " p=line.split()\n"
        " if len(p)>4 and p[4].startswith(prefix) and p[4].endswith('/rootfs'):\n"
        "  targets.append(p[4])\n"
        "failed=[]\n"
        "for target in targets:\n"
        " r=subprocess.run(['fusermount3','-u',target],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)\n"
        " if r.returncode: failed.append((target,r.stderr.strip()))\n"
        "print('unmounted',len(targets)-len(failed),'failed',len(failed))\n"
        "if failed:\n"
        " print(failed[:3])\n"
        " raise SystemExit(1)\n"
    )
    core.command(
        [
            "srun",
            f"--jobid={core.JOB_ID}",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={core.NODE0}",
            "--cpus-per-task=1",
            "python3",
            "-c",
            cleanup,
            prefix,
        ],
        timeout=180,
    )


def wait_clean(run_id: str) -> None:
    _wait_clean(run_id)
    _cleanup_run_fuse_mounts(run_id)


core.wait_clean = wait_clean


if __name__ == "__main__":
    raise SystemExit(core.main())
