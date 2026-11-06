#!/usr/bin/env python3
"""Profile Qwen3-8B BrowseComp c512 new-method and native HiCache.

Both cases reuse the exact 4P:4D launch environments from the completed H100
matrix.  The only workload change is shortening the formal closed-loop window
from 1200 seconds to 100 seconds.  A Python shim places only model servers
under Nsight Systems and makes inference.py delimit that formal window with
the CUDA profiler API.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


REPO = Path("/home/shangy/dualpd")
SLIME = Path(__file__).resolve().parents[2]
PD = SLIME / "examples/pd"
LOWC_CONTROLLER = SLIME / "tools/dualpd/browsecomp_qwen3_8b_lowc_h100_matrix.py"
PYTHON_WRAPPER = SLIME / "tools/dualpd/nsys_python_wrapper.sh"
TRACE_CONVERTER = SLIME / "tools/dualpd/nsys_sqlite_to_chrome.py"
NSYS = Path("/orcd/software/core/001/pkg/cuda/13.0.1/bin/nsys")
ROOT = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "nsys-browsecomp-qwen3-8b-h100-4p4d-c512-w300-m100-20261025"
)
STATE = ROOT / "state.json"
PORTS = (
    27300, 27320, 27400, 27420, 28300, 28320, 28400, 28420,
    27301, 27321, 27401, 27421, 27510, 27520, 8750,
    61000, 61001, 61002, 61003, 57651, 57652, 57680, 57603, 57690,
)


def load_lowc_module() -> Any:
    spec = importlib.util.spec_from_file_location("browsecomp_qwen3_8b_lowc", LOWC_CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LOWC_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lowc = load_lowc_module()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def update_state(state: dict[str, Any], case: str, status: str, **extra: Any) -> None:
    state[case] = {**state.get(case, {}), "status": status, "updated_at": time.time(), **extra}
    atomic_json(STATE, state)


def replace_link(path: Path, target: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)


def prepare_shim(case_name: str, real_bin: Path) -> Path:
    shim = ROOT / "control" / f"{case_name}-bin"
    shim.mkdir(parents=True, exist_ok=True)
    replace_link(shim / "python", PYTHON_WRAPPER)
    for program in ("mooncake_master", "mooncake_client"):
        target = real_bin / program
        if target.exists():
            replace_link(shim / program, target)
    return shim


def case_environment(case_name: str, run_dir: Path) -> tuple[dict[str, str], list[str], Path]:
    mode = "new_method" if case_name == "new-method" else "native_mooncake"
    case = ("browsecomp8b4p4d", mode, 512)
    if mode == "new_method":
        env, command = lowc.modified_environment(case, run_dir)
        real_bin = lowc.base.PD_ENV / "bin"
    else:
        env, command = lowc.baseline_environment(case, run_dir)
        real_bin = lowc.base.BASELINE_ENV / "bin"
    shim = prepare_shim(case_name, real_bin)
    old_path = env.get("PATH", os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"))
    env.update(
        RUN_DIR=str(run_dir),
        MAX_INFLIGHT="512",
        WARMUP_SECONDS="300",
        MAX_WARMUP_SECONDS="420",
        MEASURE_SECONDS="100",
        PD_ENV_BIN=str(shim),
        PATH=f"{shim}:{real_bin}:{old_path}",
        NSYS_REAL_PYTHON=str(real_bin / "python"),
        NSYS_BIN=str(NSYS),
        NSYS_OUTPUT_DIR=str(run_dir / "nsys"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env, command, real_bin


def gpu_pids() -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid not in pids:
            pids.append(pid)
    return pids


def listening_ports() -> set[int]:
    result = subprocess.run(
        ["ss", "-H", "-ltn"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    )
    found: set[int] = set()
    for line in result.stdout.splitlines():
        local = line.split()[3] if len(line.split()) >= 4 else ""
        try:
            found.add(int(local.rsplit(":", 1)[1]))
        except (IndexError, ValueError):
            continue
    return found.intersection(PORTS)


def cleanup_gate(timeout: float = 360.0, reap_owned: bool = False) -> None:
    # Give launchers their normal dependency-ordered shutdown first.  After a
    # profiled case, nsys can exit at capture-stop and leave an SGLang child
    # re-parented to PID 1.  On an exclusive allocation it is safe to reap the
    # remaining GPU process groups, but never do this in the initial node gate.
    graceful_deadline = time.monotonic() + (min(timeout, 90.0) if reap_owned else timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = gpu_pids()
        ports = listening_ports()
        if not pids and not ports:
            return
        if reap_owned and time.monotonic() >= graceful_deadline:
            groups: set[int] = set()
            for pid in pids:
                try:
                    groups.add(os.getpgid(pid))
                except (ProcessLookupError, PermissionError):
                    continue
            for group in groups:
                try:
                    os.killpg(group, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(15)
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        time.sleep(2)
    raise RuntimeError(
        f"post-case cleanup gate failed: GPU PIDs={gpu_pids()}, ports={sorted(listening_ports())}"
    )


def gpu_health_gate() -> None:
    """Require every allocated H100 to create and execute a CUDA context."""
    code = """
import json
import torch
count = torch.cuda.device_count()
assert count == 8, f"expected 8 visible GPUs, found {count}"
devices = []
for index in range(count):
    value = torch.ones(1024, device=f"cuda:{index}")
    value.mul_(2)
    torch.cuda.synchronize(index)
    devices.append({"index": index, "name": torch.cuda.get_device_name(index)})
print(json.dumps(devices))
"""
    result = subprocess.run(
        [str(lowc.base.PD_ENV / "bin/python"), "-c", code],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    (ROOT / "gpu-health-gate.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"GPU health gate failed (exit {result.returncode})")


def validate_boundaries(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "closed_loop_boundaries.json"
    if not path.is_file():
        raise RuntimeError(f"missing {path}")
    value = json.loads(path.read_text())
    boundary_status = value.get("status", value.get("boundary_reason"))
    if boundary_status != "steady_churn":
        raise RuntimeError(f"capture did not reach steady churn: {boundary_status}")
    seconds = float(value.get("measurement_seconds", 0))
    if not 95 <= seconds <= 115:
        raise RuntimeError(f"unexpected formal measurement duration: {seconds:.3f}s")
    return value


def export_traces(case_name: str, run_dir: Path) -> dict[str, Any]:
    nsys_dir = run_dir / "nsys"
    reports = sorted(nsys_dir.glob("*.nsys-rep"))
    if len(reports) != 8:
        raise RuntimeError(f"expected 8 nsys reports, found {len(reports)} in {nsys_dir}")
    databases: list[Path] = []
    for report in reports:
        database = report.with_suffix(".sqlite")
        subprocess.run(
            [
                str(NSYS), "export", "--quiet=true", "--force-overwrite=true",
                "--type=sqlite", f"--output={database}", str(report),
            ],
            check=True,
        )
        databases.append(database)
        stats_file = report.with_suffix(".stats.txt")
        with stats_file.open("w") as output:
            subprocess.run(
                [
                    str(NSYS), "stats", "--force-export=false", "--format=csv",
                    "--report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
                    str(database),
                ],
                text=True, stdout=output, stderr=subprocess.STDOUT, check=True,
            )

    edge_json = ROOT / f"{case_name}-edge-trace.json"
    conversion = subprocess.run(
        [str(TRACE_CONVERTER), "--output", str(edge_json), *(str(path) for path in databases)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    subprocess.run(["gzip", "-9", "-kf", str(edge_json)], check=True)
    return {
        "reports": [str(path) for path in reports],
        "sqlite": [str(path) for path in databases],
        "edge_json": str(edge_json),
        "edge_json_gz": str(edge_json) + ".gz",
        "conversion": conversion.stdout.strip(),
        "edge_json_bytes": edge_json.stat().st_size,
    }


def write_readme(state: dict[str, Any]) -> None:
    lines = [
        "# BrowseComp Qwen3-8B H100 c512 Nsight Systems",
        "",
        "- 拓扑：单节点 8×H100，4P:4D，TP=1，c512。",
        "- 工作负载：BrowseComp 固定顺序，300 秒预热，100 秒正式稳态采集。",
        "- 方法：当前新方法与原生 Mooncake HiCache；其余参数复用已完成矩阵。",
        "- Edge：在 `edge://tracing/` 中打开 `*-edge-trace.json` 或其 gzip 版本。",
        "- 原始数据：每个方法目录下保留 8 份 `.nsys-rep`、SQLite 导出和 stats。",
        "",
        "| 方法 | 状态 | 稳态秒数 | Edge JSON |",
        "|---|---|---:|---|",
    ]
    for case in ("new-method", "hicache"):
        item = state.get(case, {})
        boundary = item.get("boundaries", {})
        trace = item.get("trace", {})
        lines.append(
            f"| {case} | {item.get('status', 'pending')} | "
            f"{float(boundary.get('measurement_seconds', 0)):.2f} | "
            f"`{trace.get('edge_json', '—')}` |"
        )
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")


def run_case(case_name: str, state: dict[str, Any]) -> None:
    run_dir = ROOT / case_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "nsys").mkdir(parents=True, exist_ok=True)
    env, command, _ = case_environment(case_name, run_dir)
    # A retry must not carry a stale return code/error into a later successful
    # state record.
    state[case_name] = {}
    update_state(
        state, case_name, "running", run_dir=str(run_dir), command=command,
        started_at=time.time(), node=os.uname().nodename,
    )
    with (run_dir / "controller.log").open("a") as log:
        log.write(f"[{time.strftime('%FT%TZ', time.gmtime())}] starting {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(command, cwd=PD, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        update_state(state, case_name, "failed", returncode=completed.returncode)
        raise RuntimeError(f"{case_name} launcher exited {completed.returncode}")
    cleanup_gate(reap_owned=True)
    boundaries = validate_boundaries(run_dir)
    trace = export_traces(case_name, run_dir)
    update_state(
        state, case_name, "complete", completed_at=time.time(),
        boundaries=boundaries, trace=trace,
    )
    write_readme(state)


def main() -> None:
    required = [LOWC_CONTROLLER, PYTHON_WRAPPER, TRACE_CONVERTER, NSYS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing inputs: " + ", ".join(missing))
    ROOT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    atomic_json(ROOT / "configuration.json", {
        "model": str(lowc.MODEL), "workload": "BrowseComp", "concurrency": 512,
        "topology": "4P:4D TP=1", "warmup_seconds": 300,
        "measurement_seconds": 100, "nsys": str(NSYS),
        "cases": ["new-method", "hicache"],
    })
    (ROOT / "FAILED").unlink(missing_ok=True)
    cleanup_gate(timeout=30)
    gpu_health_gate()
    for case_name in ("new-method", "hicache"):
        if state.get(case_name, {}).get("status") == "complete":
            continue
        run_case(case_name, state)
    cleanup_gate()
    write_readme(state)
    (ROOT / "FAILED").unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / "FAILED").write_text(f"{type(error).__name__}: {error}\n")
        raise
