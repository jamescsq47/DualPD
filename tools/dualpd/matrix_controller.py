#!/usr/bin/env python3
"""Fail-closed serial controller for the 2026-10-07 H100 matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time


SLIME = Path(__file__).resolve().parents[2]
REPO = SLIME.parent
PD = SLIME / "examples/pd"
SCRATCH = Path("/orcd/scratch/orcd/003/shangy/dualpd")
RUNS = SCRATCH / "runs"
CONTROL = RUNS / "h100-24-matrix-20261007"
PD_ENV = Path("/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd")
BASELINE_ENV = Path("/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd_baseline")
MODIFIED_SGLANG = REPO / "sglang_pd/python"
MODEL_8B = SCRATCH / "models/Qwen3-8B"
MODEL_32B = Path("/orcd/compute/songhan/001/home/shangy/models/Qwen3-32B-9216db57")
DATA = SCRATCH / "datasets"


CASES = []
for workload, concurrencies in (("browsecomp32b", (192, 256, 320)), ("mixed8b", (384, 512, 640))):
    for mode in ("new_method", "colocated", "no_reverse", "native_mooncake"):
        for concurrency in concurrencies:
            CASES.append((workload, mode, concurrency))


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def case_key(case: tuple[str, str, int]) -> str:
    workload, mode, concurrency = case
    return f"{workload}-{mode}-c{concurrency}"


def run_dir(case: tuple[str, str, int]) -> Path:
    workload, mode, concurrency = case
    prefix = (
        "browsecomp-qwen3-32b-tp2-h100"
        if workload == "browsecomp32b"
        else "mixed1to1-qwen3-8b-h100"
    )
    return RUNS / f"{prefix}-{mode.replace('_', '-')}-c{concurrency}-w300-m1200-20261007-r3"


def clean_environment() -> dict[str, str]:
    keep = {
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "HF_TOKEN",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.setdefault("HOME", "/home/shangy")
    env.setdefault("LANG", "C.UTF-8")
    env["HF_HOME"] = str(DATA / "huggingface-cache")
    return env


def common_environment(case: tuple[str, str, int], directory: Path) -> dict[str, str]:
    workload, _, concurrency = case
    env = clean_environment()
    is_32b = workload == "browsecomp32b"
    env.update(
        MODEL_PATH=str(MODEL_32B if is_32b else MODEL_8B),
        PD_DATA_ROOT=str(DATA),
        MATH_DATA=str(DATA / "dapo-math-17k/dapo-math-17k.jsonl"),
        QA_DATA=str(DATA / "browsecomp/bc_train.jsonl"),
        WORKLOAD_CONFIG="",
        RUN_DIR=str(directory),
        TEMPERATURE="0",
        TOP_P="1",
        TOP_K="-1",
        REQUESTS="680" if is_32b else "8192",
        MAX_INFLIGHT=str(concurrency),
        MATH_RATIO="0" if is_32b else "0.5",
        SCHEDULE_FILE=str(
            PD / "configs/workloads/fixed_browsecomp_source_order_n680.json"
            if is_32b
            else CONTROL / "fixed_random_s2026_n8192.json"
        ),
        PRESERVE_SOURCE_ORDER="true" if is_32b else "false",
        DISPATCH_POLICY="fixed",
        SEED="2026",
        WARMUP_SECONDS="300",
        MAX_WARMUP_SECONDS="420",
        MEASURE_SECONDS="1200",
        CLOSED_LOOP="true",
        WARMUP_REQUESTS="0",
        MODEL_CONTEXT_LENGTH="40960",
        MODEL_MAX_RESPONSE_LENGTH="36864",
        SEARCH_GPU="7",
        SEARCH_START_AFTER_MODELS="true",
        SEARCH_SERVER_EMBEDDING_CACHE=os.environ.get(
            "SEARCH_SERVER_EMBEDDING_CACHE",
            str(next((DATA / "huggingface-cache/hub/datasets--miaolu3--browsecomp-plus/snapshots").glob(
                "*/corpus_embeddings.pkl"
            ), PD / "data/browsecomp/artifacts/search/corpus_embeddings.pkl")),
        ),
        PD_INFERENCE_RETURN_LOGPROB="false",
        MEM_FRACTION_STATIC="0.80",
        MAX_EXISTING_GPU_MEMORY_MB="1024",
        PD_IDLE_GPU_MAX_MIB="1024",
        SLIME_HTTP_READ_TIMEOUT_SECONDS="3600",
        POST_ANALYZER="pd_offload",
    )
    return env


def modified_environment(case: tuple[str, str, int], directory: Path) -> tuple[dict[str, str], list[str]]:
    workload, _, _ = case
    env = common_environment(case, directory)
    env_bin = PD_ENV / "bin"
    cuda_lib = PD_ENV / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    env.update(
        PD_ENV_BIN=str(env_bin),
        PATH=f"{env_bin}:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        PYTHONPATH=f"{MODIFIED_SGLANG}:{PD}:{REPO}",
        LD_LIBRARY_PATH=f"{cuda_lib}:{PD_ENV / 'lib'}",
        FAST_TOOL_THRESHOLD_SECONDS="1",
        DIRECT_WAIT_SECONDS="1",
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE="true",
        SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE="false",
        SGLANG_AGENTIC_KV_FORCE_SLOW_PATH="false",
        SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE="false",
        SGLANG_PD_ABLATION_P2D_PREBIND="false",
        SGLANG_PD_ABLATION_RANDOM_ROUTING="false",
        D_TARGET_KV_FRACTION="1.0",
        P2D_SPILL_DELAY_SECONDS="0.5",
        P2D_HOST_STAGING="true",
        # The user-specified capacities are totals across the two physical
        # Prefill workers/ranks in both matrices: D->P 512 GiB and P->D
        # 256 GiB.  Both launchers interpret these settings per physical P.
        P2D_HOST_ARENA_GIB_PER_P="128",
        SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB="128",
        SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB="256",
        SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA="1",
        SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA="1",
        SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER="1",
        SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB="8",
        SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB="768",
        SGLANG_AGENTIC_KV_D2H_INFLIGHT="2",
        P_H2D_MAX_INFLIGHT="2",
        SGLANG_AGENTIC_KV_P2D_D2H_WORKERS="2",
        SGLANG_AGENTIC_KV_P2D_H2D_WORKERS="2",
        SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS="4096",
        SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS="4096",
    )
    if workload == "browsecomp32b":
        env.update(
            PREFILL_GPU_GROUPS="0,4",
            DECODE_GPU_GROUPS="1,5;2,6;3,7",
            PREFILL_TP_SIZE="2",
            DECODE_TP_SIZE="2",
            PREFILL_PORTS="31340",
            BOOTSTRAP_PORTS="29340",
            DECODE_PORTS="31341 31342 31343",
            DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.60",
            ROUTER_PORT="31350",
            ROUTER_PROMETHEUS_PORT="31360",
            SEARCH_PORT="8750",
            SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH="32",
            SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW="8",
        )
        script = PD / "scripts/new_method/run_qwen3_32b_tp2_browsecomp_2p6d.sh"
    else:
        env.update(
            PREFILL_GPUS="0 4",
            DECODE_GPUS="1 2 3 5 6 7",
            PREFILL_PORTS="27300 27400",
            BOOTSTRAP_PORTS="28300 28400",
            DECODE_PORTS="27301 27302 27303 27401 27402 27403",
            DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80 0.80 0.60",
            ROUTER_PORT="27310",
            ROUTER_PROMETHEUS_PORT="27320",
            SEARCH_PORT="8730",
            SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH="16",
            SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW="4",
        )
        script = PD / "scripts/new_method/run_2p6d_numa_case.sh"
    return env, ["bash", str(script)]


def baseline_environment(case: tuple[str, str, int], directory: Path) -> tuple[dict[str, str], list[str]]:
    workload, mode, _ = case
    env = common_environment(case, directory)
    env_bin = BASELINE_ENV / "bin"
    cuda_lib = BASELINE_ENV / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    env.update(
        PD_ENV_BIN=str(env_bin),
        PATH=f"{env_bin}:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        PYTHONPATH="",
        LD_LIBRARY_PATH=f"{cuda_lib}:{BASELINE_ENV / 'lib'}",
        UCX_LOG_LEVEL="info",
    )
    if mode == "colocated":
        if workload == "browsecomp32b":
            env.update(
                MODEL_GPU_GROUPS="0,4;1,5;2,6;3,7",
                MODEL_TP_SIZE="2",
                MODEL_PORTS="33400 33401 33402 33403",
                MODEL_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.60",
                ROUTER_PORT="33410",
                SEARCH_PORT="9340",
            )
        else:
            env.update(
                MODEL_GPUS="0 1 2 3 4 5 6 7",
                MODEL_TP_SIZE="1",
                MODEL_PORTS="32100 32101 32102 32103 32104 32105 32106 32107",
                MODEL_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80 0.80 0.80 0.80 0.60",
                ROUTER_PORT="32110",
                SEARCH_PORT="9210",
            )
        return env, ["bash", str(PD / "scripts/baseline/run_colocated_case.sh")]

    env["CASE_MODE"] = mode
    if workload == "browsecomp32b":
        env.update(
            PREFILL_GPU_GROUPS="0,4",
            DECODE_GPU_GROUPS="1,5;2,6;3,7",
            PREFILL_TP_SIZE="2",
            DECODE_TP_SIZE="2",
            PREFILL_PORTS="30400",
            PREFILL_BOOTSTRAP_PORTS="28400",
            DECODE_PORTS="30401 30402 30403",
            PREFILL_MEM_FRACTION_STATICS="0.80",
            DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.60",
            ROUTER_PORT="30410",
            ROUTER_PROMETHEUS_PORT="30420",
            SEARCH_PORT="9360",
            P_HICACHE_SIZE="64",
            D_HICACHE_SIZE="56",
            MOONCAKE_MASTER_PORT="29451",
            MOONCAKE_CLIENT_PORT="29452",
            MOONCAKE_METADATA_PORT="29480",
            MOONCAKE_METRICS_PORT="29403",
            MOONCAKE_CLIENT_HTTP_PORT="29490",
        )
    else:
        env.update(
            PREFILL_GPUS="0 1",
            DECODE_GPUS="2 3 4 5 6 7",
            PREFILL_TP_SIZE="1",
            DECODE_TP_SIZE="1",
            PREFILL_PORTS="23100 23101",
            PREFILL_BOOTSTRAP_PORTS="24100 24101",
            DECODE_PORTS="23102 23103 23104 23105 23106 23107",
            PREFILL_MEM_FRACTION_STATICS="0.80 0.80",
            DECODE_MEM_FRACTION_STATICS="0.80 0.80 0.80 0.80 0.80 0.60",
            ROUTER_PORT="23110",
            ROUTER_PROMETHEUS_PORT="23120",
            SEARCH_PORT="8730",
            P_HICACHE_SIZE="128",
            D_HICACHE_SIZE="56",
            MOONCAKE_MASTER_PORT="25151",
            MOONCAKE_CLIENT_PORT="25152",
            MOONCAKE_METADATA_PORT="25180",
            MOONCAKE_METRICS_PORT="25103",
            MOONCAKE_CLIENT_HTTP_PORT="25190",
            TP1_LOCAL_NUMA="false",
        )
    return env, ["bash", str(PD / "scripts/baseline/run_pd_case.sh")]


def case_environment(case: tuple[str, str, int], directory: Path) -> tuple[dict[str, str], list[str]]:
    return (
        modified_environment(case, directory)
        if case[1] == "new_method"
        else baseline_environment(case, directory)
    )


def gpu_processes() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=30,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def configured_ports(env: dict[str, str]) -> list[int]:
    ports: set[int] = set()
    for key, value in env.items():
        if not (key.endswith("_PORT") or key.endswith("_PORTS")):
            continue
        for token in value.replace(",", " ").split():
            if token.isdigit():
                ports.add(int(token))
    return sorted(ports)


def occupied_ports(ports: list[int]) -> list[int]:
    occupied: list[int] = []
    for port in ports:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError:
            occupied.append(port)
        finally:
            probe.close()
    return occupied


def wait_ports_free(env: dict[str, str], timeout: float = 300.0) -> None:
    """Require all case ports to remain free across three probes."""
    ports = configured_ports(env)
    deadline = time.monotonic() + timeout
    consecutive_free = 0
    while True:
        busy = occupied_ports(ports)
        if not busy:
            consecutive_free += 1
            if consecutive_free >= 3:
                return
        else:
            consecutive_free = 0
        if time.monotonic() >= deadline:
            raise RuntimeError(f"configured ports remained occupied: {busy}")
        time.sleep(2)


def memory_events() -> dict[str, int]:
    cgroup = Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    path = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "memory.events"
    if not path.exists():
        return {}
    return {
        key: int(value)
        for key, value in (line.split() for line in path.read_text().splitlines())
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def ensure_analysis(directory: Path) -> None:
    """Create the standard derived summary without modifying raw run data."""
    summary = directory / "offload_analysis_summary.json"
    if summary.exists():
        return
    required = (
        directory / "closed_loop_boundaries.json",
        directory / "requests.jsonl",
        directory / "engine_metrics.jsonl",
        directory / "engine_throughput_2s.jsonl",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("cannot postprocess incomplete run: " + ", ".join(missing))
    subprocess.run(
        [
            str(PD_ENV / "bin/python"),
            str(PD / "scripts/tools/analyze_pd_offload.py"),
            "--run-dir",
            str(directory),
        ],
        cwd=SLIME,
        check=True,
        timeout=600,
    )


def summarize(directory: Path, before_events: dict[str, int]) -> dict[str, object]:
    summary = json.loads((directory / "offload_analysis_summary.json").read_text())
    boundary = json.loads((directory / "closed_loop_boundaries.json").read_text())
    launch = json.loads((directory / "launch_config.json").read_text())
    launch_env = launch.get("env", {})
    prefill_tp = int(launch_env.get("PREFILL_TP_SIZE", launch_env.get("MODEL_TP_SIZE", 1)))
    decode_tp = int(launch_env.get("DECODE_TP_SIZE", launch_env.get("MODEL_TP_SIZE", 1)))
    if summary.get("status") != "steady_churn":
        raise RuntimeError(f"unexpected status: {summary.get('status')}")
    if float(summary.get("warmup_seconds", 0)) < 299:
        raise RuntimeError("warmup shorter than 299 seconds")
    if float(summary.get("measurement_seconds", 0)) < 1199:
        raise RuntimeError("measurement shorter than 1199 seconds")
    start_state = boundary["state_at_measurement_start"]
    end_state = boundary["state_at_measurement_end"]
    failures = int(end_state["failures"]) - int(start_state["failures"])
    if failures:
        raise RuntimeError(f"measurement failures={failures}")

    start = float(boundary["measurement_start_wall"])
    end = float(boundary["measurement_end_wall"])
    keys = {
        "p_kv": "sglang_token_usage",
        "p_queue": "sglang_num_queue_reqs",
        "p_inflight": "sglang_num_prefill_inflight_queue_reqs",
        "d_kv": "sglang_token_usage",
        "d_running": "sglang_num_running_reqs",
        "d_queue": "sglang_num_queue_reqs",
        "d_prealloc": "sglang_num_decode_prealloc_queue_reqs",
        "d_transfer": "sglang_num_decode_transfer_queue_reqs",
    }
    values = {key: [] for key in keys}
    tail_decode_counters: list[tuple[float, float]] = []
    with (directory / "engine_metrics.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if not start <= float(row["ts"]) <= end:
                continue
            role = row.get("role")
            if role == "decode" and float(row["ts"]) >= end - 120:
                tail_decode_counters.append(
                    (
                        float(row["ts"]),
                        sum(
                            float(
                                endpoint["metrics"].get(
                                    "sglang_realtime_tokens_total|mode=decode", 0.0
                                )
                            )
                            for endpoint in row.get("endpoint_metrics", [])
                        ),
                    )
                )
            for endpoint in row.get("endpoint_metrics", []):
                metrics = endpoint["metrics"]
                selected = (
                    ("p_kv", "p_queue", "p_inflight")
                    if role == "prefill"
                    else ("d_kv", "d_running", "d_queue", "d_prealloc", "d_transfer")
                    if role == "decode"
                    else ()
                )
                for key in selected:
                    values[key].append(float(metrics.get(keys[key], 0.0)))

    if len(tail_decode_counters) < 2:
        raise RuntimeError("missing Decode liveness samples in final 120 seconds")
    tail_decode_tokens = max(
        0.0, tail_decode_counters[-1][1] - tail_decode_counters[0][1]
    )
    if tail_decode_tokens <= 0:
        raise RuntimeError("Decode made no token progress in final 120 seconds")

    reuse = summary.get("completion_weighted_reuse", {})
    after_events = memory_events()
    event_delta = {
        key: after_events.get(key, 0) - before_events.get(key, 0)
        for key in set(before_events) | set(after_events)
    }
    if event_delta.get("oom", 0) or event_delta.get("oom_kill", 0):
        raise RuntimeError(f"new cgroup OOM event: {event_delta}")
    return {
        "status": summary["status"],
        "warmup_seconds": summary["warmup_seconds"],
        "measurement_seconds": summary["measurement_seconds"],
        "completed_agents": summary["completed_agents"],
        "failures": failures,
        "agent_s": summary["completion_rps"],
        "prefill_tps": summary["prefill_compute_tps"],
        "decode_tps": summary["decode_tps"],
        "decode_tps_per_gpu": summary["decode_tps_per_gpu"],
        # The analyzer's *_gpus fields count HTTP endpoints (logical TP
        # engines).  CUDA device-timer counters sum all TP ranks, so divide by
        # both endpoint count and TP size for the physical-GPU table column.
        "prefill_forward_per_gpu": (
            summary["prefill_gpu_busy_fraction"] / summary["prefill_gpus"] / prefill_tp
        ),
        "decode_forward_per_gpu": (
            summary["decode_gpu_busy_fraction"] / summary["decode_gpus"] / decode_tp
        ),
        "prefill_per_agent": summary["prefill_compute_tokens_per_completed_agent"],
        "decode_per_agent": summary["decode_tokens_per_completed_agent"],
        "tail_120s_decode_tokens": tail_decode_tokens,
        "parent_kv_reuse": reuse.get("page_aligned_parent_prefix_reuse_fraction", 0.0),
        **{key: mean(series) for key, series in values.items()},
        "mooncake": summary.get("mooncake", {}),
        "memory_events_delta": event_delta,
    }


def validate_inputs() -> None:
    required = [
        PD_ENV / "bin/python",
        BASELINE_ENV / "bin/python",
        MODIFIED_SGLANG / "sglang/__init__.py",
        MODEL_8B / "config.json",
        MODEL_32B / "model.safetensors.index.json",
        DATA / "browsecomp/bc_train.jsonl",
        DATA / "dapo-math-17k/dapo-math-17k.jsonl",
        PD / "configs/workloads/fixed_browsecomp_source_order_n680.json",
        CONTROL / "fixed_random_s2026_n8192.json",
        PD / "data/browsecomp/artifacts/search/corpus_embeddings.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing inputs:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_inputs()
    CONTROL.mkdir(parents=True, exist_ok=True)
    states_path = CONTROL / "state.json"
    states = json.loads(states_path.read_text()) if states_path.exists() else {}
    if args.dry_run:
        for index, case in enumerate(CASES, 1):
            directory = run_dir(case)
            env, command = case_environment(case, directory)
            print(
                index,
                case_key(case),
                directory,
                command,
                {key: env.get(key) for key in (
                    "MODEL_PATH", "MAX_INFLIGHT", "PREFILL_GPUS", "PREFILL_GPU_GROUPS",
                    "DECODE_GPUS", "DECODE_GPU_GROUPS", "SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH",
                )},
            )
        return

    current: subprocess.Popen[str] | None = None

    def stop(_signum: int, _frame: object) -> None:
        nonlocal current
        if current is not None and current.poll() is None:
            os.killpg(current.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    for index, case in enumerate(CASES, 1):
        key = case_key(case)
        directory = run_dir(case)
        if states.get(key, {}).get("status") == "complete":
            continue
        if directory.exists():
            # The launcher may have completed all raw measurements before the
            # controller itself failed in deterministic postprocessing.  Such
            # a run is recoverable only when its closed-loop boundary is
            # explicitly complete and no GPU process survived.
            boundary_path = directory / "closed_loop_boundaries.json"
            if not boundary_path.exists():
                raise RuntimeError(f"non-complete run directory already exists: {directory}")
            boundary = json.loads(boundary_path.read_text())
            if boundary.get("status") != "steady_churn":
                raise RuntimeError(f"existing run is not recoverable: {directory}")
            occupied = gpu_processes()
            if occupied:
                raise RuntimeError(f"GPU processes present before recovering {key}: {occupied}")
            ensure_analysis(directory)
            before_events = states.get(key, {}).get("memory_events_before")
            if before_events is None:
                now_events = memory_events()
                if now_events.get("oom", 0) or now_events.get("oom_kill", 0):
                    raise RuntimeError(
                        f"cannot establish OOM baseline for existing run: {now_events}"
                    )
                before_events = now_events
            result = summarize(directory, before_events)
            atomic_json(directory / "h100_result.json", result)
            states.setdefault(key, {}).update(
                status="complete",
                index=index,
                total=len(CASES),
                ended=time.time(),
                recovered_from_raw=True,
                run_dir=str(directory),
                result=result,
            )
            atomic_json(states_path, states)
            continue
        env, command = case_environment(case, directory)
        wait_ports_free(env)
        occupied = gpu_processes()
        if occupied:
            raise RuntimeError(f"GPU processes present before {key}: {occupied}")
        directory.mkdir(parents=True)
        atomic_json(directory / "launch_config.json", {"command": command, "env": env})
        before_events = memory_events()
        states[key] = {
            "status": "running",
            "index": index,
            "total": len(CASES),
            "started": time.time(),
            "run_dir": str(directory),
            "memory_events_before": before_events,
        }
        atomic_json(states_path, states)
        atomic_json(CONTROL / "heartbeat.json", {"case": key, "index": index, "ts": time.time()})
        try:
            with (directory / "launcher.log").open("w") as log:
                current = subprocess.Popen(
                    command,
                    cwd=SLIME,
                    env=env,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 7200
                while current.poll() is None:
                    if time.monotonic() > deadline:
                        os.killpg(current.pid, signal.SIGTERM)
                        raise RuntimeError("case exceeded 2-hour limit")
                    atomic_json(
                        CONTROL / "heartbeat.json",
                        {"case": key, "index": index, "pid": current.pid, "ts": time.time()},
                    )
                    time.sleep(15)
            if current.returncode:
                raise RuntimeError(f"launcher exit={current.returncode}")
            ensure_analysis(directory)
            result = summarize(directory, before_events)
            atomic_json(directory / "h100_result.json", result)
            occupied = gpu_processes()
            if occupied:
                raise RuntimeError(f"GPU processes survived completed launcher: {occupied}")
            states[key].update(status="complete", ended=time.time(), result=result)
            atomic_json(states_path, states)
        except BaseException as exc:
            if current is not None and current.poll() is None:
                os.killpg(current.pid, signal.SIGTERM)
                try:
                    current.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    pass
            states[key].update(status="failed", ended=time.time(), error=repr(exc))
            atomic_json(states_path, states)
            atomic_json(CONTROL / "heartbeat.json", {"case": key, "status": "failed", "ts": time.time()})
            raise
        finally:
            current = None
    atomic_json(CONTROL / "heartbeat.json", {"status": "complete", "total": len(CASES), "ts": time.time()})


if __name__ == "__main__":
    main()
