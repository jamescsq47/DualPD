"""BrowseComp resource observations and full-window summary contracts.

No experiment controller, model import, process cleanup or GPU mutation occurs
at import. Resource probes run only when explicitly called by the controller.
"""
import json
from pathlib import Path
import socket
import statistics
import subprocess


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


def ensure_analysis(directory: Path, *, python: Path, root: Path) -> None:
    """Create the standard derived summary without modifying raw run data."""
    summary = directory / 'offload_analysis_summary.json'
    if summary.exists():
        return
    required = (
        directory / 'closed_loop_boundaries.json',
        directory / 'requests.jsonl',
        directory / 'engine_metrics.jsonl',
        directory / 'engine_throughput_2s.jsonl',
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError('cannot postprocess incomplete run: ' + ', '.join(missing))
    subprocess.run(
        [str(python), str(root / 'scripts/tools/analyze_pd_offload.py'),
         '--run-dir', str(directory)],
        cwd=root, check=True, timeout=600,
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
