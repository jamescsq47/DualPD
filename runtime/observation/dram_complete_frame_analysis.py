"""Offline native-PD analysis using only complete, synchronous P/D frames.

Original counters/requests/boundaries remain immutable. A missing endpoint is
never treated as a zero counter. The standard analyzer and acceptance gate run
against a separately derived view with unchanged measurement boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


P_TOKEN = "sglang_realtime_tokens_total|mode=prefill_compute"
D_TOKEN = "sglang_realtime_tokens_total|mode=decode"
P_GPU = "sglang_gpu_execution_seconds_total|category=forward_extend"
D_GPU = "sglang_gpu_execution_seconds_total|category=forward_decode"
VIEW = "analysis-complete-frames-v1"
DEFAULT_ANALYSIS_PYTHON = sys.executable


def complete_frames(rows, env, start, end):
    expected = {role: {f"http://127.0.0.1:{p}" for p in env[key].split()}
                for role, key in [("prefill", "PREFILL_PORTS"), ("decode", "DECODE_PORTS")]}
    keys = {"prefill": (P_TOKEN, P_GPU), "decode": (D_TOKEN, D_GPU)}
    grouped = {}
    for row in rows:
        if row.get("role") not in expected:
            continue
        frame = grouped.setdefault(float(row["ts"]), {})
        if row["role"] in frame:
            raise RuntimeError("duplicate role within telemetry frame")
        frame[row["role"]] = row
    frames, rejected = [], []
    previous = {}
    for ts, frame in sorted(grouped.items()):
        # Validate every observed endpoint, including partial frames that will
        # not be used for aggregation. A reset cannot hide in a rejected row.
        for role, row in frame.items():
            for endpoint in row.get("endpoint_metrics", []):
                if endpoint.get("endpoint") not in expected[role]:
                    continue
                for key in keys[role]:
                    if key not in endpoint.get("metrics", {}):
                        continue
                    value = endpoint["metrics"][key]
                    identity = (role, endpoint["endpoint"], key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise RuntimeError(f"invalid observed counter: {identity}")
                    if identity in previous and value < previous[identity]:
                        raise RuntimeError(f"counter reset: {identity}")
                    previous[identity] = value
        reason = None
        if set(frame) != set(expected):
            reason = "missing P/D role"
        else:
            for role in expected:
                endpoints = frame[role].get("endpoint_metrics", [])
                if (len(endpoints) != len(expected[role]) or
                        {x["endpoint"] for x in endpoints} != expected[role]):
                    reason = f"incomplete {role} endpoint set"
                    break
                for endpoint in endpoints:
                    for key in keys[role]:
                        value = endpoint.get("metrics", {}).get(key)
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                            reason = f"missing/invalid {role} counter"
        if reason:
            rejected.append(dict(ts=ts, reason=reason))
            continue
        for role in expected:
            for key in keys[role]:
                value = sum(x["metrics"][key] for x in frame[role]["endpoint_metrics"])
                if not math.isclose(value, frame[role].get("metrics", {}).get(key, float("nan")), rel_tol=1e-9):
                    raise RuntimeError("aggregate does not match complete endpoint counters")
        frames.append((ts, frame))
    window = [(ts, f) for ts, f in frames if start <= ts <= end]
    if len(window) < 2:
        raise RuntimeError("insufficient complete measurement frames")
    span = window[-1][0] - window[0][0]
    if span < .95 * (end - start) or window[0][0] - start > 15 or end - window[-1][0] > 15:
        raise RuntimeError("complete counter observations do not cover measurement edges")
    tail = [(ts, f) for ts, f in window if ts >= end - 120]
    if len(tail) < 2 or tail[-1][0] - tail[0][0] < 60 or tail[0][0] > end - 90:
        raise RuntimeError("insufficient complete final-120s observations")
    progress = tail[-1][1]["decode"]["metrics"][D_TOKEN] - tail[0][1]["decode"]["metrics"][D_TOKEN]
    if progress <= 0:
        raise RuntimeError("Decode made no token progress in final 120 seconds")
    audit = dict(policy="Use common complete P+D frames, no zero-fill or interpolation",
                 expected_endpoints={k: sorted(v) for k, v in expected.items()},
                 measurement_complete_frames=len(window), counter_span_seconds=span,
                 first_counter_offset_seconds=window[0][0]-start,
                 last_counter_age_seconds=end-window[-1][0], tail_120s_decode_tokens=progress,
                 maximum_complete_frame_gap_seconds=max(b[0]-a[0] for a,b in zip(window, window[1:])),
                 derived_engine_rate_interval="irregular complete-frame observations, not 2-second samples",
                 excluded_frames=rejected)
    return frames, audit


def prepare_analysis(directory, *, python=None, project_root=None):
    directory = Path(directory)
    boundary = json.loads((directory / "closed_loop_boundaries.json").read_text())
    if boundary["warmup_seconds"] < 300 or boundary["measurement_seconds"] < 1200:
        raise RuntimeError("incomplete 300+1200 window; rerun required")
    source = directory / "engine_metrics.jsonl"
    with source.open() as stream:
        rows = [json.loads(line) for line in stream]
    env = json.loads((directory / "launch_config.json").read_text())["env"]
    frames, audit = complete_frames(rows, env, boundary["measurement_start_wall"],
                                   boundary["measurement_end_wall"])
    audit["raw_engine_metrics_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    view = directory / VIEW
    view.mkdir(exist_ok=True)
    for name in ("closed_loop_boundaries.json", "launch_config.json", "requests.jsonl", "logs"):
        target = view / name
        if not target.exists():
            target.symlink_to(directory / name, target_is_directory=name == "logs")
    with (view / "engine_metrics.jsonl").open("w") as stream:
        for _, frame in frames:
            for role in ("prefill", "decode"):
                stream.write(json.dumps(frame[role]) + "\n")
    # Derived rate curves use the same complete frames as the summary.
    with (view / "engine_throughput_2s.jsonl").open("w") as stream:
        for (old_ts, old), (ts, new) in zip(frames, frames[1:]):
            for role, counter in (("prefill", P_TOKEN), ("decode", D_TOKEN)):
                metrics = new[role]["metrics"]
                rate = (metrics[counter] - old[role]["metrics"][counter]) / (ts-old_ts)
                row = dict(ts=ts, role=role, prompt_tokens_per_second=rate if role=="prefill" else 0,
                           generation_tokens_per_second=rate if role=="decode" else 0,
                           running_requests=metrics.get("sglang_num_running_reqs", 0),
                           decode_transfer_requests=metrics.get("sglang_num_decode_transfer_queue_reqs", 0),
                           prefill_inflight_requests=metrics.get("sglang_num_prefill_inflight_queue_reqs", 0))
                stream.write(json.dumps(row) + "\n")
    python = DEFAULT_ANALYSIS_PYTHON if python is None else str(python)
    root = (Path(__file__).resolve().parents[2] if project_root is None
            else Path(project_root))
    analyzer = root / "runtime/analysis/analyze_pd_offload.py"
    subprocess.run([python, str(analyzer), "--run-dir", str(view)], check=True, timeout=600)
    for name in ("offload_analysis_summary.json", "steady_offload_analysis.png"):
        old = directory / name
        backup = directory / (old.stem + ".legacy-aggregate" + old.suffix)
        if old.exists() and not backup.exists():
            shutil.copy2(old, backup)
        shutil.copy2(view / name, old)
    audit["corrected_decode_tps"] = json.loads((view / "offload_analysis_summary.json").read_text())["decode_tps"]
    (directory / "complete_frame_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return view
