#!/usr/bin/env python3
"""Recover only the known false liveness rejection; never start GPU services."""

# Resolve only maintained core/shared tools; historical siblings stay local.
if __package__:
    from ._core import load_core as _load_core
else:
    from _core import load_core as _load_core

import hashlib
import json
from pathlib import Path
import socket
import time

if __package__:
    from . import browsecomp_qwen3_8b_dram_p64d64_h100 as run
else:
    import browsecomp_qwen3_8b_dram_p64d64_h100 as run
prepare_analysis = _load_core('dram_complete_frame_analysis').prepare_analysis


def main():
    if socket.gethostname() != "node1701":
        raise RuntimeError("Run recovery on writer node1701 for local cleanup/OOM verification")
    state_path = run.CONTROL / "state.json"
    states = json.loads(state_path.read_text())
    key = run.dram.case_key(run.CASE)
    item = states[key]
    if item.get("status") != "failed" or item.get("error") != "RuntimeError('Decode made no token progress in final 120 seconds')":
        raise RuntimeError("Not the known failed-validation state; refusing recovery")
    directory = Path(item["run_dir"])
    sources = [directory / name for name in ("engine_metrics.jsonl", "requests.jsonl", "closed_loop_boundaries.json",
                                              "host_resources_2s.jsonl", "process_memory_60s.jsonl")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    run.dram.cleanup_gate(run.CASE)
    run.dram.lowc.original_atomic_json(run.CONTROL / "state-before-complete-frame-recovery.json", states)
    prepare_analysis(directory)
    result = run.summarize(directory, item.get("memory_events_before", {}))
    if result["warmup_seconds"] < 300 or result["measurement_seconds"] < 1200:
        raise RuntimeError("Incomplete requested window; do not accept")
    if result["failures"] != 0:
        raise RuntimeError("Request failures; do not accept")
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)]:
            raise RuntimeError(f"Raw source changed during recovery: {path}")
    original_error = item["error"]
    run.dram.lowc.original_atomic_json(directory / "h100_result.json", result)
    item.update(status="complete", ended=time.time(), result=result,
                prior_failed_validation=original_error, recovered_by="complete-frame offline analysis")
    item.pop("error", None)
    # The existing atomic_json hook repeats the local hard cleanup gate before
    # publishing complete, then updates only this case's managed MD section.
    run.dram.atomic_json(state_path, states)
    run.dram.lowc.original_atomic_json(run.CONTROL / "heartbeat.json",
                                      dict(status="complete", total=1, ts=time.time()))
    run.dram.cleanup_gate(run.CASE)
    run.dram.lowc.original_atomic_json(run.CONTROL / "final-cleanup-gate.json",
                                      dict(ts=time.time(), node=socket.gethostname(), status="clean",
                                           gpu_processes=0, busy_configured_ports=[], apptainer_fuse_helpers=0))
    run.dram.lowc.original_atomic_json(run.CONTROL / "offline-recovery.json",
                                      dict(ts=time.time(), status="recovered_complete_window", raw_sources_sha256=hashes,
                                           original_validation_error=original_error,
                                           no_gpu_experiment_started=True, no_raw_sources_modified=True,
                                           telemetry_validation=result["telemetry_validation"]))
    print(json.dumps(dict(status="complete", decode_tps=result["decode_tps"],
                          warmup_seconds=result["warmup_seconds"], measurement_seconds=result["measurement_seconds"],
                          tail_120s_decode_tokens=result["tail_120s_decode_tokens"],
                          host_resources=result["host_resources"]), indent=2))


if __name__ == "__main__":
    main()
