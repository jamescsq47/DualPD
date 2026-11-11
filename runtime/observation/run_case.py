"""Run one existing case with opt-in DRAM observation; never choose/retry cases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

from .dram_resource_sampler import Sampler, summarize


def run_case(directory: Path, method: str, endpoints: list[str], command: list[str],
             mooncake_endpoint: str | None = None) -> int:
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"observation requires a fresh run directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PD_DRAM_TRACE_DIR"] = str(directory / "dram-pools")
    environment["PD_DRAM_SAMPLE_SECONDS"] = "2"
    # The custom engine exposes its own read-only arena counters. Only stock
    # HiCache needs an external observer; never patch custom engine classes.
    if method == "native_mooncake":
        observer = str(Path(__file__).resolve().parent / "host_pools")
        environment["PYTHONPATH"] = observer + os.pathsep + environment.get("PYTHONPATH", "")
    sampler = Sampler(directory, method, endpoints, mooncake_endpoint)
    child = None
    handlers = {}

    def forward(signum, _frame):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        child = subprocess.Popen(command, env=environment, start_new_session=True)
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, forward)
        sampler.start()
        code = child.wait()
    finally:
        # Only our child group is signalled. Existing case traps own engine/
        # container shutdown, including any nested sessions they created.
        if child is not None and child.poll() is None:
            forward(signal.SIGTERM, None)
            child.wait()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if sampler.thread.ident is not None:
            sampler.stop()
    if code:
        return code if code > 0 else 128 - code
    boundary = directory / "closed_loop_boundaries.json"
    if not boundary.is_file():
        raise RuntimeError("case returned without a completed measurement window; raw samples retained")
    window = json.loads(boundary.read_text())
    if (window.get("warmup_seconds", 0) < float(os.environ.get("WARMUP_SECONDS", 300))
            or window.get("measurement_seconds", 0) < float(os.environ.get("MEASURE_SECONDS", 1200))):
        raise RuntimeError("incomplete warmup/measurement window; raw samples retained, rerun required")
    result = summarize(directory)
    print(json.dumps({"host_resources": str(directory / "host_resource_summary.json"),
                      "valid_time_fraction": result["valid_time_fraction"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("new_method", "native_mooncake"), required=True)
    parser.add_argument("--ports", required=True, help="space-separated local P and D service ports")
    parser.add_argument("--mooncake-port", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide the existing case command after --")
    endpoints = [f"http://127.0.0.1:{int(port)}" for port in args.ports.split()]
    if len(endpoints) != 8:
        parser.error("this DRAM recipe expects four P and four D endpoints")
    master = f"http://127.0.0.1:{args.mooncake_port}" if args.mooncake_port else None
    if args.method == "native_mooncake" and master is None:
        parser.error("native_mooncake requires --mooncake-port")
    return run_case(args.run_dir, args.method, endpoints, command, master)


if __name__ == "__main__":
    raise SystemExit(main())
