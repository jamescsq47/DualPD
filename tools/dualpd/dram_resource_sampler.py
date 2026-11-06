"""Run-local DRAM/CPU sampler; retain complete Prometheus responses and bytes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import urllib.request


GIB = 1024**3


def fetch_metrics(endpoint):
    try:
        with urllib.request.urlopen(endpoint + "/metrics", timeout=2) as response:
            return dict(endpoint=endpoint, ts=time.time(), text=response.read().decode())
    except Exception as exc:
        return dict(endpoint=endpoint, ts=time.time(), error=str(exc))


def metric_value(raw, name):
    values = []
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+(\S+)")
    for line in raw.splitlines():
        match = pattern.match(line)
        if match:
            values.append(float(match.group(1)))
    return sum(values) if values else None


def memory_info(path):
    values = {}
    for line in path.read_text().splitlines():
        fields = line.replace(":", "").split()
        if len(fields) >= 2 and fields[1].isdigit():
            values[fields[0]] = int(fields[1]) * (1024 if fields[-1] == "kB" else 1)
    return values


def process_memory(trace_dir):
    """PSS avoids counting a shared memfd once per CUDA/process mapping."""
    marker = f"PD_DRAM_TRACE_DIR={trace_dir}".encode()
    rows = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if marker not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            mem = memory_info(directory / "smaps_rollup")
            status = memory_info(directory / "status")
            rows.append(dict(pid=int(directory.name), pss_bytes=mem.get("Pss", 0),
                             rss_bytes=mem.get("Rss", 0),
                             private_bytes=mem.get("Private_Clean", 0) + mem.get("Private_Dirty", 0),
                             locked_bytes=status.get("VmLck", 0)))
        except (OSError, ValueError, ProcessLookupError):
            continue
    return rows


class Sampler:
    def __init__(self, directory, method, endpoints, mooncake_endpoint=None):
        self.directory = Path(directory)
        self.trace_dir = self.directory / "dram-pools"
        self.method = method
        self.endpoints = endpoints
        self.mooncake_endpoint = mooncake_endpoint
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="dram-resource-sampler", daemon=True)
        # smaps_rollup walks hundreds of GiB of shared CUDA mappings. A single
        # scan can take tens of seconds: never put it on the 2-second KV path.
        self.process_thread = threading.Thread(target=self.run_process_memory,
                                               name="dram-process-pss", daemon=True)
        self.process_snapshot = None
        self.process_failure = None
        self.failure = None

    def start(self):
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.process_thread.start()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=20)
        # A procfs read cannot be cancelled safely. Keep at most one daemon
        # reader and do not let its shutdown delay the matrix/cleanup gate.
        self.process_thread.join(timeout=1)
        if self.thread.is_alive():
            raise RuntimeError("DRAM sampler did not stop")
        if self.failure:
            raise RuntimeError(f"DRAM sampler failed: {self.failure}")

    def run_process_memory(self):
        try:
            with (self.directory / "process_memory_60s.jsonl").open("a", buffering=1) as stream:
                while not self.stop_event.is_set():
                    started = time.time()
                    processes = process_memory(self.trace_dir)
                    snapshot = dict(started_ts=started, ts=time.time(), processes=processes,
                                    process_pss_bytes=sum(p["pss_bytes"] for p in processes),
                                    process_private_bytes=sum(p["private_bytes"] for p in processes))
                    snapshot["duration_seconds"] = snapshot["ts"] - started
                    stream.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
                    self.process_snapshot = snapshot
                    self.stop_event.wait(max(0, 60 - (time.time() - started)))
        except BaseException as exc:
            self.process_failure = repr(exc)

    def cached_process_memory(self, now):
        snapshot = self.process_snapshot
        age = now - snapshot["ts"] if snapshot else None
        valid = bool(snapshot and snapshot["processes"] and 0 <= age <= 120
                     and not self.process_failure)
        return dict(process_snapshot_ts=snapshot["ts"] if snapshot else None,
                    process_scan_seconds=snapshot["duration_seconds"] if snapshot else None,
                    process_sample_age_seconds=age, process_sample_valid=valid,
                    process_sample_error=self.process_failure,
                    processes=snapshot["processes"] if valid else [],
                    process_pss_bytes=snapshot["process_pss_bytes"] if valid else None,
                    process_private_bytes=snapshot["process_private_bytes"] if valid else None)

    def sample(self, pool):
        row = dict(ts=time.time(), method=self.method, pools=[], pool_errors=[])
        for path in self.trace_dir.glob("*.latest.json"):
            try:
                payload = json.loads(path.read_text())
                if row["ts"] - payload["ts"] > 6:
                    row["pool_errors"].append(f"stale observer: {path.name}")
                    continue
                row["pool_errors"].extend(payload.get("errors", []))
                for arena in payload["pools"]:
                    row["pools"].append({**arena, "pid": payload["pid"],
                                         "role": payload["role"], "port": payload["port"],
                                         "observed_ts": payload["ts"]})
            except (OSError, ValueError, KeyError) as exc:
                row["pool_errors"].append(f"{path.name}: {exc}")
        row["pool_count"] = len(row["pools"])
        for label in ("d2p", "p2d", "prefill", "decode"):
            matching = [p for p in row["pools"]
                        if (p["kind"] == "arena" and p["direction"] == label)
                        or (p["kind"] == "hicache" and p["role"] == label)]
            row[label + "_used_bytes"] = sum(p["used_bytes"] for p in matching)
            row[label + "_capacity_bytes"] = sum(p["capacity_bytes"] for p in matching)
            row[label + "_pool_count"] = len(matching)
        wanted = ("d2p", "p2d") if self.method == "new_method" else ("prefill", "decode")
        row["valid_local_sample"] = not row["pool_errors"] and all(
            row[k + "_pool_count"] == 4 for k in wanted)
        node = memory_info(Path("/proc/meminfo"))
        row["node_total_bytes"] = node.get("MemTotal")
        row["node_available_bytes"] = node.get("MemAvailable")
        row["node_used_bytes"] = node["MemTotal"] - node["MemAvailable"]
        row.update(self.cached_process_memory(row["ts"]))
        cg = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split(":")[-1].strip().lstrip("/")
        for filename in ("memory.current", "memory.peak", "memory.max"):
            try:
                value = (cg / filename).read_text().strip()
                row["cgroup_" + filename.replace(".", "_")] = int(value) if value.isdigit() else value
            except OSError:
                pass
        try:
            row["cgroup_cpu_stat"] = dict(line.split() for line in (cg / "cpu.stat").read_text().splitlines())
        except OSError:
            pass
        urls = list(self.endpoints)
        if self.mooncake_endpoint:
            urls.append(self.mooncake_endpoint)
        raw = list(pool.map(fetch_metrics, urls))
        if self.mooncake_endpoint:
            master = raw[-1].get("text", "")
            row["mooncake_used_bytes"] = metric_value(master, "master_allocated_bytes")
            row["mooncake_capacity_bytes"] = metric_value(master, "master_total_capacity_bytes")
        else:
            row["mooncake_used_bytes"] = 0
            row["mooncake_capacity_bytes"] = 0
        row["valid_sample"] = row["valid_local_sample"] and row["mooncake_used_bytes"] is not None
        row["local_used_bytes"] = sum(row[k + "_used_bytes"] for k in wanted)
        row["local_capacity_bytes"] = sum(row[k + "_capacity_bytes"] for k in wanted)
        row["total_kv_used_bytes"] = (row["local_used_bytes"] + row["mooncake_used_bytes"]
                                      if row["valid_sample"] else None)
        row["scrape_duration_seconds"] = time.time() - row["ts"]
        return row, raw

    def run(self):
        try:
            with ThreadPoolExecutor(max_workers=9) as pool, \
                    (self.directory / "host_resources_2s.jsonl").open("a", buffering=1) as stream, \
                    gzip.open(self.directory / "raw_prometheus_2s.jsonl.gz", "at") as raw_stream:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    row, raw = self.sample(pool)
                    row["written_ts"] = time.time()
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                    raw_stream.write(json.dumps(dict(ts=row["ts"], responses=raw)) + "\n")
                    raw_stream.flush()
                    self.stop_event.wait(max(0, 2 - (time.monotonic() - started)))
        except BaseException as exc:
            self.failure = repr(exc)


def summarize(directory):
    directory = Path(directory)
    boundary = json.loads((directory / "closed_loop_boundaries.json").read_text())
    start, end = boundary["measurement_start_wall"], boundary["measurement_end_wall"]
    rows = [json.loads(line) for line in (directory / "host_resources_2s.jsonl").open()]
    fields = ["ts", "measurement_offset_seconds", "phase", "valid_sample", "pool_count",
              "d2p_used_bytes", "p2d_used_bytes", "prefill_used_bytes", "decode_used_bytes",
              "local_capacity_bytes", "mooncake_used_bytes", "mooncake_capacity_bytes",
              "total_kv_used_bytes", "process_pss_bytes", "process_private_bytes",
              "process_snapshot_ts", "process_scan_seconds", "process_sample_age_seconds",
              "process_sample_valid", "process_sample_error",
              "node_used_bytes", "node_available_bytes", "cgroup_memory_current"]
    with (directory / "host_resources_2s.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["measurement_offset_seconds"] = row["ts"] - start
            row["phase"] = "warmup" if row["ts"] < start else ("measurement" if row["ts"] <= end else "cleanup")
            writer.writerow(row)
    pool_fields = ["ts", "measurement_offset_seconds", "phase", "observed_ts", "pid", "role", "port",
                   "kind", "direction", "object_id", "used_bytes", "capacity_bytes", "committed_bytes"]
    with (directory / "host_pools_2s.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pool_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            for arena in row["pools"]:
                writer.writerow({**arena, "ts": row["ts"],
                                 "measurement_offset_seconds": row["measurement_offset_seconds"],
                                 "phase": row["phase"]})
    window = [r for r in rows if start <= r["ts"] <= end]
    valid = [r for r in window if r.get("valid_sample")]
    coverage = sum(max(0, min(rows[i + 1]["ts"] if i + 1 < len(rows) else end, end, r["ts"] + 6)
                       - max(r["ts"], start))
                   for i, r in enumerate(rows) if r.get("valid_sample")) / (end - start)
    result = dict(window_start=start, window_end=end, samples=len(window), valid_samples=len(valid),
                  valid_time_fraction=coverage, source="direct pool counters + Mooncake master byte gauges",
                  units="bytes; GiB=2^30 bytes", interval_seconds=2, fields={})
    result["process_pss_interval_seconds"] = 60
    result["process_pss_valid_time_fraction"] = sum(
        max(0, min(rows[i + 1]["ts"] if i + 1 < len(rows) else end, end, r["ts"] + 6)
            - max(r["ts"], start))
        for i, r in enumerate(rows) if r.get("valid_sample") and r.get("process_sample_valid")) / (end - start)
    for name in fields:
        if not name.endswith("bytes") and name != "cgroup_memory_current":
            continue
        values = [r[name] for r in valid if isinstance(r.get(name), (int, float)) and math.isfinite(r[name])]
        if not values:
            continue
        area = duration = 0
        for i, row in enumerate(rows):
            if not row.get("valid_sample") or not isinstance(row.get(name), (int, float)):
                continue
            dt = max(0, min(rows[i + 1]["ts"] if i + 1 < len(rows) else end, end, row["ts"] + 6)
                     - max(row["ts"], start))
            area += dt * row[name]
            duration += dt
        result["fields"][name] = dict(mean_bytes=area / duration if duration else None,
                                      peak_bytes=max(values), mean_gib=area / duration / GIB if duration else None,
                                      peak_gib=max(values) / GIB)
    (directory / "host_resource_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if coverage < 0.95:
        raise RuntimeError(f"Host measurement coverage {coverage:.1%} < 95%; raw data preserved")
    if result["process_pss_valid_time_fraction"] < 0.95:
        raise RuntimeError("Process PSS coverage < 95%; raw data preserved")
    return result
