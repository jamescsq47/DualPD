#!/usr/bin/env python3
"""One audited BrowseComp regression, reusing the dedicated seven-day job.

No engine code is changed here. Use --prepare on the login node, obtain an
independent audit for that exact plan, then --wait-for-job. Only a RUNNING
allocation can start a step; an observation error never triggers a new job.
The existing launcher owns shutdown. A failed cleanup gate stops this runner
without killing unknown processes or starting another experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

# Login and compute hosts can resolve the same /home mount to different NFS
# aliases. Keep the selected checkout spelling stable across launch/preflight;
# content hashes, not host-specific realpath spelling, identify its source.
ROOT = Path(os.environ.get("DUALPD_ROOT", Path(__file__).absolute().parents[2]))
ENGINE = ROOT.parent / "sglang_pd"
PYTHON = ROOT / ".venv/bin/python"
SCRIPT = ROOT / "scripts/cluster" / Path(__file__).name


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, timeout=60, **kwargs).strip()


def load_recipe():
    spec = importlib.util.spec_from_file_location(
        "cleanup_browsecomp_recipe",
        SCRIPT.with_name("browsecomp_qwen3_8b_lowc_h100_matrix.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The historical recipe resolves __file__ internally. Its deployment roots
    # must follow the explicitly selected checkout, not a node's mount alias.
    module.base.PD = ROOT
    module.base.SLIME = ROOT
    module.base.REPO = ROOT.parent
    module.base.MODIFIED_SGLANG = ENGINE / "python"
    return module


def environment(directory):
    recipe = load_recipe()
    env, launch = recipe.modified_environment(
        ("browsecomp8b4p4d", "new_method", 256), directory,
    )
    # No package install or mutation of shared pd/pd_baseline environments.
    env.update(
        PD_ENV_BIN=str(PYTHON.parent),
        PATH=f"{PYTHON.parent}:{recipe.base.PD_ENV / 'bin'}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        PYTHONPATH=f"{ENGINE / 'python'}:{ROOT}",
        DUALPD_ENGINE_ROOT=str(ENGINE),
        MAX_CONTEXT_LENGTH="40960", MAX_RESPONSE_LENGTH="36864",
        TERM="dumb",
        # Preserve the historical formal protocol, also valid as a regression
        # gate. Shorter smoke runs must never enter the accepted results table.
        WARMUP_SECONDS="300", MAX_WARMUP_SECONDS="420", MEASURE_SECONDS="1200",
    )
    env.pop("HF_TOKEN", None)  # Everything needed is cached; never persist a token.
    return env, launch


def source_state():
    if command(["git", "-C", str(ENGINE), "status", "--porcelain"]):
        raise RuntimeError("engine must be committed and clean before GPU validation")
    hashes = frontend_sources(ROOT)
    return {
        "engine_revision": command(["git", "-C", str(ENGINE), "rev-parse", "HEAD"]),
        "frontend_serving_sha256": hashes,
        "runner_sha256": sha(SCRIPT),
    }


def frontend_sources(root):
    # Include new/untracked and ignored source too: importable modules do not
    # become harmless just because they are absent from git ls-files.
    files = {p for p in root.glob("*.py")}
    for name in ("runtime", "data", "configs", "scripts"):
        files.update(p for p in (root / name).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".sh", ".sbatch", ".yaml", ".yml", ".json", ".toml"}
                     and "__pycache__" not in p.parts)
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def prepare(directory, job_id, stage, reference=None):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", stage):
        raise ValueError("stage must be a simple identifier")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "plan.json").exists():
        raise RuntimeError("plan already exists; use a fresh stage/attempt directory")
    if stage != "baseline" and reference is None:
        raise RuntimeError("a refactor stage needs an accepted before-result reference")
    env, launch = environment(directory)
    required = [PYTHON, Path(env["MODEL_PATH"]) / "config.json",
                Path(env["QA_DATA"]), Path(env["SCHEDULE_FILE"]),
                Path(env["SEARCH_SERVER_EMBEDDING_CACHE"])]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"missing validation input: {path}")
    write_json(directory / "plan.json", {
        "stage": stage, "job_id": str(job_id), "directory": str(directory),
        "source": source_state(), "env": env, "command": launch,
        "input_sha256": {str(p): sha(p) for p in required[1:4]},
        "reference": ({"path": str(reference.resolve()), "sha256": sha(reference)}
                      if reference is not None else None),
        "protocol": "BrowseComp Qwen3-8B TP1 4P4D c256 warmup300 measurement1200",
    })


def validate_plan(path):
    plan = json.loads(path.read_text())
    if Path(plan["directory"]).resolve() != path.parent.resolve():
        raise RuntimeError("plan was moved from its isolated run directory")
    if source_state() != plan["source"]:
        raise RuntimeError("source changed since the independent audit plan was prepared")
    env, launch = environment(path.parent)
    if env != plan["env"] or launch != plan["command"]:
        raise RuntimeError("launch recipe changed since plan preparation")
    for name, digest in plan["input_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"input changed: {name}")
    if plan.get("reference"):
        reference = plan["reference"]
        if sha(Path(reference["path"])) != reference["sha256"]:
            raise RuntimeError("reference result changed since audit")
    audit = json.loads((path.parent / "audit.json").read_text())
    if audit.get("decision") != "GO" or audit.get("plan_sha256") != sha(path):
        raise RuntimeError("missing independent audit GO for this exact plan")
    return plan


def job_state(job_id):
    record = command(["scontrol", "show", "job", str(job_id), "-o"])
    return dict(re.findall(r"(?:^|\s)(\w+)=([^\s]*)", record))


def wait_for_job(path):
    plan = validate_plan(path)
    directory = path.parent
    with (directory / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            # A timeout is an observation failure, not proof the job ended.
            try:
                live = job_state(plan["job_id"])
            except (subprocess.SubprocessError, OSError) as exc:
                write_json(directory / "controller.json", {
                    "status": "observation_error", "error": str(exc), "at": time.time(),
                })
                time.sleep(30)
                continue
            status = live.get("JobState")
            write_json(directory / "controller.json", {
                "status": status, "job_id": plan["job_id"],
                "node": live.get("NodeList"), "at": time.time(),
            })
            if status == "RUNNING":
                break
            if status not in {"PENDING", "CONFIGURING"}:
                raise RuntimeError(f"allocation is not usable: {live}")
            time.sleep(30)
        validate_plan(path)
        args = ["srun", "--jobid", plan["job_id"], "--overlap", "--exact",
                "--nodes=1", "--ntasks=1", "--cpus-per-task=224", "--cpu-bind=none", "--mem=1800G",
                "--gres=gpu:h100:8", str(PYTHON), str(SCRIPT),
                "--on-node", "--plan", str(path)]
        with (directory / "step.log").open("a") as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
        write_json(directory / "controller.json", {
            "status": "finished" if result.returncode == 0 else "failed",
            "returncode": result.returncode, "job_id": plan["job_id"], "at": time.time(),
        })
        if result.returncode:
            raise RuntimeError(f"node validation step exited {result.returncode}; inspect step.log")


def helpers():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            name = (entry / "comm").read_text().strip()
            if name in {"apptainer", "squashfuse_ll", "fuse-overlayfs"}:
                found.append({"pid": int(entry.name), "name": name})
        except (OSError, ProcessLookupError):
            pass
    return found


def resources(env):
    base = load_recipe().base
    return {"gpu_processes": base.gpu_processes(),
            "occupied_ports": base.occupied_ports(base.configured_ports(env)),
            "helpers": helpers()}


def cleanup_gate(env, timeout=300):
    deadline = time.monotonic() + timeout
    free = 0
    while True:
        state = resources(env)
        free = free + 1 if not any(state.values()) else 0
        if free >= 3:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"resource gate failed, no unknown processes killed: {state}")
        time.sleep(2)


def read_decode_progress(handle):
    """Consume complete JSONL records; keep a concurrent partial write for later."""
    latest = None
    while True:
        offset = handle.tell()
        line = handle.readline()
        if not line or not line.endswith("\n"):
            handle.seek(offset)
            return latest
        row = json.loads(line)
        if row.get("role") == "decode":
            count = sum(float(endpoint["metrics"].get(
                "sglang_realtime_tokens_total|mode=decode", 0,
            )) for endpoint in row.get("endpoint_metrics", []))
            latest = float(row["ts"]), count


def supervise(process, directory, max_seconds=7800):
    started = time.monotonic()
    last_growth = started
    previous = None
    metrics = None
    stop_reason = None
    def stop(signum, _frame):
        nonlocal stop_reason
        stop_reason = f"received signal {signum}"
    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while process.poll() is None:
            now = time.monotonic()
            if stop_reason:
                raise RuntimeError(stop_reason)
            if now - started > max_seconds:
                raise RuntimeError("validation exceeded bounded startup/workload/cleanup time")
            if metrics is None and (directory / "engine_metrics.jsonl").exists():
                metrics = (directory / "engine_metrics.jsonl").open()
                last_growth = now
            sample = read_decode_progress(metrics) if metrics else None
            if sample is not None:
                if previous is None or sample[1] > previous[1]:
                    last_growth = now
                elif sample[1] < previous[1]:
                    raise RuntimeError("Decode counter reset during validation")
                previous = sample
            # The completed window is immutable; metric polling is expected to
            # stop during dependency-ordered service drain and postprocessing.
            finished = (directory / "closed_loop_boundaries.json").exists()
            if previous and not finished:
                if time.time() - previous[0] > 90:
                    raise RuntimeError("Decode metrics stale for more than 90 seconds")
                if now - last_growth > 300:
                    raise RuntimeError("Decode made no progress for 300 seconds")
            write_json(directory / "runtime.json", {
                "status": "draining" if finished else "running", "at": time.time(),
                "launcher_pid": process.pid, "decode_sample": previous,
            })
            time.sleep(10)
        if process.returncode:
            raise RuntimeError(f"BrowseComp launcher exited {process.returncode}")
    finally:
        if metrics:
            metrics.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def stop_owned_launcher(process):
    deadline = time.monotonic() + 300
    # The outer wrapper itself has no trap. Its child pipeline does; signal
    # their exact, separately-created group so the child's EXIT trap runs.
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    process.wait(timeout=max(0, deadline - time.monotonic()))
    # The wrapper can exit before its pipeline trap removes the ledgers.
    # GPU/port zero alone does not prove that final cleanup has finished.
    while owned_group_alive(process.pid):
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("owned launcher process-group drain", 300)
        time.sleep(1)


def owned_group_alive(pgid):
    rows = command(["ps", "-eLo", "pgid=,stat="]).splitlines()
    return any(int(fields[0]) == pgid and not fields[1].startswith("Z")
               for fields in (row.split() for row in rows) if len(fields) >= 2)


def node_lock_path(job_id):
    # Must not depend on a stage's directory: all plans on this allocated node
    # share one advisory slot even if their scratch parents differ.
    return Path(f"/tmp/dualpd-{os.getuid()}-job-{job_id}.experiment.lock")


def parse_cpu_list(value):
    cpus = set()
    for span in value.strip().split(","):
        bounds = [int(n) for n in span.split("-")]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus


def numa_preflight():
    affinity = set(os.sched_getaffinity(0))
    available = {
        int(path.parent.name[4:]): bool(parse_cpu_list(path.read_text()) & affinity)
        for path in Path("/sys/devices/system/node").glob("node*/cpulist")
    }
    domains = []
    for bus in command(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]).splitlines():
        domain, rest = bus.strip().lower().split(":", 1)
        path = Path("/sys/bus/pci/devices") / f"{int(domain, 16):04x}:{rest}" / "numa_node"
        node = int(path.read_text())
        if not available.get(node):
            raise RuntimeError(f"GPU NUMA {node} has no CPUs in this step's affinity")
        domains.append(node)
    return {"cpu_affinity": sorted(affinity), "gpu_numa_nodes": domains,
            "cpu_numa_domains_available": available}


def comparison(plan, result):
    tps = float(result["decode_tps"])
    if not math.isfinite(tps) or tps <= 0 or result["completed_agents"] <= 0:
        raise RuntimeError("no valid positive Decode throughput/completions")
    if not plan.get("reference"):
        return {"acceptance": "baseline", "decode_ratio": None}
    reference_path = Path(plan["reference"]["path"])
    reference = json.loads(reference_path.read_text())
    previous_plan = json.loads((reference_path.parent / "plan.json").read_text())
    # Absolute run paths necessarily differ; all other effective knobs and
    # workload inputs must be byte-for-byte the same for the comparison.
    previous_env = {k: v for k, v in previous_plan["env"].items() if k != "RUN_DIR"}
    current_env = {k: v for k, v in plan["env"].items() if k != "RUN_DIR"}
    if (previous_env != current_env or previous_plan["input_sha256"] != plan["input_sha256"]
            or not reference.get("cleanup_passed")
            or reference.get("acceptance") not in {"baseline", "pass"}):
        raise RuntimeError("reference is not an accepted run under identical settings")
    ratio = tps / float(reference["decode_tps"])
    return {"acceptance": "pass" if ratio >= 0.9 else "regression_needs_repeat",
            "decode_ratio": ratio, "reference": str(reference_path)}


def run_on_node(path):
    plan = validate_plan(path)
    directory = path.parent
    job_id = plan["job_id"]
    if os.environ.get("SLURM_JOB_ID") != job_id:
        raise RuntimeError("must execute inside the exact dedicated allocation")
    live = job_state(job_id)
    nodes = command(["scontrol", "show", "hostnames", live.get("NodeList", "")]).splitlines()
    if live.get("JobState") != "RUNNING" or nodes != [socket.gethostname().split(".")[0]]:
        raise RuntimeError("not on the single RUNNING allocated node")
    gpu_names = command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
    if len(gpu_names) != 8 or any("H100" not in name for name in gpu_names):
        raise RuntimeError(f"expected dedicated eight-H100 node, found {gpu_names}")
    write_json(directory / "numa-preflight.json", numa_preflight())
    env = plan["env"]
    # One node-wide slot across every stage, in addition to each controller lock.
    fd = os.open(node_lock_path(job_id), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "launch_config.json").exists():
            raise RuntimeError("never overwrite an attempted run; prepare a fresh attempt")
        write_json(directory / "pre-cleanup.json", cleanup_gate(env, timeout=30))
        base = load_recipe().base
        before = base.memory_events()
        write_json(directory / "launch_config.json", {
            "env": env, "command": plan["command"], "job_id": job_id,
            "node": socket.gethostname(), "source": plan["source"],
        })
        failure = None
        with (directory / "launcher.log").open("w") as log:
            process = subprocess.Popen(plan["command"], env=env, cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                supervise(process, directory)
            except BaseException as exc:
                failure = str(exc)
                # Signal only this exact owned launcher, allowing its existing
                # nested supervisor to perform the full ordered shutdown.
                try:
                    stop_owned_launcher(process)
                except subprocess.TimeoutExpired:
                    failure += "; owned supervisor did not exit within 300s"
            finally:
                try:
                    final = cleanup_gate(env)
                    write_json(directory / "post-cleanup.json", final)
                except Exception as exc:
                    failure = f"{failure or ''}; cleanup failed: {exc}"
        try:
            if failure:
                raise RuntimeError(failure)
            validate_plan(path)
            base.ensure_analysis(directory)
            result = base.summarize(directory, before)
            compared = comparison(plan, result)
            write_json(directory / "result.json", {
                **result, **compared, "stage": plan["stage"], "source": plan["source"],
                "job_id": job_id, "node": socket.gethostname(), "cleanup_passed": True,
            })
            if compared["acceptance"] == "regression_needs_repeat":
                raise RuntimeError("Decode throughput fell over 10%; repeat/investigate before next stage")
            write_json(directory / "runtime.json", {"status": "complete", "at": time.time()})
        except BaseException as exc:
            write_json(directory / "runtime.json", {
                "status": "failed", "error": str(exc), "at": time.time(),
            })
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--wait-for-job", action="store_true")
    modes.add_argument("--on-node", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--stage", default="baseline")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    if args.prepare:
        if not args.job_id or not args.job_id.isdigit():
            parser.error("--prepare requires a numeric --job-id")
        prepare(path.parent, args.job_id, args.stage, args.reference)
    elif args.wait_for_job:
        wait_for_job(path)
    else:
        run_on_node(path)


if __name__ == "__main__":
    main()
