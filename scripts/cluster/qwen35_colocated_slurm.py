#!/usr/bin/env python3
"""Run a two-node Qwen colocated SWE baseline in Slurm.

Each allocated H100 node owns one complete Prefill+Decode TP8 server.  The
launcher deliberately uses the isolated ``pd_baseline`` environment and
scrubs every Agentic-PD environment variable before starting a component.
Long-lived components are owned by ``process_supervisor`` and can therefore
be stopped by exact run/component identity without PID-pattern matching.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

from process_supervisor import control as process_control, supervise
import qwen35_slurm as site


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
DEFAULT_PYTHON = (
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/"
    "pd_baseline/bin/python"
)
DEFAULT_WORKLOAD_PYTHON = (
    "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python"
)
DEFAULT_SGLANG_ROOT = ROOT.parent / "sglang-src/pd_baseline"
DEFAULT_MODEL = ROOT.parent / "models/Qwen3.8-27B"
DEFAULT_DATA = ROOT.parent / "datasets"
DEFAULT_SIF = (
    Path("/orcd/pool/004")
    / getpass.getuser()
    / "dualpd/containers/swebench-verified-sif"
)
DEFAULT_WORKLOAD = (
    ROOT
    / "configs/experiments/"
    "swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer.yaml"
)
DEFAULT_LOCAL_ROOT = Path("/var/tmp") / getpass.getuser() / "dualpd-colocated"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _server_command(
    args, host: str, port: int, gpus: list[int]
) -> list[str]:
    return [
        str(Path(args.python).resolve()),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model.resolve()),
        "--host",
        host,
        "--port",
        str(port),
        "--tp-size",
        str(args.tp_size),
        "--ep-size",
        str(getattr(args, "ep_size", 1)),
        "--context-length",
        "131072",
        "--page-size",
        "64",
        "--mem-fraction-static",
        "0.8",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bfloat16",
        *(
            []
            if getattr(args, "auto_kernel_backends", False)
            else [
                "--attention-backend", "triton",
                "--linear-attn-backend", "triton",
                "--moe-runner-backend", "triton",
            ]
        ),
        "--sampling-backend",
        "flashinfer",
        "--mamba-scheduler-strategy",
        getattr(args, "mamba_buffer_strategy", "extra_buffer"),
        "--mamba-track-interval",
        "64",
        "--mamba-full-memory-ratio",
        "0.5",
        "--random-seed",
        "2026",
        "--numa-node",
        *("0" if gpu < 4 else "1" for gpu in gpus),
        "--reasoning-parser",
        getattr(args, "reasoning_parser", "glm45"),
        "--tool-call-parser",
        getattr(args, "tool_call_parser", "qwen3_coder"),
        *(
            ["--enable-deterministic-inference"]
            if getattr(args, "deterministic_inference", False)
            else []
        ),
        "--enable-metrics",
        "--skip-server-warmup",
        "--uvicorn-access-log-exclude-prefixes",
        "/get_load",
        "/metrics",
        "/health",
    ]


def _base_env(args, engine_id: str, gpus: list[int] | None = None) -> dict[str, str]:
    env_bin = str(Path(args.python).resolve().parent)
    cache = args.local_root / "compiler-cache" / engine_id
    return {
        "PATH": os.pathsep.join((env_bin, "/usr/bin", os.environ.get("PATH", ""))),
        "PYTHONPATH": str(
            Path(getattr(args, "sglang_root", DEFAULT_SGLANG_ROOT)).resolve()
            / "python"
        ),
        "PYTHONNOUSERSITE": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "NVCC_CCBIN": "/usr/bin/g++",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_ARENA_TEST": "2",
        "SGLANG_STACK_SOFT_LIMIT_KIB": "2048",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus if gpus is not None else range(8))),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "true",
    }


def build_manifest(args, nodes: list[str], inventories: list[dict[str, str]]) -> dict:
    if not SAFE_NAME.fullmatch(args.run_id):
        raise ValueError("unsafe run ID")
    run_dir = args.run_dir.resolve()
    workload_output = run_dir / "workload"
    local_run = args.local_root / args.run_id
    python = str(Path(args.python).resolve())
    workload_python = str(Path(args.workload_python).resolve())
    if not nodes:
        raise ValueError("at least one physical node is required")
    node0 = nodes[0]
    ip0 = inventories[0]["ip"]
    node1 = nodes[1] if len(nodes) > 1 else node0
    ip1 = inventories[1]["ip"] if len(inventories) > 1 else ip0
    ip_by_node = {node: inventory["ip"] for node, inventory in zip(nodes, inventories)}
    job_by_node = {node: str(args.job_id) for node in nodes[:2]}
    if len(nodes) == 3:
        job_by_node[nodes[2]] = str(args.extra_job_id)
    workers = []
    node_group_counts = [
        getattr(args, "node0_groups", None) or args.groups_per_node,
    ]
    if len(nodes) > 1:
        node_group_counts.append(
            getattr(args, "node1_groups", None) or args.groups_per_node
        )
    if len(nodes) == 3:
        node_group_counts.append(1)
    for index, (node, ip) in enumerate(zip(nodes, (inventory["ip"] for inventory in inventories))):
        for group in range(node_group_counts[index]):
            engine_id = (
                f"collocated-{index}"
                if node_group_counts[index] == 1
                else f"collocated-{index}-{group}"
            )
            first_gpu = group * args.tp_size
            gpus = list(range(first_gpu, first_gpu + args.tp_size))
            port = args.worker_port + 10 * group
            workers.append(
                {
                    "node": node,
                    "job_id": job_by_node[node],
                    "engine_id": engine_id,
                    "local_dir": str(local_run / engine_id),
                    "environment": _base_env(args, engine_id, gpus),
                    "command": _server_command(args, "0.0.0.0", port, gpus),
                    "health": f"http://{ip}:{port}/model_info",
                    "node_index": index,
                    "group_index": group,
                    "port": port,
                    "gpus": gpus,
                }
            )
    router_env = _base_env(args, "router")
    router_env["CUDA_VISIBLE_DEVICES"] = ""
    router = {
        "node": node0,
        "engine_id": "router",
        "local_dir": str(local_run / "router"),
        "environment": router_env,
        "command": [
            python,
            "-m",
            "sglang_router.launch_router",
            "--worker-urls",
            *(f"http://{ip_by_node[worker['node']]}:{worker['port']}"
              for worker in workers),
            "--policy",
            "cache_aware",
            "--cache-threshold",
            "0.3",
            "--balance-abs-threshold",
            "8",
            "--balance-rel-threshold",
            "1.2",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.router_port),
            "--prometheus-port",
            str(getattr(args, "router_metrics_port", args.router_port + 1)),
        ],
        "health": f"http://{ip0}:{args.router_port}/health",
    }
    tunnel_command = [
        "/usr/bin/ssh",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
    ]
    for group in range(node_group_counts[1] if len(node_group_counts) > 1 else 0):
        remote_port = args.worker_port + 10 * group
        tunnel_port = args.metrics_tunnel_port + 10 * group
        tunnel_command.extend(("-L", f"127.0.0.1:{tunnel_port}:{ip1}:{remote_port}"))
    tunnel_command.append(node1)
    tunnel = {
        "node": node0,
        "engine_id": "metrics-tunnel",
        "local_dir": str(local_run / "metrics-tunnel"),
        "environment": {"PATH": "/usr/bin:/bin"},
        "command": tunnel_command,
    }
    workload_env = {
        "PATH": os.pathsep.join(
            (str(Path(workload_python).parent), "/usr/bin", os.environ.get("PATH", ""))
        ),
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PD_DATA_ROOT": str(args.data_root.resolve()),
        "PD_SWE_RUN_ID": args.run_id,
        "PD_SWE_PROGRESS_FILE": str(run_dir / "episode_progress.jsonl"),
        "PD_MODEL_HTTP_TRANSPORT": "aiohttp",
        "PD_INFERENCE_RETURN_LOGPROB": "false",
        "SLIME_HTTP_READ_TIMEOUT_SECONDS": "86400",
        "PD_SWE_APPTAINER_IMAGE_DIR": str(args.swe_sif_dir.resolve()),
        "PD_SWE_APPTAINER_WORK_ROOT": str(args.swe_work_root / args.run_id),
        "MIN_P": "0",
        "CUDA_VISIBLE_DEVICES": "",
    }
    local_metrics_ports = [
        args.worker_port + 10 * group for group in range(node_group_counts[0])
    ]
    remote_metrics_ports = [
        args.metrics_tunnel_port + 10 * group
        for group in range(node_group_counts[1] if len(node_group_counts) > 1 else 0)
    ]
    metrics_ports = ",".join(map(str, local_metrics_ports + remote_metrics_ports))
    metrics_endpoints = ",".join(
        f"http://{ip_by_node[worker['node']]}:{worker['port']}"
        for worker in workers
    )
    workload = {
        "node": getattr(args, "workload_node", "") or node0,
        **(
            {"job_id": str(args.workload_job_id)}
            if getattr(args, "workload_job_id", "")
            else {}
        ),
        "engine_id": "workload",
        "local_dir": str(local_run / "workload"),
        "environment": workload_env,
        "command": [
            workload_python,
            str(ROOT / "inference.py"),
            "--model",
            str(args.model.resolve()),
            "--workload-config",
            str(args.workload_config.resolve()),
            "--router-host",
            ip0,
            "--router-port",
            str(args.router_port),
            "--prefill-host",
            "127.0.0.1",
            "--prefill-port",
            str(local_metrics_ports[0]),
            "--prefill-ports",
            metrics_ports,
            "--decode-host",
            "127.0.0.1",
            "--decode-port",
            str(local_metrics_ports[0]),
            "--decode-ports",
            metrics_ports,
            "--prefill-endpoints",
            metrics_endpoints,
            "--decode-endpoints",
            metrics_endpoints,
            "--requests",
            str(args.requests),
            "--warmup-requests",
            "0",
            "--request-rate",
            "100",
            "--arrival-distribution",
            "fixed",
            "--dispatch-policy",
            "random",
            "--preserve-source-order",
            "--max-inflight",
            str(args.concurrency),
            "--metrics-interval",
            "2",
            "--seed",
            "2026",
            "--temperature",
            "0.6",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--max-context-length",
            "131072",
            "--max-response-length",
            "81920",
            "--output-dir",
            str(workload_output),
        ],
    }
    return {
        "schema_version": 1,
        "method": "native_colocated",
        "run_id": args.run_id,
        "job_id": str(args.job_id),
        "run_dir": str(run_dir),
        "model": str(args.model.resolve()),
        "baseline_python": python,
        "workload_python": workload_python,
        "settings": {
            "tp": args.tp_size,
            "ep": 1,
            "replicas": len(workers),
            "groups_per_node": args.groups_per_node,
            "node0_groups": node_group_counts[0],
            "node1_groups": node_group_counts[1] if len(node_group_counts) > 1 else 0,
            "node2_groups": node_group_counts[2] if len(node_group_counts) == 3 else 0,
            "mem_fraction_static": 0.8,
            "mamba_full_memory_ratio": 0.5,
            "context_length": 131072,
            "prefill_chunk": 8192,
            "max_prefill_tokens": 8192,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "source_order": True,
        },
        "components": {
            **{worker["engine_id"]: worker for worker in workers},
            "router": router,
            **({"metrics-tunnel": tunnel} if len(nodes) == 2 else {}),
            "workload": workload,
        },
    }


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("method") != "native_colocated":
        raise ValueError("not a colocated manifest")
    return value


def identity(manifest: dict, component: str) -> str:
    return f"dualpd-colocated:{manifest['run_id']}:{component}"


def raise_nofile_soft_limit(target: int = 65535) -> int:
    """Raise the inherited soft limit for concurrent SWE sandboxes."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired = int(target)
    if hard != resource.RLIM_INFINITY:
        desired = min(desired, int(hard))
    if desired <= 0:
        raise ValueError("open-file soft limit must be positive")
    if soft < desired:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    return desired


def lower_stack_soft_limit(target_kib: int = 2048) -> int:
    """Bound inherited thread reservations for the SWE container workload."""

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    desired = int(target_kib) * 1024
    if desired <= 0:
        raise ValueError("stack soft limit must be positive")
    if hard != resource.RLIM_INFINITY:
        desired = min(desired, int(hard))
    if soft == resource.RLIM_INFINITY or soft > desired:
        resource.setrlimit(resource.RLIMIT_STACK, (desired, hard))
    return desired


def component_action(manifest_path: Path, component: str, action: str) -> int:
    manifest = load_manifest(manifest_path)
    plan = manifest["components"][component]
    directory = Path(plan["local_dir"])
    if action in {"status", "stop"}:
        print(json.dumps(process_control(directory, identity(manifest, component), action), indent=2))
        return 0
    raise_nofile_soft_limit()
    lower_stack_soft_limit()
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("SGLANG_", "PD_", "NCCL_", "TORCH_NCCL_")):
            environment.pop(key, None)
    for key in ("SGLANG_OVERLAY_ROOT", "PYTHONPATH"):
        environment.pop(key, None)
    environment.update(plan["environment"])
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "launch.json", plan)
    return supervise(plan["command"], environment, directory, identity(manifest, component), stop_grace=120)


def detached_srun(args, manifest_path: Path, component: str) -> None:
    manifest = load_manifest(manifest_path)
    plan = manifest["components"][component]
    cpus = (
        args.cpus_per_node
        if component.startswith("collocated-")
        else max(8, min(64, args.cpus_per_node))
    )
    command = [
        "srun",
        "--jobid",
        str(plan.get("job_id", manifest["job_id"])),
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--nodelist",
        plan["node"],
        "--cpus-per-task",
        str(cpus),
        "--cpu-bind=none",
        "--export=ALL",
        sys.executable,
        str(Path(__file__).resolve()),
        "component",
        "--manifest",
        str(manifest_path),
        "--component",
        component,
        "--component-action",
        "start",
    ]
    launch_dir = Path(manifest["run_dir"]) / "launcher"
    launch_dir.mkdir(parents=True, exist_ok=True)
    with (launch_dir / f"{component}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    print(json.dumps({"component": component, "srun_pid": process.pid, "node": plan["node"]}))


def remote_control(args, manifest_path: Path, component: str, action: str) -> int:
    manifest = load_manifest(manifest_path)
    node = manifest["components"][component]["node"]
    command = [
        "srun",
        "--jobid",
        str(manifest["components"][component].get("job_id", manifest["job_id"])),
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--nodelist",
        node,
        "--cpus-per-task",
        "1",
        "--export=ALL",
        sys.executable,
        str(Path(__file__).resolve()),
        "component",
        "--manifest",
        str(manifest_path),
        "--component",
        component,
        "--component-action",
        action,
    ]
    return subprocess.call(command)


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            subprocess.run(
                ["curl", "-fsS", "--max-time", "3", url],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError:
            time.sleep(2)
    raise TimeoutError(f"timed out waiting for {url}")


def wait_router_workers(url: str, expected: int, timeout: float) -> None:
    """Wait for every asynchronous router worker registration to finish."""
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            completed = subprocess.run(
                ["curl", "-fsS", "--max-time", "3", url],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            workers = payload.get("workers", [])
            healthy = [worker for worker in workers if worker.get("is_healthy")]
            regular = payload.get("stats", {}).get("regular_count")
            last = f"total={payload.get('total')} healthy={len(healthy)} regular={regular}"
            if payload.get("total") == expected and len(healthy) == expected and regular == expected:
                return
        except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
            last = str(error)
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {expected} router workers at {url}: {last}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "start", "status", "stop", "component"))
    parser.add_argument("--job-id", default="")
    parser.add_argument("--extra-job-id", default="")
    parser.add_argument(
        "--workload-job-id",
        default="",
        help="optional separate Slurm allocation for the SWE harness",
    )
    parser.add_argument(
        "--workload-node",
        default="",
        help="node in --workload-job-id used by workload/verifier",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--component")
    parser.add_argument("--component-action", choices=("start", "status", "stop"))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--workload-python", default=DEFAULT_WORKLOAD_PYTHON)
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-parser", default="glm45")
    parser.add_argument("--deterministic-inference", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--swe-sif-dir", type=Path, default=DEFAULT_SIF)
    parser.add_argument("--swe-work-root", type=Path, default=Path("/var/tmp") / getpass.getuser() / "pd-swe-apptainer")
    parser.add_argument("--workload-config", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--ep-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument(
        "--mamba-buffer-strategy",
        choices=("no_buffer", "extra_buffer"),
        default="extra_buffer",
    )
    parser.add_argument(
        "--auto-kernel-backends",
        action="store_true",
        help="let SGLang select attention/MoE backends instead of forcing Triton",
    )
    parser.add_argument("--groups-per-node", type=int, default=1)
    parser.add_argument(
        "--single-node",
        action="store_true",
        help="place every colocated replica on the sole allocated H100 node",
    )
    parser.add_argument(
        "--node0-groups", type=int,
        help="TP groups on the first node; defaults to --groups-per-node",
    )
    parser.add_argument(
        "--node1-groups", type=int,
        help="TP groups on the second node; defaults to --groups-per-node",
    )
    parser.add_argument("--cpus-per-node", type=int, default=224)
    parser.add_argument("--worker-port", type=int, default=24000)
    parser.add_argument("--router-port", type=int, default=24002)
    parser.add_argument("--router-metrics-port", type=int, default=24003)
    parser.add_argument("--metrics-tunnel-port", type=int, default=24100)
    parser.add_argument("--ib-interface", default="ibp41s0f0")
    parser.add_argument("--ib-device", default="mlx5_1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "component":
        if not args.manifest or not args.component or not args.component_action:
            raise SystemExit("component mode requires manifest/component/action")
        return component_action(args.manifest.resolve(), args.component, args.component_action)
    if not args.job_id or not args.run_dir:
        raise SystemExit("--job-id and --run-dir are required")
    if bool(getattr(args, "workload_job_id", "")) != bool(
        getattr(args, "workload_node", "")
    ):
        raise SystemExit(
            "--workload-job-id and --workload-node must be supplied together"
        )
    args.node0_groups = getattr(args, "node0_groups", None) or args.groups_per_node
    args.node1_groups = getattr(args, "node1_groups", None) or args.groups_per_node
    for node_label, group_count in (
        ("node0", args.node0_groups), ("node1", args.node1_groups)
    ):
        if group_count < 1 or args.tp_size * group_count > 8:
            raise SystemExit(
                f"{node_label}: tp-size * groups-per-node must use between one and eight GPUs"
            )
    args.run_dir = args.run_dir.resolve()
    manifest_path = args.run_dir / "colocated-manifest.json"
    if args.action == "prepare":
        if not args.run_id:
            raise SystemExit("prepare requires --run-id")
        fields, nodes = site.allocation(
            args.job_id, min_nodes=1 if args.single_node else 2
        )
        selected = nodes[:1] if args.single_node else nodes[:2]
        job_by_node = {node: args.job_id for node in selected}
        if args.extra_job_id:
            _, extra_nodes = site.allocation(args.extra_job_id, min_nodes=1)
            if extra_nodes[0] in selected:
                raise RuntimeError("third replica must use a distinct node")
            selected.append(extra_nodes[0])
            job_by_node[extra_nodes[0]] = args.extra_job_id
        inventories = [
            site.node_inventory(job_by_node[node], node, args.ib_interface, args.ib_device)
            for node in selected
        ]
        if args.workload_job_id:
            _, workload_nodes = site.allocation(args.workload_job_id, min_nodes=1)
            if args.workload_node not in workload_nodes:
                raise RuntimeError(
                    f"workload node {args.workload_node} is not in Slurm job "
                    f"{args.workload_job_id}"
                )
        for path in (
            args.python,
            args.workload_python,
            args.model,
            args.data_root,
            args.swe_sif_dir,
            args.workload_config,
        ):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(path)
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        args.job_id = str(args.job_id)
        manifest = build_manifest(args, selected, inventories)
        manifest["job_end_time"] = fields.get("EndTime")
        _write_json(manifest_path, manifest)
        print(manifest_path)
        return 0
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_manifest(manifest_path)
    args.job_id = manifest["job_id"]
    workers = tuple(
        name for name in manifest["components"] if name.startswith("collocated-")
    )
    components = (*workers, "router", *(("metrics-tunnel",) if "metrics-tunnel" in manifest["components"] else ()), "workload")
    if args.action == "start":
        started = []
        try:
            for component in workers:
                detached_srun(args, manifest_path, component)
                started.append(component)
            for component in workers:
                wait_http(manifest["components"][component]["health"], 1800)
            detached_srun(args, manifest_path, "router")
            started.append("router")
            wait_http(manifest["components"]["router"]["health"], 300)
            router_health = manifest["components"]["router"]["health"]
            wait_router_workers(
                router_health.rsplit("/", 1)[0] + "/workers", len(workers), 300
            )
            if "metrics-tunnel" in manifest["components"]:
                detached_srun(args, manifest_path, "metrics-tunnel")
                started.append("metrics-tunnel")
                time.sleep(3)
            detached_srun(args, manifest_path, "workload")
            started.append("workload")
        except BaseException:
            for component in reversed(started):
                try:
                    remote_control(args, manifest_path, component, "stop")
                except BaseException:
                    pass
            raise
        return 0
    if args.action == "status":
        result = 0
        for component in components:
            result |= remote_control(args, manifest_path, component, "status")
        return result
    result = 0
    for component in reversed(components):
        try:
            result |= remote_control(args, manifest_path, component, "stop")
        except (OSError, subprocess.SubprocessError):
            result |= 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
