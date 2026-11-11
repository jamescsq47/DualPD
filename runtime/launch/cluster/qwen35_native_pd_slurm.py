#!/usr/bin/env python3
"""Run stock SGLang Qwen3.5 PD baselines on allocated H100 nodes.

The default is the one-way native NIXL P->D baseline.  ``--native-mooncake``
instead enables native HiCache/Mooncake decode offload and prefill restore.
No Agentic-PD module is imported in either mode.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import resource
import shlex
import subprocess
import sys
import time

if __package__:
    from .process_supervisor import control as process_control, supervise
    from . import qwen35_slurm as site
else:
    from process_supervisor import control as process_control, supervise
    import qwen35_slurm as site


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
DEFAULT_PYTHON = os.environ.get("BASELINE_PYTHON", sys.executable)
DEFAULT_WORKLOAD_PYTHON = sys.executable
DEFAULT_SGLANG_ROOT = Path(os.environ.get("BASELINE_ENGINE_ROOT", ROOT.parent / "sglang-baseline"))
DEFAULT_MODEL = ROOT.parent / "models/Qwen3.5-122B-A10B"
DEFAULT_DATA = ROOT.parent / "datasets"
DEFAULT_SIF = ROOT.parent / "containers/swebench-verified-sif"
DEFAULT_WORKLOAD = (
    ROOT
    / "configs/experiments/"
    "swe_bench_verified_apptainer.yaml"
)
DEFAULT_LOCAL_ROOT = Path("/var/tmp") / getpass.getuser() / "dualpd-native-pd"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def base_env(args, engine_id: str, gpus: list[int] | None = None) -> dict[str, str]:
    env_bin = str(Path(args.python).resolve().parent)
    cache = args.local_root / "compiler-cache" / engine_id
    visible = gpus if gpus is not None else []
    environment = {
        "PATH": os.pathsep.join((env_bin, "/usr/local/cuda/bin", "/usr/bin", "/bin")),
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
        # Mooncake embeds a Go runtime in every TP rank.  Leaving GOMAXPROCS
        # at the 224-core node default lets each rank create hundreds of OS
        # threads under blocked cgo storage calls; with eight ranks plus the
        # SWE verifier this eventually makes pthread_create fail with EAGAIN.
        # HiCache I/O is serialized per rank below the Python boundary, so a
        # small Go scheduler pool preserves transfer concurrency while keeping
        # thread count independent of node CPU count.
        "GOMAXPROCS": "4",
        # Each H100 worker owns hundreds of long-lived CUDA/Mooncake/LLVM
        # threads.  Bound glibc's per-thread arena expansion and reserve a
        # 2 MiB stack so high-concurrency HiCache runs do not exhaust Linux's
        # overcommit budget while Triton compiles a previously unseen kernel.
        "MALLOC_ARENA_MAX": "2",
        "MALLOC_ARENA_TEST": "2",
        "SGLANG_STACK_SOFT_LIMIT_KIB": "2048",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, visible)),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        # Upstream v0.5.21 builds the HiCache hash helper lazily through
        # torch.utils.cpp_extension.  Its default cache follows $HOME on this
        # cluster, where the shared filesystem can return EIO for flock(2)
        # and permanently kill the HiCache prefetch thread.  Keep the
        # upstream implementation unchanged, but put its build directory and
        # lock on node-local storage.  All engines on a node use the same
        # upstream extension, so share one local build instead of compiling
        # it once per TP group; the two physical nodes have distinct /var/tmp.
        "TORCH_EXTENSIONS_DIR": str(
            args.local_root / "compiler-cache" / "torch-extensions"
        ),
        "FLASHINFER_WORKSPACE_BASE": str(cache / "flashinfer"),
        "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "true",
        "UCX_NET_DEVICES": f"{args.ib_device}:1",
        "NCCL_SOCKET_IFNAME": args.ib_interface,
        "GLOO_SOCKET_IFNAME": args.ib_interface,
    }
    if getattr(args, "disable_mlir_multithreading", False):
        # MLIR otherwise creates a fresh LLVM thread pool when a previously
        # unseen Triton kernel is compiled.  A HiCache run can reach that path
        # only after its large Host tiers have consumed most of CommitLimit,
        # where pthread_create fails even though physical RAM remains.  Serial
        # compilation changes only compile latency, not generated code, the
        # serving topology, or cache semantics.
        environment["MLIR_DISABLE_MULTITHREADING"] = "1"
    if not getattr(args, "use_installed_sglang", False):
        environment["PYTHONPATH"] = str(args.sglang_root.resolve() / "python")
    return environment


def common_server_command(
    args, role: str, host: str, port: int, gpus: list[int]
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
        (
            "--mamba-radix-cache-strategy"
            if getattr(args, "mamba_cache_strategy_flag", "scheduler") == "radix"
            else "--mamba-scheduler-strategy"
        ),
        (
            getattr(args, "mamba_buffer_strategy", None)
            if getattr(args, "mamba_buffer_strategy", None) is not None
            else (
                "extra_buffer"
                if role == "decode" or getattr(args, "native_mooncake", False)
                else "no_buffer"
            )
        ),
        "--mamba-track-interval",
        "64",
        "--mamba-full-memory-ratio",
        str(getattr(args, "mamba_full_memory_ratio", 0.9)),
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
        *(
            ["--disable-overlap-schedule"]
            if getattr(args, "disable_overlap_schedule", False)
            else []
        ),
        "--enable-metrics",
        "--skip-server-warmup",
        "--uvicorn-access-log-exclude-prefixes",
        "/get_load",
        "/metrics",
        "/health",
        "--disaggregation-transfer-backend",
        "nixl",
        "--disaggregation-ib-device",
        args.ib_device,
    ]


def mooncake_extra_config(args, local_hostname: str) -> str:
    """Return the stock Mooncake storage config shared by P and D ranks."""

    storage_ip = args.storage_ip
    return json.dumps(
        {
            "master_server_address": f"{storage_ip}:{args.mooncake_master_port}",
            "local_hostname": local_hostname,
            "metadata_server": (
                f"http://{storage_ip}:{args.mooncake_metadata_port}/metadata"
            ),
            "global_segment_size": "0",
            "protocol": args.mooncake_protocol,
            "device_name": "",
            "prefetch_threshold": args.mooncake_prefetch_threshold,
            "prefetch_timeout_base": 5,
            "prefetch_timeout_per_ki_token": 0.5,
        },
        separators=(",", ":"),
    )


def environment_cuda_runtime_lib(python: Path) -> Path:
    """Locate CUDA runtime libraries in the selected Python environment.

    The launcher itself can run under a different Python minor version from
    the SGLang environment (for example, login Python 3.14 and baseline Python
    3.12).  Deriving this path from ``sys.version_info`` therefore points
    Mooncake at a directory that does not exist.
    """

    env_root = Path(python).resolve().parent.parent
    candidates = sorted(
        path
        for path in (env_root / "lib").glob("python3.*")
        if (path / "site-packages").is_dir()
    )
    if not candidates:
        candidates = [
            env_root
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
        ]
    return candidates[-1] / "site-packages/nvidia/cuda_runtime/lib"


def build_manifest(args, nodes: list[str], inventories: list[dict[str, str]]) -> dict:
    if not SAFE_NAME.fullmatch(args.run_id):
        raise ValueError("unsafe run ID")
    run_dir = args.run_dir.resolve()
    local_run = args.local_root / args.run_id
    p_node, d_node = nodes[0], nodes[-1]
    p_ip, d_ip = inventories[0]["ip"], inventories[-1]["ip"]
    native_mooncake = bool(getattr(args, "native_mooncake", False))
    job_by_node = {node: str(args.job_id) for node in nodes[:-1]}
    job_by_node[d_node] = str(args.extra_job_id if len(nodes) == 3 else args.job_id)
    python = str(Path(args.python).resolve())
    workload_python = str(Path(args.workload_python).resolve())
    components: dict[str, dict] = {}
    if native_mooncake:
        storage_node = args.storage_node
        storage_ip = args.storage_ip
        storage_job_id = str(args.storage_job_id)
        env_bin = Path(args.python).resolve().parent
        master_binary_override = getattr(args, "mooncake_master_binary", None)
        master_binary = (
            Path(master_binary_override).resolve()
            if master_binary_override is not None
            else env_bin / "mooncake_master"
        )
        service_env = base_env(args, "mooncake-services")
        service_env["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
        cuda_runtime_lib = environment_cuda_runtime_lib(Path(args.python))
        library_paths = [str(cuda_runtime_lib)]
        if master_binary_override is not None:
            # The isolated source build links against the system libcurl.  Put
            # the system directory ahead of its build-time Conda RPATH without
            # changing the stock environment used by all other experiments.
            library_paths.insert(0, "/usr/lib/x86_64-linux-gnu")
        service_env["LD_LIBRARY_PATH"] = ":".join(library_paths)
        components["mooncake-master"] = {
            "node": storage_node,
            "job_id": storage_job_id,
            "engine_id": "mooncake-master",
            "local_dir": str(local_run / "mooncake-master"),
            "environment": service_env,
            "command": [
                str(master_binary),
                f"--rpc_port={args.mooncake_master_port}",
                "--enable_http_metadata_server=true",
                f"--http_metadata_server_port={args.mooncake_metadata_port}",
                "--eviction_high_watermark_ratio=0.90",
                # Retain the lower eviction quantum used by the preceding
                # c256 diagnostics.  The optional master override fixes the
                # independent EAGAIN issue by reusing a bounded worker pool.
                "--eviction_ratio=0.05",
                f"--metrics_port={args.mooncake_metrics_port}",
            ],
            "health": f"http://{storage_ip}:{args.mooncake_metrics_port}/health",
        }
        components["mooncake-client"] = {
            "node": storage_node,
            "job_id": storage_job_id,
            "engine_id": "mooncake-client",
            "local_dir": str(local_run / "mooncake-client"),
            "environment": service_env,
            "command": [
                str(env_bin / "mooncake_client"),
                f"--host={storage_ip}",
                f"--port={args.mooncake_client_port}",
                f"--global_segment_size={args.mooncake_store_size_gb} GB",
                f"--master_server_address={storage_ip}:{args.mooncake_master_port}",
                f"--metadata_server=http://{storage_ip}:{args.mooncake_metadata_port}/metadata",
                f"--protocol={args.mooncake_protocol}",
                "--threads=16",
                "--enable_http_server=true",
                f"--http_port={args.mooncake_client_http_port}",
            ],
            "health": f"http://{storage_ip}:{args.mooncake_client_http_port}/health",
        }
    prefill_ports = []
    decode_ports = []
    bootstrap_ports = []
    prefill_endpoints = []
    decode_endpoints = []
    prefill_groups = (
        getattr(args, "prefill_groups_per_node", None) or args.groups_per_node
    )
    decode_groups = (
        getattr(args, "decode_groups_per_node", None) or args.groups_per_node
    )

    mixed_one_prefill_per_node = bool(
        getattr(args, "mixed_one_prefill_per_node", False)
    )
    single_node_1p3d = bool(getattr(args, "single_node_1p3d", False))
    single_node_2p6d = bool(getattr(args, "single_node_2p6d", False))
    single_node_2p2d = bool(getattr(args, "single_node_2p2d", False))
    if single_node_1p3d:
        placements = [
            ("prefill", p_node, p_ip, args.prefill_port, 1, 0),
            ("decode", p_node, p_ip, args.decode_port, 3, 2),
        ]
    elif single_node_2p2d:
        placements = [
            ("prefill", p_node, p_ip, args.prefill_port, 1, 0),
            ("decode", p_node, p_ip, args.decode_port, 1, 2),
            ("prefill", p_node, p_ip, args.prefill_port + 10, 1, 4),
            ("decode", p_node, p_ip, args.decode_port + 10, 1, 6),
        ]
    elif single_node_2p6d:
        # One-node TP1 layout used by the Qwen3.5-9B SWE500 matrix.  Split
        # each NUMA domain between one P and three D replicas, matching the
        # established A100 2P:6D placement while keeping every listener unique.
        placements = [
            ("prefill", p_node, p_ip, args.prefill_port, 1, 0),
            ("prefill", p_node, p_ip, args.prefill_port + 10, 1, 4),
            ("decode", p_node, p_ip, args.decode_port, 3, 1),
            ("decode", p_node, p_ip, args.decode_port + 30, 3, 5),
        ]
    elif mixed_one_prefill_per_node:
        # Two-node 2P:6D layout for TP=2: every physical node owns one
        # Prefill group on GPUs 0-1 and three Decode groups on GPUs 2-7.
        # Ports may repeat across hosts; endpoint URLs remain unique.
        placements = []
        for node, inventory in zip(nodes, inventories, strict=True):
            placements.append(
                ("prefill", node, inventory["ip"], args.prefill_port, 1, 0)
            )
            placements.append(
                (
                    "decode", node, inventory["ip"], args.decode_port,
                    decode_groups, args.tp_size,
                )
            )
    else:
        placements = [
            (
                "prefill", node, inventory["ip"], args.prefill_port,
                prefill_groups if len(nodes) == 2 else 1, 0,
            )
            for node, inventory in zip(nodes[:-1], inventories[:-1])
        ] + [("decode", d_node, d_ip, args.decode_port, decode_groups, 0)]
    for role, node, ip, base_port, group_count, gpu_offset in placements:
        for group in range(group_count):
            engine_id = f"{role}-{len(prefill_endpoints) if role == 'prefill' else len(decode_endpoints)}"
            gpu_start = gpu_offset + group * args.tp_size
            gpus = list(range(gpu_start, gpu_start + args.tp_size))
            port = base_port + 10 * group
            command = common_server_command(args, role, "0.0.0.0", port, gpus)
            command.extend(("--disaggregation-mode", role))
            environment = base_env(args, engine_id, gpus)
            if native_mooncake:
                # Compile the hybrid GDN/Triton kernels with SGLang's stock
                # fake-bootstrap PD warmup before the high-concurrency SWE
                # workload and Mooncake population consume the node's commit
                # budget.  Health is published only after this finishes.
                command.remove("--skip-server-warmup")
                if args.mooncake_compat is not None:
                    python_paths = [str(args.mooncake_compat.resolve())]
                    if not getattr(args, "use_installed_sglang", False):
                        python_paths.append(
                            str(args.sglang_root.resolve() / "python")
                        )
                    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
                environment["MC_TCP_ENABLE_CONNECTION_POOL"] = "1"
                environment["SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB"] = str(
                    args.mamba_hicache_size_gb
                )
                environment["LD_LIBRARY_PATH"] = str(
                    environment_cuda_runtime_lib(Path(args.python))
                )
                hicache_args = [
                    # Use SGLang's native direct-copy layout for the hybrid
                    # Attention/Mamba pools.  Besides being the recommended
                    # Qwen3.5 pairing, this avoids the legacy page_first
                    # Triton index-copy path while retaining Mooncake's
                    # page-wise zero-copy storage interface.
                    "--hicache-mem-layout", "page_first_direct",
                    "--hicache-io-backend", "direct",
                    "--hicache-storage-backend", "mooncake",
                    "--hicache-storage-backend-extra-config",
                    mooncake_extra_config(args, ip),
                ]
                if role == "prefill":
                    command.extend(
                        [
                            "--enable-hierarchical-cache",
                            "--hicache-size", str(args.prefill_hicache_size_gb),
                            "--hicache-write-policy", "write_through",
                            "--hicache-storage-prefetch-policy",
                            args.mooncake_prefetch_policy,
                            *hicache_args,
                        ]
                    )
                else:
                    command.extend(
                        [
                            "--disaggregation-decode-enable-offload-kvcache",
                            "--hicache-size", str(args.decode_hicache_size_gb),
                            *hicache_args,
                        ]
                    )
            if role == "prefill":
                bootstrap = args.bootstrap_port + 10 * len(bootstrap_ports)
                prefill_ports.append(port)
                prefill_endpoints.append(f"http://{ip}:{port}")
                bootstrap_ports.append(bootstrap)
                command.extend(
                    ("--disaggregation-bootstrap-port", str(bootstrap))
                )
                if not native_mooncake:
                    command.append("--disable-radix-cache")
            else:
                decode_ports.append(port)
                decode_endpoints.append(f"http://{ip}:{port}")
            components[engine_id] = {
                "node": node,
                "job_id": job_by_node[node],
                "engine_id": engine_id,
                "role": role,
                "local_dir": str(local_run / engine_id),
                "environment": environment,
                "command": command,
                "health": f"http://{ip}:{port}/health",
                "port": port,
                "gpus": gpus,
            }

    router_command = [
        python,
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.router_port),
        "--prometheus-port",
        str(args.router_metrics_port),
        "--policy",
        getattr(args, "router_policy", "round_robin"),
        # A saturated TP8 decode scheduler can delay the Python health reply
        # even while Forward is making progress.  Keep health checking, but do
        # not eject the only decode worker after three transient 5 s misses.
        "--health-check-timeout-secs",
        "30",
        "--health-failure-threshold",
        "5",
        # NIXL's native receiver admission timeout is 300 s.  Bound the outer
        # request slightly above it so a lost P/D handoff cannot retain a
        # workload slot for the router's 1800 s default after the Decode side
        # has already forgotten the transfer.
        "--request-timeout-secs",
        "360",
        # A P->D NIXL receiver timeout is a per-request admission failure, not
        # evidence that the sole TP8 Prefill/Decode endpoint is dead.  The
        # router's default circuit breaker otherwise turns ten such timeouts
        # into a 60-second global 503 storm even while both workers keep
        # serving successful requests.  Preserve the native 300 s transfer
        # timeout (and its observable retry cost), but do not amplify it into
        # endpoint-wide downtime.
        "--disable-circuit-breaker",
    ]
    for endpoint, bootstrap in zip(prefill_endpoints, bootstrap_ports, strict=True):
        router_command.extend(("--prefill", endpoint, str(bootstrap)))
    for endpoint in decode_endpoints:
        router_command.extend(("--decode", endpoint))
    router_env = base_env(args, "router")
    components["router"] = {
        "node": p_node,
        "engine_id": "router",
        "local_dir": str(local_run / "router"),
        "environment": router_env,
        "command": router_command,
        "health": f"http://{p_ip}:{args.router_port}/health",
    }

    workload_env = {
        "PATH": os.pathsep.join((str(Path(workload_python).parent), "/usr/bin", "/bin")),
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
    components["workload"] = {
        "node": getattr(args, "workload_node", "") or p_node,
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
            str(ROOT / "runtime/inference.py"),
            "--model", str(args.model.resolve()),
            "--workload-config", str(args.workload_config.resolve()),
            "--router-host", p_ip,
            "--router-port", str(args.router_port),
            "--prefill-host", p_ip,
            "--prefill-port", str(prefill_ports[0]),
            "--prefill-ports", ",".join(map(str, prefill_ports)),
            "--decode-host", d_ip,
            "--decode-port", str(decode_ports[0]),
            "--decode-ports", ",".join(map(str, decode_ports)),
            "--prefill-endpoints", ",".join(prefill_endpoints),
            "--decode-endpoints", ",".join(decode_endpoints),
            "--requests", str(args.requests),
            "--warmup-requests", "0",
            "--request-rate", "100",
            "--arrival-distribution", "fixed",
            "--dispatch-policy", "random",
            "--preserve-source-order",
            "--max-inflight", str(args.concurrency),
            "--metrics-interval", "2",
            "--seed", "2026",
            "--temperature", "0.6",
            "--top-p", "0.95",
            "--top-k", "20",
            "--max-context-length", "131072",
            "--max-response-length", "81920",
            "--output-dir", str(run_dir / "workload"),
        ],
    }
    if native_mooncake:
        components["workload"]["command"].extend(
            (
                "--pd-enable-decode-offload-kvcache",
                "--pd-hicache-storage-backend", "mooncake",
                "--pd-hicache-storage-dir", str(run_dir / "hicache"),
                "--pd-hicache-storage-prefetch-policy",
                args.mooncake_prefetch_policy,
                "--pd-hicache-prefetch-threshold",
                str(args.mooncake_prefetch_threshold),
            )
        )
    return {
        "schema_version": 1,
        "method": (
            "native_pd_hicache_mooncake" if native_mooncake
            else "native_pd_no_reverse"
        ),
        "run_id": args.run_id,
        "job_id": str(args.job_id),
        "launch_transport": getattr(args, "launch_transport", "srun"),
        "run_dir": str(run_dir),
        "model": str(args.model.resolve()),
        "baseline_python": python,
        "sglang_root": str(args.sglang_root.resolve()),
        "use_installed_sglang": bool(
            getattr(args, "use_installed_sglang", False)
        ),
        "workload_python": workload_python,
        "nodes": {
            "prefill": list(nodes) if mixed_one_prefill_per_node else (
                list(nodes[:-1]) if len(nodes) == 3 else p_node
            ),
            "decode": list(nodes) if mixed_one_prefill_per_node else d_node,
        },
        "addresses": {
            "prefill": [inventory["ip"] for inventory in inventories]
            if mixed_one_prefill_per_node else (
                [inventory["ip"] for inventory in inventories[:-1]]
                if len(nodes) == 3 else p_ip
            ),
            "decode": [inventory["ip"] for inventory in inventories]
            if mixed_one_prefill_per_node else d_ip,
        },
        "settings": {
            "tp": args.tp_size,
            "ep": 1,
            "prefill_groups": len(prefill_endpoints),
            "decode_groups": len(decode_endpoints),
            "mixed_one_prefill_per_node": mixed_one_prefill_per_node,
            "single_node_1p3d": single_node_1p3d,
            "single_node_2p6d": single_node_2p6d,
            "single_node_2p2d": single_node_2p2d,
            "mem_fraction_static": 0.8,
            "mamba_full_memory_ratio": getattr(
                args, "mamba_full_memory_ratio", 0.9
            ),
            "disable_overlap_schedule": bool(
                getattr(args, "disable_overlap_schedule", False)
            ),
            "context_length": 131072,
            "prefill_chunk": 8192,
            "max_prefill_tokens": 8192,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "source_order": True,
            "router_policy": getattr(args, "router_policy", "round_robin"),
            "compatibility_profile": (
                "v0.5.21_qwen35_hybrid_fixed_tp_workset"
                if native_mooncake and args.mooncake_compat is not None
                else "none"
            ),
            "p_to_d": "native_nixl",
            "d_to_p": "native_hicache_mooncake" if native_mooncake else "disabled",
            "prefill_radix_cache": native_mooncake,
            "hicache": native_mooncake,
            "mooncake": native_mooncake,
            "prefill_hicache_size_gb_per_rank": (
                args.prefill_hicache_size_gb if native_mooncake else 0
            ),
            "decode_hicache_size_gb_per_rank": (
                args.decode_hicache_size_gb if native_mooncake else 0
            ),
            "mamba_hicache_size_gb_per_rank": (
                args.mamba_hicache_size_gb if native_mooncake else 0
            ),
            "mooncake_store_size_gb": (
                args.mooncake_store_size_gb if native_mooncake else 0
            ),
        },
        "components": components,
    }


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("method") not in {
        "native_pd_no_reverse", "native_pd_hicache_mooncake"
    }:
        raise ValueError("not a native PD manifest")
    return value


def identity(manifest: dict, component: str) -> str:
    return f"dualpd-native-pd:{manifest['run_id']}:{component}"


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
        try:
            result = process_control(directory, identity(manifest, component), action)
        except (ConnectionRefusedError, FileNotFoundError):
            # A completed workload leaves an auditable process record but no
            # live supervisor socket.  Treat that as already stopped so a
            # best-effort run cleanup remains idempotent.
            result = {"ok": True, "running": False, "already_stopped": True}
        print(json.dumps(result, indent=2))
        return 0
    raise_nofile_soft_limit()
    lower_stack_soft_limit(
        int(plan["environment"].get("SGLANG_STACK_SOFT_LIMIT_KIB", "2048"))
    )
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("SGLANG_", "PD_", "NCCL_", "TORCH_NCCL_")):
            environment.pop(key, None)
    for key in ("SGLANG_OVERLAY_ROOT", "PYTHONPATH"):
        environment.pop(key, None)
    environment.update(plan["environment"])
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "launch.json", plan)
    return supervise(
        plan["command"], environment, directory,
        identity(manifest, component), stop_grace=120,
    )


def detached_srun(args, manifest_path: Path, component: str) -> None:
    manifest = load_manifest(manifest_path)
    plan = manifest["components"][component]
    cpus = args.cpus_per_worker if component.startswith(("prefill-", "decode-")) else 32
    command = [
        "srun", "--jobid", str(plan.get("job_id", manifest["job_id"])), "--overlap", "--nodes=1",
        "--ntasks=1", "--nodelist", plan["node"], "--cpus-per-task", str(cpus),
        "--cpu-bind=none", "--export=ALL", sys.executable,
        str(Path(__file__).resolve()), "component", "--manifest", str(manifest_path),
        "--component", component, "--component-action", "start",
    ]
    launch_dir = Path(manifest["run_dir"]) / "launcher"
    launch_dir.mkdir(parents=True, exist_ok=True)
    with (launch_dir / f"{component}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    print(json.dumps({"component": component, "srun_pid": process.pid, "node": plan["node"]}))


def ssh_options() -> list[str]:
    return [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
    ]


def ssh_component_command(manifest_path: Path, component: str, action: str) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "component",
        "--manifest", str(manifest_path), "--component", component,
        "--component-action", action,
    ]


def detached_ssh(args, manifest_path: Path, component: str) -> None:
    """Start a component from the allocated node's sshd, independent of this session.

    The Slurm allocations remain the authority that makes the nodes exclusive.
    This transport is only for clusters where a long-lived child ``srun`` is
    cancelled when the login/controller session disappears.  ``setsid -f``
    makes the remote supervisor a child of sshd/init rather than of the ssh
    session, while the supervisor's authenticated socket remains the sole stop
    mechanism.
    """
    manifest = load_manifest(manifest_path)
    plan = manifest["components"][component]
    directory = Path(plan["local_dir"])
    launch_log = directory / "ssh-launch.log"
    command = ssh_component_command(manifest_path, component, "start")
    remote = " ".join((
        "umask 077;",
        "mkdir -p", shlex.quote(str(directory)) + ";",
        "nohup setsid -f", shlex.join(command),
        "</dev/null", ">>" + shlex.quote(str(launch_log)), "2>&1",
    ))
    completed = subprocess.run(
        ["ssh", *ssh_options(), plan["node"], remote],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ssh launch failed for {component} on {plan['node']}: "
            f"{completed.stderr.strip()}"
        )
    print(json.dumps({"component": component, "transport": "ssh", "node": plan["node"]}))


def remote_control(args, manifest_path: Path, component: str, action: str) -> int:
    manifest = load_manifest(manifest_path)
    node = manifest["components"][component]["node"]
    command = [
        "srun", "--jobid", str(manifest["components"][component].get("job_id", manifest["job_id"])), "--overlap", "--nodes=1",
        "--ntasks=1", "--nodelist", node, "--cpus-per-task", "1",
        "--export=ALL", sys.executable, str(Path(__file__).resolve()), "component",
        "--manifest", str(manifest_path), "--component", component,
        "--component-action", action,
    ]
    return subprocess.call(command)


def remote_control_ssh(manifest_path: Path, component: str, action: str) -> int:
    manifest = load_manifest(manifest_path)
    plan = manifest["components"][component]
    return subprocess.call([
        "ssh", *ssh_options(), plan["node"],
        shlex.join(ssh_component_command(manifest_path, component, action)),
    ])


def launch_component(args, manifest_path: Path, component: str) -> None:
    if args.launch_transport == "ssh":
        detached_ssh(args, manifest_path, component)
    else:
        detached_srun(args, manifest_path, component)


def control_component(
    args, manifest_path: Path, component: str, action: str,
) -> int:
    if args.launch_transport == "ssh":
        return remote_control_ssh(manifest_path, component, action)
    return remote_control(args, manifest_path, component, action)


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["curl", "-fsS", "--max-time", "3", url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if completed.returncode == 0:
            return
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {url}")


def drop_model_weight_page_cache(manifest: dict) -> None:
    """Release clean model-file pages after both workers have loaded.

    The TP8 Qwen3.5 workers have already copied their weights to GPU by this
    point.  Keeping another roughly 240 GiB of clean safetensors pages in the
    Decode node's page cache leaves too little of its 1000 GB Slurm cgroup for
    the fixed 512 GiB Mooncake tier plus the hybrid Attention/Mamba Host pools.
    POSIX_FADV_DONTNEED is local, advisory, read-only, and leaves mapped or
    dirty pages alone; it does not alter the model or any inference setting.
    """
    script = r"""
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path.suffix in {".safetensors", ".bin"}
)
advised = 0
failed = []
for path in files:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        advised += path.stat().st_size
    except OSError as error:
        failed.append({"path": str(path), "error": str(error)})
    finally:
        os.close(fd)
print(json.dumps({"files": len(files), "advised_bytes": advised, "failed": failed}))
"""
    nodes = sorted(
        {
            component["node"]
            for name, component in manifest["components"].items()
            if name.startswith(("prefill-", "decode-"))
        }
    )
    for node in nodes:
        command = [
            str(Path(manifest["baseline_python"]).resolve()),
            "-c",
            script,
            manifest["model"],
        ]
        if manifest.get("launch_transport") == "ssh":
            remote_command = ["ssh", *ssh_options(), node, shlex.join(command)]
        else:
            remote_command = [
                "srun",
                "--jobid",
                str(manifest["job_id"]),
                "--overlap",
                "--nodes=1",
                "--ntasks=1",
                "--nodelist",
                node,
                "--cpus-per-task",
                "1",
                "--export=ALL",
                *command,
            ]
        completed = subprocess.run(
            remote_command, check=True, capture_output=True, text=True
        )
        result = json.loads(completed.stdout)
        if result["failed"]:
            raise RuntimeError(
                f"model page-cache release failed on {node}: {result['failed']}"
            )
        print(
            json.dumps(
                {
                    "component": "model-page-cache",
                    "node": node,
                    "files": result["files"],
                    "advised_gib": round(result["advised_bytes"] / 2**30, 2),
                }
            )
        )


def verify_baseline_environment(args) -> dict:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("SGLANG_", "PD_", "NCCL_", "TORCH_NCCL_")):
            environment.pop(key, None)
    environment.pop("PYTHONPATH", None)
    if not getattr(args, "use_installed_sglang", False):
        environment["PYTHONPATH"] = str(args.sglang_root.resolve() / "python")
    environment["PYTHONNOUSERSITE"] = "1"
    checker = ROOT / "runtime/launch/environment.py"
    completed = subprocess.run(
        [str(Path(args.python).resolve()), str(checker), "--expect", "baseline"],
        env=environment, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


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
    parser.add_argument(
        "--storage-job-id",
        default="",
        help=(
            "optional allocation for the Mooncake store; when omitted the "
            "store follows the workload allocation (or the Prefill node if "
            "the workload has no separate allocation)"
        ),
    )
    parser.add_argument(
        "--storage-node",
        default="",
        help="node in --storage-job-id used by Mooncake master/client",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--component")
    parser.add_argument("--component-action", choices=("start", "status", "stop"))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--workload-python", default=DEFAULT_WORKLOAD_PYTHON)
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    parser.add_argument(
        "--use-installed-sglang",
        action="store_true",
        help=(
            "run the SGLang package installed in --python instead of overlaying "
            "--sglang-root/python; useful for official images with bundled Rust "
            "extensions while retaining --sglang-root as audited source provenance"
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-parser", default="glm45")
    parser.add_argument(
        "--mamba-full-memory-ratio",
        type=float,
        default=0.9,
        help=(
            "GPU-memory split used by the stock/baseline SGLang Mamba pool; "
            "baseline keeps the upstream 0.9 default"
        ),
    )
    parser.add_argument(
        "--mamba-cache-strategy-flag",
        choices=("scheduler", "radix"),
        default="radix",
        help=(
            "select only the SGLang CLI spelling for the Mamba cache strategy; "
            "this launcher's official v0.5.21 baseline defaults to "
            "--mamba-radix-cache-strategy, while older baselines can opt into "
            "--mamba-scheduler-strategy"
        ),
    )
    parser.add_argument("--deterministic-inference", action="store_true")
    parser.add_argument(
        "--disable-mlir-multithreading",
        action="store_true",
        help=(
            "compile late Triton kernels without creating an LLVM worker pool; "
            "useful when large Host cache mappings approach CommitLimit"
        ),
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help=(
            "serialize scheduler Forward/result processing; useful for "
            "auditing hybrid Mamba checkpoint ownership with HiCache"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--swe-sif-dir", type=Path, default=DEFAULT_SIF)
    parser.add_argument("--swe-work-root", type=Path, default=Path("/var/tmp") / getpass.getuser() / "pd-swe-apptainer")
    parser.add_argument("--workload-config", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument(
        "--native-mooncake", action="store_true",
        help="enable stock HiCache/Mooncake D-to-P restore",
    )
    parser.add_argument(
        "--mooncake-compat", type=Path, default=None,
        help="optional legacy Qwen3.5 compatibility layer",
    )
    parser.add_argument(
        "--mooncake-master-binary", type=Path, default=None,
        help=(
            "optional isolated mooncake_master executable; only the master "
            "component is replaced, while the selected Python environment and "
            "all worker/client binaries remain unchanged"
        ),
    )
    # Hybrid Qwen3.5 allocates this amount independently for Attention KV and
    # Mamba state.  Stock HiCache also requires each host Attention pool to be
    # larger than the corresponding ~22 GiB GPU pool, so 32 GiB is the minimum
    # safe round value and consumes about 512 GiB per TP8 node in total.
    parser.add_argument("--prefill-hicache-size-gb", type=int, default=32)
    parser.add_argument("--decode-hicache-size-gb", type=int, default=32)
    parser.add_argument(
        "--mamba-hicache-size-gb",
        type=int,
        default=32,
        help=(
            "separate Qwen3.5 Mamba snapshot pool per rank; keeping this "
            "independent avoids coupling recurrent-state capacity to the KV "
            "memory cgroup"
        ),
    )
    parser.add_argument("--mooncake-store-size-gb", type=int, default=512)
    parser.add_argument("--mooncake-prefetch-threshold", type=int, default=64)
    parser.add_argument(
        "--mooncake-prefetch-policy",
        choices=("wait_complete", "timeout", "best_effort"),
        default="wait_complete",
        help=(
            "Qwen3.5 hybrid KV/Mamba restore must finish both pools before "
            "the request is admitted to Forward"
        ),
    )
    parser.add_argument("--mooncake-protocol", choices=("tcp",), default="tcp")
    parser.add_argument("--mooncake-master-port", type=int, default=25151)
    parser.add_argument("--mooncake-metadata-port", type=int, default=25180)
    parser.add_argument("--mooncake-metrics-port", type=int, default=25103)
    parser.add_argument("--mooncake-client-port", type=int, default=25152)
    parser.add_argument("--mooncake-client-http-port", type=int, default=25190)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--ep-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument(
        "--mamba-buffer-strategy",
        choices=("no_buffer", "extra_buffer"),
        default=None,
        help="override the role-dependent Mamba buffer strategy",
    )
    parser.add_argument(
        "--auto-kernel-backends",
        action="store_true",
        help="let SGLang select attention/MoE backends instead of forcing Triton",
    )
    parser.add_argument("--groups-per-node", type=int, default=4)
    parser.add_argument(
        "--prefill-groups-per-node", type=int,
        help="Prefill TP groups on its node; defaults to --groups-per-node",
    )
    parser.add_argument(
        "--decode-groups-per-node", type=int,
        help="Decode TP groups on its node; defaults to --groups-per-node",
    )
    parser.add_argument(
        "--mixed-one-prefill-per-node", action="store_true",
        help=(
            "place one Prefill group and --decode-groups-per-node Decode "
            "groups on each of exactly two physical nodes"
        ),
    )
    parser.add_argument(
        "--single-node-1p3d",
        action="store_true",
        help=(
            "on one H100 node place TP2 Prefill on GPUs 0-1 and three TP2 "
            "Decode groups on GPUs 2-3/4-5/6-7"
        ),
    )
    parser.add_argument(
        "--single-node-2p6d",
        action="store_true",
        help=(
            "on one H100 node place TP1 Prefill on GPUs 0/4 and Decode on "
            "GPUs 1/2/3/5/6/7"
        ),
    )
    parser.add_argument(
        "--single-node-2p2d",
        action="store_true",
        help=(
            "on one H100 node place TP2 Prefill on GPUs 0-1/4-5 and "
            "TP2 Decode on GPUs 2-3/6-7"
        ),
    )
    parser.add_argument(
        "--cpus-per-worker", type=int, default=224,
        help="expose the full node CPU set so SGLang can bind each TP rank to its GPU NUMA node",
    )
    parser.add_argument("--prefill-port", type=int, default=24200)
    parser.add_argument("--decode-port", type=int, default=24300)
    parser.add_argument("--bootstrap-port", type=int, default=24500)
    parser.add_argument("--router-port", type=int, default=24400)
    parser.add_argument("--router-metrics-port", type=int, default=24401)
    parser.add_argument(
        "--router-policy",
        choices=("round_robin", "power_of_two", "random"),
        default="round_robin",
        help=(
            "round_robin is the corrected v0.5.21 baseline default: that "
            "release's power_of_two monitor expects legacy "
            "aggregate.total_tokens while workers publish loads[], causing "
            "all load samples to parse as -1 and biasing one Prefill replica"
        ),
    )
    parser.add_argument("--ib-interface", default="ibp41s0f0")
    parser.add_argument("--ib-device", default="mlx5_1")
    parser.add_argument(
        "--launch-transport", choices=("srun", "ssh"), default="srun",
        help=(
            "component control transport; ssh is for already allocated exclusive "
            "nodes whose child srun steps do not survive the controller session"
        ),
    )
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
    if bool(getattr(args, "storage_job_id", "")) != bool(
        getattr(args, "storage_node", "")
    ):
        raise SystemExit(
            "--storage-job-id and --storage-node must be supplied together"
        )
    if getattr(args, "storage_job_id", "") and not args.native_mooncake:
        raise SystemExit(
            "--storage-job-id/--storage-node require --native-mooncake"
        )
    args.prefill_groups_per_node = (
        getattr(args, "prefill_groups_per_node", None) or args.groups_per_node
    )
    args.decode_groups_per_node = (
        getattr(args, "decode_groups_per_node", None) or args.groups_per_node
    )
    single_topologies = sum(
        int(value)
        for value in (
            args.single_node_1p3d,
            args.single_node_2p6d,
            args.single_node_2p2d,
        )
    )
    if single_topologies > 1:
        parser.error("select only one single-node topology")
    if args.single_node_1p3d or args.single_node_2p6d or args.single_node_2p2d:
        if args.mixed_one_prefill_per_node or args.extra_job_id:
            raise SystemExit(
                "single-node topology cannot be combined with mixed/extra-node placement"
            )
        required_tp = 1 if args.single_node_2p6d else 2
        if args.tp_size != required_tp:
            raise SystemExit(
                f"selected single-node topology requires --tp-size {required_tp}"
            )
        args.prefill_groups_per_node = 1 if args.single_node_1p3d else 2
        args.decode_groups_per_node = (
            3 if args.single_node_1p3d
            else 6 if args.single_node_2p6d
            else 2
        )
    if args.mixed_one_prefill_per_node:
        if args.extra_job_id:
            raise SystemExit("mixed placement is restricted to one two-node job")
        if args.prefill_groups_per_node != 1:
            raise SystemExit("mixed placement requires one Prefill group per node")
        if args.tp_size * (1 + args.decode_groups_per_node) > 8:
            raise SystemExit(
                "mixed Prefill and Decode groups exceed eight GPUs per node"
            )
    for role, group_count in (
        ("prefill", args.prefill_groups_per_node),
        ("decode", args.decode_groups_per_node),
    ):
        if group_count < 1 or args.tp_size * group_count > 8:
            raise SystemExit(
                f"{role}: tp-size * groups-per-node must use between one and eight GPUs"
            )
    if args.cpus_per_worker < 8:
        raise SystemExit("reserve at least eight CPUs per model worker")
    args.run_dir = args.run_dir.resolve()
    manifest_path = args.run_dir / "native-pd-manifest.json"
    if args.action == "prepare":
        if not args.run_id:
            raise SystemExit("prepare requires --run-id")
        fields, nodes = site.allocation(
            args.job_id,
            min_nodes=(
                1
                if (
                    args.single_node_1p3d
                    or args.single_node_2p6d
                    or args.single_node_2p2d
                )
                else 2
            ),
        )
        selected = (
            nodes[:1]
            if (
                args.single_node_1p3d
                or args.single_node_2p6d
                or args.single_node_2p2d
            )
            else nodes[:2]
        )
        job_by_node = {node: args.job_id for node in selected}
        if args.extra_job_id:
            _, extra_nodes = site.allocation(args.extra_job_id, min_nodes=1)
            if extra_nodes[0] in selected:
                raise RuntimeError("Decode must use a distinct third node")
            if args.tp_size != 8 or args.prefill_groups_per_node != 1:
                raise RuntimeError("three-node native PD requires one TP8 Prefill per source node")
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
        if args.native_mooncake:
            if args.storage_job_id:
                _, storage_nodes = site.allocation(
                    args.storage_job_id, min_nodes=1
                )
                if args.storage_node not in storage_nodes:
                    raise RuntimeError(
                        f"storage node {args.storage_node} is not in Slurm job "
                        f"{args.storage_job_id}"
                    )
                storage_inventory = site.node_inventory(
                    args.storage_job_id, args.storage_node,
                    args.ib_interface, args.ib_device,
                )
                args.storage_job_id = str(args.storage_job_id)
                args.storage_ip = storage_inventory["ip"]
            elif args.workload_job_id:
                storage_inventory = site.node_inventory(
                    args.workload_job_id, args.workload_node,
                    args.ib_interface, args.ib_device,
                )
                args.storage_node = args.workload_node
                args.storage_job_id = args.workload_job_id
                args.storage_ip = storage_inventory["ip"]
            else:
                args.storage_node = selected[0]
                args.storage_job_id = args.job_id
                args.storage_ip = inventories[0]["ip"]
        if len(selected) > 1:
            for node, inventory in zip(selected[:-1], inventories[:-1]):
                site.transport_probe(job_by_node[node], node, inventories[-1]["ip"], args.python, args.ib_device)
            site.transport_probe(job_by_node[selected[-1]], selected[-1], inventories[0]["ip"], args.python, args.ib_device)
        required_paths = [
            args.python, args.workload_python, args.model, args.data_root,
            args.swe_sif_dir, args.workload_config,
        ]
        if not args.use_installed_sglang:
            required_paths.append(args.sglang_root / "python/sglang")
        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(path)
        if (
            args.native_mooncake
            and args.mooncake_compat is not None
            and not args.mooncake_compat.is_dir()
        ):
            raise FileNotFoundError(args.mooncake_compat)
        if (
            args.native_mooncake
            and args.mooncake_master_binary is not None
            and not args.mooncake_master_binary.is_file()
        ):
            raise FileNotFoundError(args.mooncake_master_binary)
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        args.job_id = str(args.job_id)
        manifest = build_manifest(args, selected, inventories)
        manifest["job_end_time"] = fields.get("EndTime")
        manifest["baseline_environment"] = verify_baseline_environment(args)
        write_json(manifest_path, manifest)
        print(manifest_path)
        return 0
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_manifest(manifest_path)
    args.job_id = manifest["job_id"]
    # A prepared manifest fixes the control transport so later status/stop
    # commands cannot accidentally address the component through another path.
    args.launch_transport = manifest.get("launch_transport", args.launch_transport)
    workers = tuple(
        name for name in manifest["components"]
        if name.startswith(("prefill-", "decode-"))
    )
    services = tuple(
        name for name in ("mooncake-master", "mooncake-client")
        if name in manifest["components"]
    )
    components = (*services, *workers, "router", "workload")
    if args.action == "start":
        started: list[str] = []
        try:
            for component in services:
                launch_component(args, manifest_path, component)
                started.append(component)
                wait_http(manifest["components"][component]["health"], 600)
            # SGLang chooses an ephemeral torch.distributed rendezvous port
            # inside each single-node TP launcher.  Starting several TP
            # launchers on the same host at once leaves a check-then-bind
            # race: two launchers can select the same apparently-free port,
            # and one later fails with EADDRINUSE.  Wait for each worker to
            # finish binding before starting the next one.  This changes only
            # startup ordering; the steady-state topology is unchanged.
            for component in workers:
                launch_component(args, manifest_path, component)
                started.append(component)
                wait_http(manifest["components"][component]["health"], 1800)
            if manifest.get("method") == "native_pd_hicache_mooncake":
                drop_model_weight_page_cache(manifest)
            launch_component(args, manifest_path, "router")
            started.append("router")
            wait_http(manifest["components"]["router"]["health"], 300)
            launch_component(args, manifest_path, "workload")
            started.append("workload")
        except BaseException:
            for component in reversed(started):
                try:
                    control_component(args, manifest_path, component, "stop")
                except BaseException:
                    pass
            raise
        return 0
    if args.action == "status":
        result = 0
        for component in components:
            result |= control_component(args, manifest_path, component, "status")
        return result
    result = 0
    for component in reversed(components):
        try:
            result |= control_component(args, manifest_path, component, "stop")
        except (OSError, subprocess.SubprocessError):
            result |= 1
    return result


if __name__ == "__main__":
    raise SystemExit(main())
