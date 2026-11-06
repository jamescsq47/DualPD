"""Adapt the upstream v3 launcher to this cluster's existing Slurm job.

The upstream data plane and control semantics remain untouched.  This file owns
only site discovery, paths, compiler selection, an immutable v3 config, and
explicit placement.  The default runtime transport is ``srun``; an explicit
``ssh`` transport is available for already allocated exclusive nodes on sites
that cancel child Slurm steps when the controller session disappears.  The
``probe``, ``prepare`` and ``plan`` actions never start a GPU process.
"""

import argparse
import getpass
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
DEFAULT_PYTHON = "/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python"
DEFAULT_ENGINE = ROOT.parent / "sglang_pd"
DEFAULT_MODEL = ROOT.parent / "models/Qwen3.8-27B"
DEFAULT_DATA = ROOT.parent / "datasets"
DEFAULT_APPTAINER = "/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer"
DEFAULT_CC = "/usr/bin/gcc"
DEFAULT_CXX = "/usr/bin/g++"
DEFAULT_NVCC_CCBIN = DEFAULT_CXX
DEFAULT_SYSTEM_BIN = "/usr/bin"
DEFAULT_D2P_HOST_GIB_PER_RANK = 64
DEFAULT_P2D_HOST_GIB_PER_RANK = 32
# Two physical lanes already saturate the per-GPU Host DMA path on the H100
# nodes (about 50 GiB/s aggregate in the production indexed-copy benchmark).
# Four lanes add little bandwidth while doubling the number of simultaneous
# HBM/PCIe contenders seen by Prefill and Decode forwards.  Keep control-path
# queueing asynchronous, but bound physical work to the useful concurrency.
DEFAULT_DIRECT_LANES = 2
DEFAULT_HOST_LANES = 2
DEFAULT_HOST_LOAD_WORKERS = 1
# H100 exposes CUDA stream priorities from 0 (lowest) through -5 (highest).
# Agentic Host/Direct progress uses normal-priority background streams, so put
# model Forward at the actual hardware maximum instead of merely one level
# above background work.
DEFAULT_FORWARD_STREAM_PRIORITY = -5
# Every rank owns several independent NIXL agents (the two Direct endpoints
# plus source/target Host transports).  Eight progress threads *per agent*
# oversubscribes a TP8 node without increasing the H100/IB data-path
# bandwidth.  Two leaves independent progress for each agent while keeping
# the CPU control plane small enough not to contend with the scheduler.
DEFAULT_NIXL_PROGRESS_THREADS = 2
DEFAULT_SIF_DIR = (
    Path("/orcd/pool/004") / getpass.getuser()
    / "dualpd/containers/swebench-verified-sif"
)
DEFAULT_DOCKER_WORKLOAD = (
    ROOT / "examples/pd/configs/experiments/"
    "swe_bench_verified_openenv_structured_tool_8k_t64_500.yaml"
)
DEFAULT_APPTAINER_WORKLOAD = (
    ROOT / "examples/pd/configs/experiments/"
    "swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer.yaml"
)
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
DEFAULT_LOCAL_ROOT = Path("/var/tmp") / getpass.getuser() / "dualpd-v3"


def output(argv, *, env=None):
    return subprocess.check_output(argv, text=True, env=env, timeout=60).strip()


def allocation(job_id, *, min_nodes=2):
    raw = output(["scontrol", "show", "job", "-o", str(job_id)])
    fields = dict(item.split("=", 1) for item in raw.split() if "=" in item)
    if fields.get("JobState") != "RUNNING":
        raise RuntimeError(f"Slurm job {job_id} is not RUNNING")
    owner = fields.get("UserId", "").split("(", 1)[0]
    if owner != getpass.getuser():
        raise RuntimeError(f"Slurm job {job_id} is owned by {owner}, not this user")
    if (fields.get("OverSubscribe") != "NO"
            or "gres/gpu:h100:8" not in fields.get("TresPerNode", "")):
        raise RuntimeError("Slurm job must own whole nodes with eight H100 GPUs per node")
    nodes = output(["scontrol", "show", "hostnames", fields["NodeList"]]).splitlines()
    if len(nodes) < min_nodes:
        raise RuntimeError(f"Slurm job requires at least {min_nodes} allocated nodes")
    return fields, nodes


def on_node(job_id, node, argv, *, cpus=1, env=None):
    command = ["srun", "--jobid", str(job_id), "--overlap", "--nodes=1",
               "--ntasks=1", "--nodelist", node, "--cpus-per-task", str(cpus),
               "--export=ALL", *argv]
    try:
        return output(command, env=env)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # A HiCache run can temporarily put the node above Linux's heuristic
        # commit limit.  slurmd then may reject creation of a harmless probe
        # step even though the existing exclusive allocation and sshd remain
        # healthy.  Runtime launch already supports SSH on this site; use the
        # same allocated-node path for read-only inventory/transport probes.
        return output(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
                node, shlex.join([str(value) for value in argv]),
            ],
            env=env,
        )


def node_inventory(job_id, node, interface, ib_device):
    if not SAFE_NAME.fullmatch(node) or not SAFE_NAME.fullmatch(interface):
        raise ValueError("unsafe Slurm node or interface name")
    if not SAFE_NAME.fullmatch(ib_device):
        raise ValueError("unsafe IB device name")
    script = (
        "set -euo pipefail; "
        "printf 'host=%s\\n' \"$(hostname -s)\"; "
        "printf 'gpus=%s\\n' \"$(nvidia-smi -L | wc -l)\"; "
        "printf 'gpu_name=%s\\n' \"$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd, -)\"; "
        "printf 'gpu_min_mib=%s\\n' \"$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)\"; "
        "printf 'mem_kib=%s\\n' \"$(awk '/MemTotal/{print $2}' /proc/meminfo)\"; "
        "printf 'memlock=%s\\n' \"$(ulimit -l)\"; "
        "printf 'docker=%s\\n' \"$(command -v docker || true)\"; "
        "printf 'podman=%s\\n' \"$(command -v podman || true)\"; "
        "printf 'apptainer=%s\\n' \"$(command -v apptainer || true)\"; "
        f"printf 'apptainer_shared=%s\\n' \"$([ -x {shlex.quote(DEFAULT_APPTAINER)} ] && "
        f"echo {shlex.quote(DEFAULT_APPTAINER)} || true)\"; "
        "printf 'slurm_oci=%s\\n' \"$([ -f /etc/slurm/oci.conf ] && echo enabled || echo disabled)\"; "
        f"printf 'rdma_state=%s\\n' \"$(cat /sys/class/infiniband/{ib_device}/ports/1/state)\"; "
        f"printf 'ip=%s\\n' \"$(ip -o -4 addr show dev {interface} | "
        "awk '{split($4,a,\"/\"); print a[1]; exit}')\""
    )
    values = dict(line.split("=", 1) for line in on_node(
        job_id, node, ["bash", "-lc", script]
    ).splitlines())
    if values.get("host") != node:
        raise RuntimeError(f"Slurm placed the probe on {values.get('host')}, expected {node}")
    if int(values.get("gpus", 0)) != 8:
        raise RuntimeError(f"{node} does not expose exactly eight GPUs")
    if "H100" not in values.get("gpu_name", "") or int(values.get("gpu_min_mib", 0)) < 80000:
        raise RuntimeError(f"{node} is not an eight-way H100 80GB node")
    if not values.get("ip"):
        raise RuntimeError(f"{node} has no IPv4 address on {interface}")
    if values.get("memlock") != "unlimited":
        raise RuntimeError(f"{node} memlock is not unlimited")
    if not values.get("rdma_state", "").endswith("ACTIVE"):
        raise RuntimeError(f"{node} {ib_device}:1 is not ACTIVE")
    return values


def transport_probe(job_id, source, peer_ip, python, ib_device):
    if not SAFE_NAME.fullmatch(source) or not SAFE_NAME.fullmatch(ib_device):
        raise ValueError("unsafe transport probe input")
    peer = str(ipaddress.ip_address(peer_ip))
    interpreter = shlex.quote(str(Path(python).resolve()))
    script = (
        "set -euo pipefail; "
        f"ip route get {peer} >/dev/null; "
        f"UCX_NET_DEVICES={ib_device}:1 {interpreter} -c \"import nixl; "
        "nixl.nixl_agent('dualpd_slurm_probe')\""
    )
    on_node(job_id, source, ["bash", "-lc", script])
    return {"peer_route_present": True, "nixl_ucx_loads": True,
            "note": "capability probe only; not an RDMA bandwidth or GDR measurement"}


def restore_prepared_placement(args, config_path, allocated_nodes):
    """Restore immutable physical placement before any runtime Slurm step."""
    cfg = json.loads(config_path.read_text())
    placement = cfg.get("slurm_nodes")
    if not isinstance(placement, dict):
        raise RuntimeError("prepared config lacks immutable slurm_nodes placement")
    for role, supplied in (
        ("prefill", args.prefill_node), ("decode", args.decode_node)
    ):
        stored = placement.get(role)
        if stored not in allocated_nodes:
            raise RuntimeError(f"prepared {role} node {stored!r} is outside this allocation")
        if supplied and supplied != stored:
            raise RuntimeError(
                f"{role} node override {supplied!r} conflicts with prepared node {stored!r}"
            )
    args.prefill_node = placement["prefill"]
    args.decode_node = placement["decode"]
    extra = placement.get("extra_prefill")
    if extra:
        if args.extra_prefill_node and args.extra_prefill_node != extra:
            raise RuntimeError("extra Prefill override conflicts with prepared node")
        args.extra_prefill_node = extra
        stored_job = cfg.get("slurm_node_jobs", {}).get(placement["decode"])
        if args.extra_job_id and args.extra_job_id != stored_job:
            raise RuntimeError("extra Slurm job override conflicts with prepared job")
        args.extra_job_id = stored_job
    return cfg


def build_config(args, prefill, decode, run_dir, workload_config, swe_work_root,
                 extra_prefill=None):
    run_id = args.run_id or run_dir.name
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("run ID must contain only letters, numbers, '.', '_' or '-'")
    token = args.control_token or secrets.token_urlsafe(32)
    workload_output = run_dir / "workload"
    inference = ROOT / "examples/pd/inference.py"
    mixed = bool(getattr(args, "mixed_one_prefill_per_node", False))
    single_node_1p3d = bool(getattr(args, "single_node_1p3d", False))
    single_node_2p6d = bool(getattr(args, "single_node_2p6d", False))
    single_node_2p2d = bool(getattr(args, "single_node_2p2d", False))
    if single_node_1p3d:
        prefill_groups, decode_groups = 1, 3
        prefill_ports = [23900]
        decode_ports = [24000, 24010, 24020]
        prefill_endpoints = [f"http://{prefill['ip']}:23900"]
        decode_endpoints = [
            f"http://{decode['ip']}:{port}" for port in decode_ports
        ]
        primary_prefill_host = prefill["ip"]
        primary_decode_host = decode["ip"]
    elif single_node_2p2d:
        prefill_groups, decode_groups = 2, 2
        prefill_ports = [23900, 23910]
        decode_ports = [24000, 24010]
        prefill_endpoints = [
            f"http://{prefill['ip']}:{port}" for port in prefill_ports
        ]
        decode_endpoints = [
            f"http://{decode['ip']}:{port}" for port in decode_ports
        ]
        primary_prefill_host = prefill["ip"]
        primary_decode_host = decode["ip"]
    elif single_node_2p6d:
        prefill_groups, decode_groups = 2, 6
        prefill_ports = [23900, 23910]
        decode_ports = [24000 + 10 * index for index in range(6)]
        prefill_endpoints = [
            f"http://{prefill['ip']}:{port}" for port in prefill_ports
        ]
        decode_endpoints = [
            f"http://{decode['ip']}:{port}" for port in decode_ports
        ]
        primary_prefill_host = prefill["ip"]
        primary_decode_host = decode["ip"]
    elif mixed:
        if extra_prefill is not None:
            raise ValueError("mixed two-node placement cannot use an extra Prefill node")
        prefill_groups, decode_groups = 2, 6
        prefill_endpoints = [
            f"http://{prefill['ip']}:23900",
            f"http://{decode['ip']}:23900",
        ]
        decode_endpoints = [
            f"http://{inventory['ip']}:{port}"
            for inventory in (prefill, decode)
            for port in (23910, 23920, 23930)
        ]
        prefill_ports = [23900, 23900]
        decode_ports = [23910, 23920, 23930, 23910, 23920, 23930]
        primary_prefill_host = prefill["ip"]
        primary_decode_host = prefill["ip"]
    else:
        prefill_groups = (
            getattr(args, "prefill_groups_per_node", None) or args.groups_per_node
        )
        decode_groups = (
            getattr(args, "decode_groups_per_node", None) or args.groups_per_node
        )
        prefill_ports = [23900 + 10 * index for index in range(prefill_groups)]
        decode_ports = [23900 + 10 * index for index in range(decode_groups)]
        prefill_endpoints = [
            f"http://{prefill['ip']}:{port}" for port in prefill_ports
        ]
        decode_endpoints = [
            f"http://{decode['ip']}:{port}" for port in decode_ports
        ]
        primary_prefill_host = prefill["ip"]
        primary_decode_host = decode["ip"]
    command = [
        str(Path(args.python).resolve()), str(inference),
        "--model", str(args.model.resolve()),
        "--workload-config", str(workload_config.resolve()),
        "--router-host", prefill["ip"], "--router-port", "23902",
        "--prefill-host", primary_prefill_host,
        "--prefill-port", str(prefill_ports[0]),
        "--prefill-ports", ",".join(map(str, prefill_ports)),
        "--decode-host", primary_decode_host,
        "--decode-port", str(decode_ports[0]),
        "--decode-ports", ",".join(map(str, decode_ports)),
        "--requests", str(args.requests), "--warmup-requests", "0",
        "--request-rate", "100", "--arrival-distribution", "fixed",
        "--dispatch-policy", "random", "--preserve-source-order",
        "--max-inflight", str(args.concurrency), "--metrics-interval", "2",
        "--seed", "2026", "--temperature", "0.6", "--top-p", "0.95",
        "--top-k", "20", "--max-context-length", "131072",
        "--max-response-length", "81920", "--output-dir", str(workload_output),
    ]
    if mixed or single_node_1p3d or single_node_2p6d or single_node_2p2d:
        command += [
            "--prefill-endpoints", ",".join(prefill_endpoints),
            "--decode-endpoints", ",".join(decode_endpoints),
        ]
    elif extra_prefill is not None:
        command += [
            "--prefill-endpoints",
            ",".join(
                [f"http://{prefill['ip']}:{port}" for port in prefill_ports]
                + [f"http://{extra_prefill['ip']}:23900"]
            ),
        ]
    model_family = getattr(args, "model_family", "qwen35_moe")
    reasoning_parser = getattr(args, "reasoning_parser", None) or (
        "minimax-append-think" if model_family == "minimax_m2" else "glm45"
    )
    tool_call_parser = getattr(args, "tool_call_parser", None) or (
        "minimax-m2" if model_family == "minimax_m2" else "qwen3_coder"
    )
    cfg = {
        "run_id": run_id,
        "mixed_one_prefill_per_node": mixed,
        "single_node_1p3d": single_node_1p3d,
        "single_node_2p6d": single_node_2p6d,
        "single_node_2p2d": single_node_2p2d,
        "launch_transport": getattr(args, "launch_transport", "srun"),
        "slurm_nodes": {
            "prefill": args.prefill_node,
            "decode": args.decode_node,
        },
        "local_root": str(args.local_root),
        "sglang_root": str(args.sglang_root.resolve()),
        "slime_root": str(ROOT.resolve()),
        "python": str(Path(args.python).resolve()),
        "model_path": str(args.model.resolve()),
        "model_family": model_family,
        "ep_size": 8 if model_family == "minimax_m2" else 1,
        "reasoning_parser": reasoning_parser,
        "tool_call_parser": tool_call_parser,
        "deterministic_inference": bool(
            getattr(args, "deterministic_inference", False)
        ),
        "cuda_home": "/usr/local/cuda",
        "system_bin": DEFAULT_SYSTEM_BIN,
        "cc": DEFAULT_CC,
        "cxx": DEFAULT_CXX,
        "nvcc_ccbin": DEFAULT_NVCC_CCBIN,
        "local_triton_cache": True,
        "debug_memory_pool": False,
        "tp_debug_batch": bool(getattr(args, "tp_debug_batch", False)),
        "tp_size": args.tp_size,
        "page_size": 64,
        "context_length": 131072,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        # Preserve 8192-token first-turn Prefill while keeping restored
        # incremental batches within FlashInfer's fused-allreduce envelope.
        "flashinfer_allreduce_max_tokens": getattr(
            args, "flashinfer_allreduce_max_tokens", 8192
        ),
        "decode_growth_tokens": getattr(args, "decode_growth_tokens", 8192),
        # These are physical I/O limits only.  Direct and Slow retain separate
        # queues and all ownership/fence semantics; reducing the lane count
        # prevents redundant HBM/PCIe contention with model Forward.
        "direct_lanes": getattr(args, "direct_lanes", DEFAULT_DIRECT_LANES),
        "d2p_direct_lanes": (
            getattr(args, "d2p_direct_lanes", None)
            or getattr(args, "direct_lanes", DEFAULT_DIRECT_LANES)
        ),
        "p2d_direct_lanes": (
            getattr(args, "p2d_direct_lanes", None)
            or getattr(args, "direct_lanes", DEFAULT_DIRECT_LANES)
        ),
        "host_lanes": getattr(args, "host_lanes", DEFAULT_HOST_LANES),
        "host_load_workers": getattr(
            args, "host_load_workers", DEFAULT_HOST_LOAD_WORKERS
        ),
        "nixl_progress_threads": getattr(
            args, "nixl_progress_threads", DEFAULT_NIXL_PROGRESS_THREADS
        ),
        "forward_stream_priority": getattr(
            args, "forward_stream_priority", DEFAULT_FORWARD_STREAM_PRIORITY
        ),
        "serialize_p2d_direct_with_forward": not getattr(
            args, "overlap_p2d_direct_with_forward", False
        ),
        "d2p_host_gib_per_rank": args.d2p_host_gib_per_rank,
        "p2d_host_gib_per_rank": args.p2d_host_gib_per_rank,
        "fast_tool_seconds": getattr(args, "fast_tool_seconds", 1.0),
        "direct_admission_seconds": 1,
        "mamba_full_memory_ratio": 0.5,
        "prefill_mamba_full_memory_ratio": 0.5,
        "decode_mamba_full_memory_ratio": 0.5,
        "seed": 2026,
        "group_control": {
            "enabled": True, "node_id": "p0",
            "listen_host": "0.0.0.0", "advertise_host": prefill["ip"],
            "port": 23904, "token": token,
        },
        "router": {
            "node_id": "p0", "engine_id": "router-0",
            "port": 23902, "metrics_port": 23903,
        },
        "workload_command": command,
        "workload_environment": {
            "PD_DATA_ROOT": str(args.data_root.resolve()),
            "PD_SWE_RUN_ID": run_id,
            "PD_SWE_PROGRESS_FILE": str(run_dir / "episode_progress.jsonl"),
            "PD_MODEL_HTTP_TRANSPORT": "aiohttp",
            "SLIME_HTTP_READ_TIMEOUT_SECONDS": "86400",
            "MIN_P": "0",
        },
        "nodes": [],
    }
    if extra_prefill is not None:
        cfg["slurm_nodes"]["extra_prefill"] = args.extra_prefill_node
        cfg["slurm_node_jobs"] = {
            args.prefill_node: str(args.job_id),
            args.decode_node: str(args.extra_job_id),
            args.extra_prefill_node: str(args.job_id),
        }
    if args.swe_sandbox_backend == "apptainer":
        cfg["workload_environment"].update(
            PD_SWE_APPTAINER_IMAGE_DIR=str(args.swe_sif_dir.resolve()),
            PD_SWE_APPTAINER_WORK_ROOT=str(swe_work_root / run_id),
        )
    if single_node_1p3d:
        d2p_by_node = {"d0": 86, "d1": 85, "d2": 85}
        layouts = [
            ("p0", "prefill-0", "prefill", prefill, [0, 1], 23900, 61900),
            ("d0", "decode-0", "decode", decode, [2, 3], 24000, 62000),
            ("d1", "decode-1", "decode", decode, [4, 5], 24010, 62010),
            ("d2", "decode-2", "decode", decode, [6, 7], 24020, 62020),
        ]
        for node_id, engine_id, role, inventory, gpus, port, reverse_port in layouts:
            entry = {
                "node_id": node_id, "engine_id": engine_id, "role": role,
                "host_ip": inventory["ip"], "gpus": gpus,
                "numa_nodes": [0 if gpu < 4 else 1 for gpu in gpus],
                "mem_fraction_static": 0.8,
                "port": port, "bootstrap_port": port + 1,
                "reverse_bootstrap_port": reverse_port,
                "ib_device": args.ib_device,
                "ucx_net_devices": f"{args.ib_device}:1",
            }
            if role == "decode":
                entry["d2p_host_gib_per_rank"] = d2p_by_node[node_id]
            else:
                entry["p2d_host_gib_per_rank"] = 128
            cfg["nodes"].append(entry)
        cfg["d2p_host_gib_total"] = 512
        cfg["p2d_host_gib_total"] = 256
    elif single_node_2p2d:
        layouts = [
            ("p0", "prefill-0", "prefill", prefill, [0, 1], 23900, 61900),
            ("d0", "decode-0", "decode", decode, [2, 3], 24000, 62000),
            ("p1", "prefill-1", "prefill", prefill, [4, 5], 23910, 61910),
            ("d1", "decode-1", "decode", decode, [6, 7], 24010, 62010),
        ]
        for node_id, engine_id, role, inventory, gpus, port, reverse_port in layouts:
            cfg["nodes"].append({
                "node_id": node_id, "engine_id": engine_id, "role": role,
                "host_ip": inventory["ip"], "gpus": gpus,
                "numa_nodes": [0 if gpu < 4 else 1 for gpu in gpus],
                "mem_fraction_static": 0.8,
                "port": port, "bootstrap_port": port + 1,
                "reverse_bootstrap_port": reverse_port,
                "ib_device": args.ib_device,
                "ucx_net_devices": f"{args.ib_device}:1",
            })
        cfg["d2p_host_gib_total"] = args.d2p_host_gib_per_rank * 4
        cfg["p2d_host_gib_total"] = args.p2d_host_gib_per_rank * 4
    elif single_node_2p6d:
        # Source-local arenas exist only on the source role.  Use integral-GiB
        # per-worker capacities whose directional sums are exactly the user's
        # requested 512 GiB D->P and 256 GiB P->D totals.
        d2p_by_node = {"d0": 86, "d1": 86, "d2": 85, "d3": 85, "d4": 85, "d5": 85}
        layouts = [
            ("p0", "prefill-0", "prefill", prefill, [0], 23900, 61900),
            ("p1", "prefill-1", "prefill", prefill, [4], 23910, 61910),
        ]
        layouts.extend(
            (
                f"d{index}", f"decode-{index}", "decode", decode, [gpu],
                24000 + 10 * index, 62000 + 10 * index,
            )
            for index, gpu in enumerate((1, 2, 3, 5, 6, 7))
        )
        for node_id, engine_id, role, inventory, gpus, port, reverse_port in layouts:
            entry = {
                "node_id": node_id, "engine_id": engine_id, "role": role,
                "host_ip": inventory["ip"], "gpus": list(gpus),
                "numa_nodes": [0 if gpu < 4 else 1 for gpu in gpus],
                "mem_fraction_static": 0.8,
                "port": port, "bootstrap_port": port + 1,
                "reverse_bootstrap_port": reverse_port,
                "ib_device": args.ib_device,
                "ucx_net_devices": f"{args.ib_device}:1",
            }
            if role == "decode":
                entry["d2p_host_gib_per_rank"] = d2p_by_node[node_id]
            else:
                entry["p2d_host_gib_per_rank"] = 128
            cfg["nodes"].append(entry)
        cfg["d2p_host_gib_total"] = 512
        cfg["p2d_host_gib_total"] = 256
    elif mixed:
        layouts = []
        for physical_index, inventory in enumerate((prefill, decode)):
            layouts.append((
                f"p{physical_index}", f"prefill-{physical_index}",
                "prefill", inventory, [0, 1], 23900, 61900,
            ))
            for local_decode_index, gpus in enumerate(((2, 3), (4, 5), (6, 7))):
                global_decode_index = physical_index * 3 + local_decode_index
                port = 23910 + 10 * local_decode_index
                layouts.append((
                    f"d{global_decode_index}",
                    f"decode-{global_decode_index}", "decode", inventory,
                    list(gpus), port, 61910 + 10 * local_decode_index,
                ))
        for node_id, engine_id, role, inventory, gpus, port, reverse_port in layouts:
            cfg["nodes"].append({
                "node_id": node_id, "engine_id": engine_id, "role": role,
                "host_ip": inventory["ip"], "gpus": list(gpus),
                "numa_nodes": [0 if gpu < 4 else 1 for gpu in gpus],
                "mem_fraction_static": 0.8,
                "port": port, "bootstrap_port": port + 1,
                "reverse_bootstrap_port": reverse_port,
                "ib_device": args.ib_device,
                "ucx_net_devices": f"{args.ib_device}:1",
            })
    else:
        for prefix, role, inventory, group_count in (
            ("p", "prefill", prefill, prefill_groups),
            ("d", "decode", decode, decode_groups),
        ):
            for index in range(group_count):
                first_gpu = index * args.tp_size
                gpus = list(range(first_gpu, first_gpu + args.tp_size))
                port = 23900 + 10 * index
                cfg["nodes"].append(
                    {
                        "node_id": f"{prefix}{index}",
                        "engine_id": f"{role}-{index}",
                        "role": role,
                        "host_ip": inventory["ip"],
                        "gpus": gpus,
                        "numa_nodes": [0 if gpu < 4 else 1 for gpu in gpus],
                        "mem_fraction_static": 0.8,
                        "port": port,
                        "bootstrap_port": port + 1,
                        "reverse_bootstrap_port": 61900 + 10 * index,
                        "ib_device": args.ib_device,
                        "ucx_net_devices": f"{args.ib_device}:1",
                    }
                )
    if extra_prefill is not None:
        cfg["nodes"].insert(prefill_groups, {
            "node_id": f"p{prefill_groups}",
            "engine_id": f"prefill-{prefill_groups}",
            "role": "prefill",
            "host_ip": extra_prefill["ip"],
            "gpus": list(range(args.tp_size)),
            "numa_nodes": [0 if gpu < 4 else 1 for gpu in range(args.tp_size)],
            "mem_fraction_static": 0.8,
            "port": 23900,
            "bootstrap_port": 23901,
            "reverse_bootstrap_port": 61900,
            "ib_device": args.ib_device,
            "ucx_net_devices": f"{args.ib_device}:1",
        })
    return cfg


def invoke_multinode(
    args, config_path, action, *, node_id=None, slurm_node=None, component=None
):
    target = slurm_node or args.prefill_node
    argv = [str(Path(args.python).resolve()), str(TOOLS / "multinode.py"),
            action, "--config", str(config_path)]
    if node_id is not None and (action == "start-worker" or component == "worker"):
        argv.extend(("--node-id", node_id))
    if component is not None:
        argv.extend(("--component", component))
    node_jobs = json.loads(config_path.read_text()).get("slurm_node_jobs", {})
    command = [
        "srun", "--jobid", node_jobs.get(target, str(args.job_id)), "--overlap", "--nodes=1",
        "--ntasks=1", "--nodelist", target, "--cpus-per-task",
        str(
            args.cpus_per_node
            if action == "start-worker" or component == "worker"
            else max(8, min(64, args.cpus_per_node))
        ),
        "--cpu-bind=none", "--export=ALL", *argv,
    ]
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("SGLANG_", "PD_", "NCCL_", "TORCH_NCCL_"))
    }
    if getattr(args, "launch_transport", "srun") == "ssh":
        ssh_options = [
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
        ]
        if args.detach:
            launcher_dir = config_path.parent / "launcher"
            launcher_dir.mkdir(parents=True, exist_ok=True)
            label = component or action
            if node_id is not None:
                label += "-" + node_id
            log_path = launcher_dir / f"{label}-{target}.log"
            remote = " ".join((
                "umask 077;",
                "mkdir -p", shlex.quote(str(launcher_dir)) + ";",
                "nohup setsid -f", shlex.join(argv),
                "</dev/null", ">>" + shlex.quote(str(log_path)), "2>&1",
            ))
            completed = subprocess.run(
                ["ssh", *ssh_options, target, remote],
                check=False, capture_output=True, text=True,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"ssh launch failed on {target}: {completed.stderr.strip()}"
                )
            print(json.dumps({
                "detached": True, "transport": "ssh", "node": target,
                "action": action, "log": str(log_path),
            }, indent=2), flush=True)
            return 0
        return subprocess.call([
            "ssh", *ssh_options, target, shlex.join(argv),
        ])
    if args.detach:
        # The component supervisor is intentionally foreground, but the srun
        # client must outlive an interactive shell/API tool session.  Give the
        # client its own session and durable file descriptors so a caller
        # disconnect cannot send SIGHUP and tear down the Slurm step (and all
        # TP ranks) in the middle of a long experiment.
        launcher_dir = config_path.parent / "launcher"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        label = component or action
        if node_id is not None:
            label += "-" + node_id
        log_path = launcher_dir / f"{label}-{target}.log"
        with log_path.open("ab", buffering=0) as output_log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=output_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        print(json.dumps({
            "detached": True,
            "srun_pid": process.pid,
            "node": target,
            "action": action,
            "log": str(log_path),
        }, indent=2), flush=True)
        return 0
    return subprocess.call(command, env=environment)


def wait_model_endpoint(node, *, timeout=600.0):
    """Wait until the fixed relay coordinator's model endpoint is ready."""

    url = f"http://{node['host_ip']}:{int(node['port'])}/model_info"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + float(timeout)
    last_error = None
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=2.0) as response:
                if 200 <= int(response.status) < 300:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(1.0)
    raise RuntimeError(
        f"coordinator model endpoint did not become ready: {url}; "
        f"last_error={last_error!r}"
    )


def start_worker_set(args, config_path, node_ids, physical_node, cfg, *,
                     establish_coordinator=False):
    """Start logical workers, establishing the relay coordinator first."""

    remaining = list(node_ids)
    if establish_coordinator and remaining:
        coordinator_id = remaining.pop(0)
        result = invoke_multinode(
            args, config_path, "start-worker", node_id=coordinator_id,
            slurm_node=physical_node,
        )
        if result:
            return result
        coordinator = next(
            node for node in cfg["nodes"] if node["node_id"] == coordinator_id
        )
        wait_model_endpoint(coordinator)
    return max(
        (
            invoke_multinode(
                args, config_path, "start-worker", node_id=node_id,
                slurm_node=physical_node,
            )
            for node_id in remaining
        ),
        default=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--prefill-node")
    parser.add_argument("--decode-node")
    parser.add_argument("--extra-job-id", default="")
    parser.add_argument("--extra-prefill-node")
    parser.add_argument("--ib-interface", default="ibp41s0f0")
    parser.add_argument("--ib-device", default="mlx5_1")
    parser.add_argument("--cpus-per-node", type=int, default=224,
                        help="CPU cores exposed to each model task (default: the full H100 node)")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-family", choices=("qwen35_moe", "minimax_m2"),
        default="qwen35_moe",
    )
    parser.add_argument("--reasoning-parser")
    parser.add_argument("--tool-call-parser")
    parser.add_argument("--deterministic-inference", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--swe-sandbox-backend", choices=("apptainer", "docker"),
                        default="apptainer")
    parser.add_argument("--apptainer-binary", type=Path,
                        default=Path(DEFAULT_APPTAINER))
    parser.add_argument("--swe-sif-dir", type=Path, default=DEFAULT_SIF_DIR)
    parser.add_argument("--swe-work-root", type=Path,
                        help="node-local mutable verifier root; default /var/tmp/<user>/pd-swe-apptainer")
    parser.add_argument("--workload-config", type=Path,
                        help="explicit workload YAML; backend-specific default otherwise")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--control-token", default="")
    parser.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_ROOT)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument(
        "--tp-debug-batch",
        action="store_true",
        help="log per-rank Prefill batch identity/shape around forward",
    )
    parser.add_argument("--tp-size", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--groups-per-node", type=int, default=1)
    parser.add_argument(
        "--mixed-one-prefill-per-node",
        action="store_true",
        help=(
            "on each of the two physical nodes place one TP=2 Prefill group "
            "on GPUs 0-1 and three TP=2 Decode groups on GPUs 2-7"
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
        "--prefill-groups-per-node", type=int,
        help="Prefill TP groups on its node; defaults to --groups-per-node",
    )
    parser.add_argument(
        "--decode-groups-per-node", type=int,
        help="Decode TP groups on its node; defaults to --groups-per-node",
    )
    parser.add_argument(
        "--d2p-host-gib-per-rank",
        type=int,
        default=DEFAULT_D2P_HOST_GIB_PER_RANK,
        help=(
            "D-to-P source Host capacity per GPU rank in GiB "
            "(default: 64; eight ranks give 512 GiB per Decode node)"
        ),
    )
    parser.add_argument(
        "--p2d-host-gib-per-rank",
        type=int,
        default=DEFAULT_P2D_HOST_GIB_PER_RANK,
        help=(
            "P-to-D source Host capacity per GPU rank in GiB "
            "(default: 32; eight ranks give 256 GiB per Prefill node)"
        ),
    )
    parser.add_argument(
        "--direct-lanes", type=int, default=DEFAULT_DIRECT_LANES,
        help="physical Direct transfers per rank (H100 default: 2)",
    )
    parser.add_argument(
        "--d2p-direct-lanes",
        type=int,
        help="D-to-P Direct transfers per rank; defaults to --direct-lanes",
    )
    parser.add_argument(
        "--p2d-direct-lanes",
        type=int,
        help="P-to-D Direct transfers per rank; defaults to --direct-lanes",
    )
    parser.add_argument(
        "--host-lanes", type=int, default=DEFAULT_HOST_LANES,
        help="physical source Host/D2H transfers per rank (H100 default: 2)",
    )
    parser.add_argument(
        "--host-load-workers", type=int, default=DEFAULT_HOST_LOAD_WORKERS,
        help="physical target Host/H2D reads per rank (H100 default: 1)",
    )
    parser.add_argument(
        "--fast-tool-seconds",
        type=float,
        default=1.0,
        help=(
            "keep tool calls no longer than this threshold on the Direct "
            "D-to-P path instead of staging through Host (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--nixl-progress-threads", type=int,
        default=DEFAULT_NIXL_PROGRESS_THREADS,
        help="NIXL progress threads per transport agent (H100 default: 2)",
    )
    parser.add_argument(
        "--forward-stream-priority", type=int,
        default=DEFAULT_FORWARD_STREAM_PRIORITY,
        help=(
            "CUDA priority for model Forward; H100 default -5 is the "
            "hardware's highest priority and keeps TP collectives ahead of "
            "normal-priority Agentic transfer streams"
        ),
    )
    parser.add_argument(
        "--flashinfer-allreduce-max-tokens",
        type=int,
        default=8192,
        help=(
            "FlashInfer fused AR+RMSNorm workspace/batch limit "
            "(H100 default: 8192)"
        ),
    )
    parser.add_argument(
        "--decode-growth-tokens",
        type=int,
        default=8192,
        help=(
            "per-Decode shared Attention-KV growth reserve withheld from new "
            "P-to-D ingress (default: 8192 tokens)"
        ),
    )
    parser.add_argument(
        "--overlap-p2d-direct-with-forward",
        action="store_true",
        help=(
            "allow active P-to-D Direct DMA to overlap the next Prefill; "
            "the H100 default serializes only physically active DMA"
        ),
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="detach long-lived srun clients from the invoking shell",
    )
    parser.add_argument(
        "--launch-transport", choices=("srun", "ssh"), default="srun",
        help=(
            "runtime component transport; ssh is only for nodes already held "
            "exclusively by the supplied Slurm allocation"
        ),
    )
    parser.add_argument("action", choices=(
        "probe", "prepare", "plan", "start-control", "control-ready",
        "start-prefill", "start-decode", "wait-ready", "start-router",
        "run-workload", "status", "stop",
    ))
    parser.add_argument("--component", choices=("control", "worker", "router", "workload"))
    parser.add_argument("--worker-role", choices=("prefill", "decode"), default="decode")
    args = parser.parse_args()
    if args.detach and args.action not in {
        "start-control", "start-prefill", "start-decode",
        "start-router", "run-workload",
    }:
        parser.error("--detach is only valid for long-lived start actions")
    if args.cpus_per_node < 32:
        parser.error("reserve at least 32 CPU cores per model node")
    if args.requests < 1 or args.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    if args.model_family == "minimax_m2" and args.tp_size != 8:
        parser.error("MiniMax-M2 requires --tp-size 8 (and uses EP=8)")
    if args.d2p_host_gib_per_rank < 1 or args.p2d_host_gib_per_rank < 1:
        parser.error("Host capacity per rank must be positive")
    if args.decode_growth_tokens < 0:
        parser.error("Decode growth reserve must be non-negative")
    if args.fast_tool_seconds < 0:
        parser.error("--fast-tool-seconds must be non-negative")
    if min(
        args.direct_lanes,
        args.d2p_direct_lanes or args.direct_lanes,
        args.p2d_direct_lanes or args.direct_lanes,
        args.host_lanes,
        args.host_load_workers,
        args.nixl_progress_threads,
    ) < 1:
        parser.error("transfer lane and progress-thread counts must be positive")
    args.prefill_groups_per_node = (
        args.prefill_groups_per_node or args.groups_per_node
    )
    args.decode_groups_per_node = (
        args.decode_groups_per_node or args.groups_per_node
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
        if args.mixed_one_prefill_per_node or args.extra_job_id or args.extra_prefill_node:
            parser.error("single-node topology cannot be combined with mixed/extra-node placement")
        required_tp = 1 if args.single_node_2p6d else 2
        if args.tp_size != required_tp:
            parser.error(
                f"selected single-node topology requires --tp-size {required_tp}"
            )
        args.prefill_groups_per_node = 1 if args.single_node_1p3d else 2
        args.decode_groups_per_node = (
            3 if args.single_node_1p3d
            else 6 if args.single_node_2p6d
            else 2
        )
    if args.mixed_one_prefill_per_node:
        if args.tp_size != 2:
            parser.error("mixed one-P/three-D placement requires --tp-size 2")
        if args.extra_job_id or args.extra_prefill_node:
            parser.error("mixed placement is strictly a two-node topology")
        args.prefill_groups_per_node = 1
        args.decode_groups_per_node = 3
    for role, group_count in (
        ("prefill", args.prefill_groups_per_node),
        ("decode", args.decode_groups_per_node),
    ):
        if group_count < 1 or args.tp_size * group_count > 8:
            parser.error(
                f"{role}: tp-size * groups-per-node must use between one and eight GPUs"
            )
    workload_config = args.workload_config or (
        DEFAULT_APPTAINER_WORKLOAD
        if args.swe_sandbox_backend == "apptainer"
        else DEFAULT_DOCKER_WORKLOAD
    )
    swe_work_root = args.swe_work_root or Path(
        f"/var/tmp/{getpass.getuser()}/pd-swe-apptainer"
    )
    if args.swe_sandbox_backend == "apptainer":
        if not args.apptainer_binary.is_file():
            parser.error(f"Apptainer binary is unavailable: {args.apptainer_binary}")
        if not workload_config.is_file():
            parser.error(f"Apptainer workload is unavailable: {workload_config}")

    fields, nodes = allocation(
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
    run_dir = args.run_dir.resolve() if args.run_dir else None
    config_path = run_dir / "multinode-v3.json" if run_dir else None
    prepared_cfg = None
    if config_path and config_path.is_file() and args.action not in {"prepare"}:
        prepared = json.loads(config_path.read_text())
        args.launch_transport = prepared.get(
            "launch_transport", args.launch_transport
        )
        prepared_decode = prepared.get("slurm_nodes", {}).get("decode")
        prepared_job = prepared.get("slurm_node_jobs", {}).get(prepared_decode)
        if prepared_job:
            if args.extra_job_id and args.extra_job_id != prepared_job:
                raise RuntimeError("extra Slurm job override conflicts with prepared job")
            args.extra_job_id = prepared_job
            _, extra_nodes = allocation(prepared_job, min_nodes=1)
            allocated = nodes + extra_nodes
        else:
            allocated = nodes
        prepared_cfg = restore_prepared_placement(args, config_path, allocated)
    args.prefill_node = args.prefill_node or nodes[0]
    extra = None
    if args.extra_job_id:
        _, extra_nodes = allocation(args.extra_job_id, min_nodes=1)
        args.extra_prefill_node = args.extra_prefill_node or nodes[1]
        args.decode_node = args.decode_node or extra_nodes[0]
        if (args.extra_prefill_node not in nodes or args.extra_prefill_node == args.prefill_node
                or args.decode_node not in extra_nodes or args.decode_node in nodes):
            parser.error("two Prefill nodes must be in the primary job and Decode in the extra job")
        if args.tp_size != 8 or args.prefill_groups_per_node != 1:
            parser.error("extra Prefill currently requires one TP8 group per node")
        extra = node_inventory(args.job_id, args.extra_prefill_node,
                               args.ib_interface, args.ib_device)
    elif args.extra_prefill_node:
        parser.error("--extra-prefill-node requires --extra-job-id")
    elif (
        args.single_node_1p3d
        or args.single_node_2p6d
        or args.single_node_2p2d
    ):
        args.decode_node = args.decode_node or nodes[0]
        if args.prefill_node != args.decode_node or args.prefill_node not in nodes:
            parser.error(
                "single-node topology requires Prefill and Decode on the sole allocated node"
            )
    else:
        args.decode_node = args.decode_node or nodes[1]
        if args.prefill_node == args.decode_node or args.prefill_node not in nodes or args.decode_node not in nodes:
            parser.error("Prefill and Decode must be distinct nodes in this allocation")
    p = node_inventory(args.job_id, args.prefill_node, args.ib_interface, args.ib_device)
    d = node_inventory(args.extra_job_id or args.job_id, args.decode_node,
                       args.ib_interface, args.ib_device)
    p["peer_probe"] = transport_probe(
        args.job_id, args.prefill_node, d["ip"], args.python, args.ib_device
    )
    d["peer_probe"] = transport_probe(
        args.extra_job_id or args.job_id, args.decode_node, p["ip"],
        args.python, args.ib_device
    )
    if extra is not None:
        extra["peer_probe"] = transport_probe(
            args.job_id, args.extra_prefill_node, d["ip"],
            args.python, args.ib_device,
        )
    if prepared_cfg is not None:
        expected_hosts = {node["host_ip"] for node in prepared_cfg["nodes"]}
        allocated_hosts = {p["ip"], d["ip"]} | (
            {extra["ip"]} if extra else set()
        )
        if expected_hosts != allocated_hosts:
            raise RuntimeError(
                "prepared Slurm placement no longer matches the nodes' IB addresses"
            )
    manifest = {
        "job_id": str(args.job_id), "job_end_time": fields.get("EndTime"),
        "prefill": p, "decode": d, "ib_device": args.ib_device,
        "ib_interface": args.ib_interface, "cpus_per_node": args.cpus_per_node,
        "tp_size": args.tp_size,
        "groups_per_node": args.groups_per_node,
        "prefill_groups_per_node": args.prefill_groups_per_node,
        "decode_groups_per_node": args.decode_groups_per_node,
        "model": (
            prepared_cfg["model_path"]
            if prepared_cfg is not None else str(args.model.resolve())
        ),
        "data_root": (
            prepared_cfg.get("workload_environment", {}).get(
                "PD_DATA_ROOT", str(args.data_root.resolve())
            )
            if prepared_cfg is not None else str(args.data_root.resolve())
        ),
        "sglang_root": (
            prepared_cfg["sglang_root"]
            if prepared_cfg is not None else str(args.sglang_root.resolve())
        ),
        "swe_sandbox_backend": args.swe_sandbox_backend,
        "workload_config": str(workload_config.resolve()),
        "swe_sif_dir": str(args.swe_sif_dir.resolve()),
        "swe_work_root": str(swe_work_root),
        "apptainer_binary": str(args.apptainer_binary.resolve()),
        "cc": DEFAULT_CC, "cxx": DEFAULT_CXX, "nvcc_ccbin": DEFAULT_NVCC_CCBIN,
        "config": str(config_path) if config_path else None,
    }
    if extra is not None:
        manifest["extra_prefill"] = extra
        manifest["extra_job_id"] = str(args.extra_job_id)
    print(json.dumps(manifest, indent=2), flush=True)
    if args.action == "probe":
        return 0
    if run_dir is None or config_path is None:
        parser.error("--run-dir is required for every action except probe")
    if args.action in {"prepare", "plan"}:
        run_dir.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
        else:
            cfg = build_config(args, p, d, run_dir, workload_config, swe_work_root,
                               extra_prefill=extra)
            temporary = config_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(cfg, indent=2) + "\n")
            os.replace(temporary, config_path)
        if args.action == "prepare":
            print(config_path)
            return 0
        return subprocess.call([str(Path(args.python).resolve()),
                                str(TOOLS / "multinode.py"), "plan",
                                "--config", str(config_path)])
    if not config_path.is_file():
        parser.error(f"prepared config does not exist: {config_path}")
    cfg = json.loads(config_path.read_text())
    prefill_ids = [node["node_id"] for node in cfg["nodes"] if node["role"] == "prefill"]
    decode_ids = [node["node_id"] for node in cfg["nodes"] if node["role"] == "decode"]
    # Runtime actions consume the immutable prepared topology.  Do not make a
    # later operator repeat prepare-only TP/group flags or accidentally give
    # every TP2 worker the full-node CPU request.
    args.tp_size = int(cfg["tp_size"])
    args.groups_per_node = len(prefill_ids)
    if args.action == "start-control":
        return invoke_multinode(args, config_path, "start-control")
    if args.action == "control-ready":
        return invoke_multinode(args, config_path, "control-ready")
    physical_by_ip = {p["ip"]: args.prefill_node, d["ip"]: args.decode_node}
    if extra is not None:
        physical_by_ip[extra["ip"]] = args.extra_prefill_node
    if args.action == "start-prefill":
        if len(prefill_ids) > 1 and not args.detach:
            parser.error("multi-group worker startup requires --detach")
        coordinator = cfg["nodes"][0]
        result = invoke_multinode(args, config_path, "start-worker",
                                  node_id=coordinator["node_id"],
                                  slurm_node=physical_by_ip[coordinator["host_ip"]])
        if result:
            return result
        wait_model_endpoint(coordinator)
        return max((invoke_multinode(
            args, config_path, "start-worker", node_id=node["node_id"],
            slurm_node=physical_by_ip[node["host_ip"]],
        ) for node in cfg["nodes"] if node["role"] == "prefill" and
                  node["node_id"] != coordinator["node_id"]), default=0)
    if args.action == "start-decode":
        if len(decode_ids) > 1 and not args.detach:
            parser.error("multi-group worker startup requires --detach")
        return max((invoke_multinode(
            args, config_path, "start-worker", node_id=node["node_id"],
            slurm_node=physical_by_ip[node["host_ip"]],
        ) for node in cfg["nodes"] if node["role"] == "decode"), default=0)
    if args.action in {"wait-ready", "start-router", "run-workload"}:
        return invoke_multinode(args, config_path, args.action)
    if args.component is None:
        parser.error("status/stop requires --component")
    if args.component == "worker":
        role_nodes = [node for node in cfg["nodes"] if node["role"] == args.worker_role]
        return max(invoke_multinode(
            args, config_path, args.action, node_id=node["node_id"],
            slurm_node=physical_by_ip[node["host_ip"]], component="worker",
        ) for node in role_nodes)
    return invoke_multinode(args, config_path, args.action, component=args.component)


if __name__ == "__main__":
    raise SystemExit(main())
