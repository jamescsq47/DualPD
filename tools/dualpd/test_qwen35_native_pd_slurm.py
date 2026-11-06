import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "qwen35_native_pd_slurm", ROOT / "qwen35_native_pd_slurm.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

workset_spec = importlib.util.spec_from_file_location(
    "qwen35_v0521_tp_workset_consensus",
    ROOT.parents[1]
    / "examples/pd/scripts/compat/qwen35_mooncake_v0521/tp_workset_consensus.py",
)
workset = importlib.util.module_from_spec(workset_spec)
workset_spec.loader.exec_module(workset)


def test_qwen35_launcher_default_model_is_qwen35():
    assert module.DEFAULT_MODEL.name == "Qwen3.5-122B-A10B"


def test_v0521_tp_workset_waits_for_missing_or_different_peer():
    gathered = torch.zeros((2, 4, 3), dtype=torch.int64)
    gathered[0, 1] = torch.tensor((1, 77, 4))
    resolved, mismatches = workset.resolve_transfer_workset(
        gathered, [(1, 77, 4)], waiting_status=1
    )
    assert resolved == [1]
    assert mismatches == [(1, 77, 4)]

    gathered[1, 1] = torch.tensor((1, 88, 4))
    resolved, mismatches = workset.resolve_transfer_workset(
        gathered, [(1, 77, 4)], waiting_status=1
    )
    assert resolved == [1]
    assert mismatches == [(1, 77, 4)]


def test_v0521_tp_workset_propagates_failure_and_success_atomically():
    gathered = torch.zeros((2, 4, 3), dtype=torch.int64)
    gathered[0, 1] = torch.tensor((1, 77, 4))
    gathered[1, 1] = torch.tensor((1, 77, 0))
    resolved, mismatches = workset.resolve_transfer_workset(
        gathered, [(1, 77, 4)], waiting_status=1
    )
    assert resolved == [0]
    assert mismatches == []

    gathered[1, 1, 2] = 4
    resolved, mismatches = workset.resolve_transfer_workset(
        gathered, [(1, 77, 4)], waiting_status=1
    )
    assert resolved == [4]
    assert mismatches == []


def test_v0521_tp_identity_workset_tolerates_queue_order_and_length_skew():
    gathered = torch.zeros((2, 4, 2), dtype=torch.int64)
    gathered[0, 0] = torch.tensor((77, 2))
    gathered[0, 1] = torch.tensor((88, 2))
    gathered[1, 0] = torch.tensor((88, 2))

    resolved, mismatches = workset.resolve_identity_workset(
        gathered, [(77, 2), (88, 2)], waiting_status=1
    )
    assert resolved == [1, 2]
    assert mismatches == [(77, 2)]

    gathered[1, 1] = torch.tensor((77, 0))
    resolved, mismatches = workset.resolve_identity_workset(
        gathered, [(77, 2), (88, 2)], waiting_status=1
    )
    assert resolved == [0, 2]
    assert mismatches == []


def test_v0521_tp_identity_workset_rejects_duplicate_identity():
    gathered = torch.zeros((2, 3, 2), dtype=torch.int64)
    gathered[0, 0] = torch.tensor((77, 2))
    gathered[1, 0] = torch.tensor((77, 2))
    gathered[1, 1] = torch.tensor((77, 2))
    resolved, mismatches = workset.resolve_identity_workset(
        gathered, [(77, 2)], waiting_status=1
    )
    assert resolved == [1]
    assert mismatches == [(77, 2)]

def test_native_worker_environment_bounds_jit_threads_and_stack(tmp_path):
    args = SimpleNamespace(
        python=module.DEFAULT_PYTHON,
        local_root=tmp_path,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        ib_device="mlx5_1",
        ib_interface="ibp41s0f0",
    )
    environment = module.base_env(args, "decode-0", [0])
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert environment["GOMAXPROCS"] == "4"
    assert environment["MALLOC_ARENA_MAX"] == "2"
    assert environment["MALLOC_ARENA_TEST"] == "2"
    assert environment["SGLANG_STACK_SOFT_LIMIT_KIB"] == "2048"
    assert "MLIR_DISABLE_MULTITHREADING" not in environment


def test_native_worker_can_disable_late_mlir_thread_pool(tmp_path):
    args = SimpleNamespace(
        python=module.DEFAULT_PYTHON,
        local_root=tmp_path,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        ib_device="mlx5_1",
        ib_interface="ibp41s0f0",
        disable_mlir_multithreading=True,
    )
    environment = module.base_env(args, "prefill-0", [0])
    assert environment["MLIR_DISABLE_MULTITHREADING"] == "1"
    assert environment["TORCH_EXTENSIONS_DIR"] == str(
        tmp_path / "compiler-cache/torch-extensions"
    )


def test_component_open_file_limit_is_raised_without_exceeding_hard_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(module.resource, "getrlimit", lambda kind: (1024, 32768))
    monkeypatch.setattr(
        module.resource,
        "setrlimit",
        lambda kind, limits: calls.append((kind, limits)),
    )
    assert module.raise_nofile_soft_limit() == 32768
    assert calls == [(module.resource.RLIMIT_NOFILE, (32768, 32768))]


def test_cuda_runtime_path_uses_selected_environment_python_version(tmp_path):
    python = tmp_path / "env" / "bin" / "python3.12"
    python.parent.mkdir(parents=True)
    python.touch()
    site_packages = tmp_path / "env" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)

    assert module.environment_cuda_runtime_lib(python) == (
        site_packages / "nvidia/cuda_runtime/lib"
    )


def test_native_mooncake_staging_defaults_fit_slurm_cgroup(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_native_pd_slurm.py",
            "prepare",
            "--job-id",
            "1",
            "--run-dir",
            "/tmp/native-pd-test",
        ],
    )
    args = module.parse_args()
    assert args.decode_hicache_size_gb == 32
    assert args.mamba_hicache_size_gb == 32
    assert args.storage_job_id == ""
    assert args.storage_node == ""
    assert args.mooncake_compat is None
    assert args.mooncake_master_binary is None
    assert args.disable_mlir_multithreading is False
    assert args.mamba_full_memory_ratio == 0.9
    assert args.mamba_cache_strategy_flag == "radix"
    assert args.router_policy == "round_robin"


def test_native_mooncake_accepts_distinct_workload_and_storage_allocations(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_native_pd_slurm.py",
            "prepare",
            "--job-id",
            "1",
            "--run-dir",
            "/tmp/native-pd-test",
            "--native-mooncake",
            "--workload-job-id",
            "2",
            "--workload-node",
            "node-workload",
            "--storage-job-id",
            "3",
            "--storage-node",
            "node-storage",
        ],
    )
    args = module.parse_args()
    assert args.workload_job_id == "2"
    assert args.workload_node == "node-workload"
    assert args.storage_job_id == "3"
    assert args.storage_node == "node-storage"


def test_native_pd_manifest_supports_one_prefill_and_four_decode_tp2_groups(tmp_path):
    args = SimpleNamespace(
        run_id="native-pd-asymmetric-test", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen38"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=128, tp_size=2, groups_per_node=4,
        prefill_groups_per_node=1, decode_groups_per_node=4,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
    )
    manifest = module.build_manifest(
        args, ["node-p", "node-d"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}],
    )

    assert manifest["method"] == "native_pd_no_reverse"
    assert manifest["settings"]["prefill_groups"] == 1
    assert manifest["settings"]["decode_groups"] == 4
    assert [name for name in manifest["components"] if name.startswith("prefill-")] == [
        "prefill-0"
    ]
    assert [name for name in manifest["components"] if name.startswith("decode-")] == [
        "decode-0", "decode-1", "decode-2", "decode-3"
    ]
    workload = manifest["components"]["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == "24200"
    assert workload[workload.index("--decode-ports") + 1] == (
        "24300,24310,24320,24330"
    )
    router = manifest["components"]["router"]["command"]
    assert router.count("--prefill") == 1
    assert router.count("--decode") == 4
    assert router[router.index("--policy") + 1] == "round_robin"
    assert router[router.index("--health-check-timeout-secs") + 1] == "30"
    assert router[router.index("--health-failure-threshold") + 1] == "5"
    assert router[router.index("--request-timeout-secs") + 1] == "360"
    assert "--disable-circuit-breaker" in router
    prefill = manifest["components"]["prefill-0"]["command"]
    assert prefill[prefill.index("--mamba-scheduler-strategy") + 1] == "no_buffer"


def test_native_pd_manifest_supports_two_node_mixed_2p6d_tp2(tmp_path):
    args = SimpleNamespace(
        run_id="native-pd-mixed-test", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen38"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=192, tp_size=2, groups_per_node=3,
        prefill_groups_per_node=1, decode_groups_per_node=3,
        mixed_one_prefill_per_node=True,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
    )
    manifest = module.build_manifest(
        args, ["node-a", "node-b"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}],
    )

    assert manifest["settings"]["mixed_one_prefill_per_node"] is True
    assert manifest["settings"]["prefill_groups"] == 2
    assert manifest["settings"]["decode_groups"] == 6
    assert manifest["settings"]["router_policy"] == "round_robin"
    assert manifest["nodes"] == {
        "prefill": ["node-a", "node-b"],
        "decode": ["node-a", "node-b"],
    }
    components = manifest["components"]
    assert components["prefill-0"]["node"] == "node-a"
    assert components["prefill-0"]["gpus"] == [0, 1]
    assert components["decode-0"]["node"] == "node-a"
    assert components["decode-0"]["gpus"] == [2, 3]
    assert components["decode-2"]["gpus"] == [6, 7]
    assert components["prefill-1"]["node"] == "node-b"
    assert components["prefill-1"]["gpus"] == [0, 1]
    assert components["decode-3"]["node"] == "node-b"
    assert components["decode-5"]["gpus"] == [6, 7]
    workload = components["workload"]["command"]
    assert workload[workload.index("--prefill-endpoints") + 1] == (
        "http://10.0.0.1:24200,http://10.0.0.2:24200"
    )
    assert workload[workload.index("--decode-endpoints") + 1] == (
        "http://10.0.0.1:24300,http://10.0.0.1:24310,"
        "http://10.0.0.1:24320,http://10.0.0.2:24300,"
        "http://10.0.0.2:24310,http://10.0.0.2:24320"
    )


def test_native_pd_manifest_supports_single_node_2p6d_tp1(tmp_path):
    args = SimpleNamespace(
        run_id="native-pd-single-test", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen35-9b"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=256, tp_size=1, groups_per_node=6,
        prefill_groups_per_node=2, decode_groups_per_node=6,
        single_node_2p6d=True,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
    )
    manifest = module.build_manifest(
        args, ["node-a"], [{"ip": "10.0.0.1"}],
    )
    components = manifest["components"]
    assert manifest["settings"]["single_node_2p6d"] is True
    workers = [
        value for name, value in components.items()
        if name.startswith(("prefill-", "decode-"))
    ]
    assert [worker["gpus"] for worker in workers] == (
        [[0], [4], [1], [2], [3], [5], [6], [7]]
    )
    workload = components["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == "24200,24210"
    assert workload[workload.index("--decode-ports") + 1] == (
        "24300,24310,24320,24330,24340,24350"
    )


def test_native_pd_manifest_supports_single_node_2p2d_tp2(tmp_path):
    args = SimpleNamespace(
        run_id="native-pd-single-2p2d", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen38"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=192, tp_size=2, groups_per_node=2,
        prefill_groups_per_node=2, decode_groups_per_node=2,
        single_node_2p2d=True,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
    )
    manifest = module.build_manifest(
        args, ["node-a"], [{"ip": "10.0.0.1"}],
    )
    components = manifest["components"]
    workers = [
        value for name, value in components.items()
        if name.startswith(("prefill-", "decode-"))
    ]
    assert manifest["settings"]["single_node_2p2d"] is True
    assert [worker["gpus"] for worker in workers] == (
        [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    workload = components["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == "24200,24210"
    assert workload[workload.index("--decode-ports") + 1] == "24300,24310"


def test_native_pd_manifest_supports_single_node_1p3d_tp2(tmp_path):
    args = SimpleNamespace(
        run_id="native-pd-single-1p3d", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen38"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=192, tp_size=2, groups_per_node=1,
        prefill_groups_per_node=1, decode_groups_per_node=3,
        single_node_1p3d=True,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
    )
    manifest = module.build_manifest(
        args, ["node-a"], [{"ip": "10.0.0.1"}],
    )
    components = manifest["components"]
    workers = [
        value for name, value in components.items()
        if name.startswith(("prefill-", "decode-"))
    ]
    assert manifest["settings"]["single_node_1p3d"] is True
    assert [worker["gpus"] for worker in workers] == (
        [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    workload = components["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == "24200"
    assert workload[workload.index("--decode-ports") + 1] == (
        "24300,24310,24320"
    )


def test_three_node_native_pd_routes_two_tp8_prefills_to_one_decode(tmp_path):
    args = SimpleNamespace(
        run_id="native-pd-three", run_dir=tmp_path / "run", job_id="10",
        extra_job_id="11", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen35"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=128, tp_size=8, groups_per_node=1,
        prefill_groups_per_node=1, decode_groups_per_node=1,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
    )
    manifest = module.build_manifest(
        args, ["node-p0", "node-p1", "node-d"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}, {"ip": "10.0.0.3"}],
    )
    assert manifest["settings"]["prefill_groups"] == 2
    assert manifest["settings"]["decode_groups"] == 1
    assert manifest["components"]["decode-0"]["job_id"] == "11"
    router = manifest["components"]["router"]["command"]
    assert router.count("--prefill") == 2
    assert router.count("--decode") == 1
    workload = manifest["components"]["workload"]["command"]
    assert workload[workload.index("--prefill-endpoints") + 1] == (
        "http://10.0.0.1:24200,http://10.0.0.2:24200"
    )


def test_native_mooncake_manifest_uses_isolated_store_and_hybrid_compat(tmp_path):
    compat = tmp_path / "compat"
    compat.mkdir()
    args = SimpleNamespace(
        run_id="native-mooncake-test", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        sglang_root=module.DEFAULT_SGLANG_ROOT,
        model=Path("/models/qwen35"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/native-pd"), requests=500,
        concurrency=128, tp_size=8, groups_per_node=1,
        prefill_groups_per_node=1, decode_groups_per_node=1,
        prefill_port=24200, decode_port=24300, bootstrap_port=24500,
        router_port=24400, router_metrics_port=24401,
        ib_interface="ibp41s0f0", ib_device="mlx5_1",
        native_mooncake=True, mooncake_compat=compat,
        mooncake_master_binary=tmp_path / "fixed" / "mooncake_master",
        storage_node="node-storage", storage_ip="10.0.0.3",
        storage_job_id="456", prefill_hicache_size_gb=32,
        decode_hicache_size_gb=64, mamba_hicache_size_gb=32,
        mooncake_store_size_gb=512,
        mooncake_prefetch_threshold=64,
        mooncake_prefetch_policy="wait_complete", mooncake_protocol="tcp",
        mooncake_master_port=25151, mooncake_metadata_port=25180,
        mooncake_metrics_port=25103, mooncake_client_port=25152,
        mooncake_client_http_port=25190,
    )
    manifest = module.build_manifest(
        args, ["node-p", "node-d"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}],
    )

    assert manifest["method"] == "native_pd_hicache_mooncake"
    assert manifest["settings"]["d_to_p"] == "native_hicache_mooncake"
    assert manifest["settings"]["mooncake_store_size_gb"] == 512
    master = manifest["components"]["mooncake-master"]
    client = manifest["components"]["mooncake-client"]
    assert master["node"] == client["node"] == "node-storage"
    assert master["job_id"] == client["job_id"] == "456"
    assert master["environment"]["SGLANG_STACK_SOFT_LIMIT_KIB"] == "2048"
    assert client["environment"]["SGLANG_STACK_SOFT_LIMIT_KIB"] == "2048"
    assert master["command"][0] == str(
        (tmp_path / "fixed" / "mooncake_master").resolve()
    )
    assert master["environment"]["LD_LIBRARY_PATH"].startswith(
        "/usr/lib/x86_64-linux-gnu:"
    )
    assert "--eviction_ratio=0.05" in master["command"]
    assert "--global_segment_size=512 GB" in client["command"]

    prefill = manifest["components"]["prefill-0"]
    decode = manifest["components"]["decode-0"]
    assert "--enable-hierarchical-cache" in prefill["command"]
    assert "--skip-server-warmup" not in prefill["command"]
    assert "--skip-server-warmup" not in decode["command"]
    assert "--disable-radix-cache" not in prefill["command"]
    assert "--disaggregation-decode-enable-offload-kvcache" in decode["command"]
    assert prefill["command"][prefill["command"].index("--mamba-scheduler-strategy") + 1] == "extra_buffer"
    assert prefill["command"][prefill["command"].index("--hicache-size") + 1] == "32"
    assert prefill["command"][prefill["command"].index("--hicache-storage-prefetch-policy") + 1] == "wait_complete"
    assert prefill["command"][prefill["command"].index("--hicache-mem-layout") + 1] == "page_first_direct"
    assert prefill["command"][prefill["command"].index("--hicache-io-backend") + 1] == "direct"
    assert decode["command"][decode["command"].index("--hicache-size") + 1] == "64"
    assert decode["command"][decode["command"].index("--mamba-full-memory-ratio") + 1] == "0.9"
    assert manifest["settings"]["mamba_full_memory_ratio"] == 0.9
    assert prefill["environment"]["SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB"] == "32"
    assert decode["environment"]["SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB"] == "32"
    assert manifest["settings"]["mamba_hicache_size_gb_per_rank"] == 32
    assert prefill["environment"]["PYTHONPATH"].split(":")[0] == str(compat)
    assert decode["environment"]["PYTHONPATH"].split(":")[0] == str(compat)
    workload = manifest["components"]["workload"]["command"]
    assert "--pd-enable-decode-offload-kvcache" in workload
    assert workload[workload.index("--pd-hicache-storage-backend") + 1] == "mooncake"
