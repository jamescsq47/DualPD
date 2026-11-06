import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "qwen35_colocated_slurm", ROOT / "qwen35_colocated_slurm.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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


def test_default_model_is_qwen38_27b():
    assert module.DEFAULT_MODEL.name == "Qwen3.8-27B"


def test_colocated_manifest_keeps_pd_settings_but_has_no_pd_mode(tmp_path):
    args = SimpleNamespace(
        run_id="colocated-test",
        run_dir=tmp_path / "run",
        job_id="123",
        python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        model=Path("/models/qwen"),
        data_root=Path("/data"),
        swe_sif_dir=Path("/sif"),
        swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/colocated"),
        requests=500,
        concurrency=128,
        cpus_per_node=224,
        worker_port=24000,
        router_port=24002,
        metrics_tunnel_port=24010,
        tp_size=8,
        groups_per_node=1,
    )
    manifest = module.build_manifest(
        args,
        ["node0", "node1"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}],
    )
    assert manifest["method"] == "native_colocated"
    assert manifest["settings"]["mamba_full_memory_ratio"] == 0.5
    assert manifest["settings"]["mem_fraction_static"] == 0.8
    assert manifest["settings"]["tp"] == 8
    assert manifest["settings"]["replicas"] == 2
    for name in ("collocated-0", "collocated-1"):
        command = manifest["components"][name]["command"]
        environment = manifest["components"][name]["environment"]
        assert "--disaggregation-mode" not in command
        assert command[command.index("--mamba-full-memory-ratio") + 1] == "0.5"
        assert command[command.index("--mem-fraction-static") + 1] == "0.8"
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert environment["MALLOC_ARENA_MAX"] == "2"
        assert environment["MALLOC_ARENA_TEST"] == "2"
        assert environment["SGLANG_STACK_SOFT_LIMIT_KIB"] == "2048"
        assert environment["PYTHONPATH"] == str(
            module.DEFAULT_SGLANG_ROOT.resolve() / "python"
        )
    workload = manifest["components"]["workload"]
    assert workload["command"][0] == str(Path(module.DEFAULT_WORKLOAD_PYTHON).resolve())
    assert manifest["baseline_python"] != manifest["workload_python"]
    assert "--preserve-source-order" in workload["command"]
    assert workload["command"][workload["command"].index("--max-inflight") + 1] == "128"
    assert workload["environment"]["PD_SWE_APPTAINER_IMAGE_DIR"] == "/sif"


def test_three_node_manifest_has_three_tp8_replicas_and_two_jobs(tmp_path):
    args = SimpleNamespace(
        run_id="colocated-three", run_dir=tmp_path / "run", job_id="10",
        extra_job_id="11", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        model=Path("/models/qwen35"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/colocated"), requests=500,
        concurrency=128, cpus_per_node=224, worker_port=24000,
        router_port=24002, metrics_tunnel_port=24100, tp_size=8,
        groups_per_node=1,
    )
    manifest = module.build_manifest(
        args, ["node0", "node1", "node2"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}, {"ip": "10.0.0.3"}],
    )
    assert manifest["settings"]["replicas"] == 3
    assert manifest["components"]["collocated-2"]["job_id"] == "11"
    assert "metrics-tunnel" not in manifest["components"]
    command = manifest["components"]["workload"]["command"]
    assert command[command.index("--prefill-endpoints") + 1] == (
        "http://10.0.0.1:24000,http://10.0.0.2:24000,http://10.0.0.3:24000"
    )


def test_colocated_manifest_expands_eight_tp2_replicas(tmp_path):
    args = SimpleNamespace(
        run_id="colocated-tp2-test",
        run_dir=tmp_path / "run",
        job_id="123",
        python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        model=Path("/models/qwen38"),
        data_root=Path("/data"),
        swe_sif_dir=Path("/sif"),
        swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/colocated"),
        requests=500,
        concurrency=128,
        cpus_per_node=224,
        worker_port=24000,
        router_port=24050,
        metrics_tunnel_port=24100,
        tp_size=2,
        groups_per_node=4,
    )
    manifest = module.build_manifest(
        args,
        ["node0", "node1"],
        [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}],
    )
    workers = [
        value for name, value in manifest["components"].items()
        if name.startswith("collocated-")
    ]
    assert manifest["settings"]["tp"] == 2
    assert manifest["settings"]["replicas"] == 8
    assert len(workers) == 8
    assert [worker["gpus"] for worker in workers[:4]] == [
        [0, 1], [2, 3], [4, 5], [6, 7]
    ]
    assert [worker["port"] for worker in workers[:4]] == [
        24000, 24010, 24020, 24030
    ]
    router = manifest["components"]["router"]["command"]
    assert sum(value.startswith("http://") for value in router) == 8
    assert router[router.index("--prometheus-port") + 1] == "24051"
    workload = manifest["components"]["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == (
        "24000,24010,24020,24030,24100,24110,24120,24130"
    )


def test_colocated_manifest_expands_eight_tp1_replicas_on_one_node(tmp_path):
    args = SimpleNamespace(
        run_id="colocated-single-tp1", run_dir=tmp_path / "run", job_id="123",
        python=module.DEFAULT_PYTHON, workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        model=Path("/models/qwen35-9b"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/colocated"), requests=500,
        concurrency=256, cpus_per_node=224, worker_port=24000,
        router_port=24090, metrics_tunnel_port=24100, tp_size=1,
        groups_per_node=8, node0_groups=8,
    )
    manifest = module.build_manifest(
        args, ["node0"], [{"ip": "10.0.0.1"}],
    )
    workers = [
        value for name, value in manifest["components"].items()
        if name.startswith("collocated-")
    ]
    assert manifest["settings"]["replicas"] == 8
    assert [worker["gpus"] for worker in workers] == [[i] for i in range(8)]
    assert "metrics-tunnel" not in manifest["components"]
    workload = manifest["components"]["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == (
        "24000,24010,24020,24030,24040,24050,24060,24070"
    )


def test_colocated_manifest_supports_one_plus_four_tp2_replicas(tmp_path):
    args = SimpleNamespace(
        run_id="colocated-asymmetric-test", run_dir=tmp_path / "run",
        job_id="123", python=module.DEFAULT_PYTHON,
        workload_python=module.DEFAULT_WORKLOAD_PYTHON,
        model=Path("/models/qwen38"), data_root=Path("/data"),
        swe_sif_dir=Path("/sif"), swe_work_root=Path("/var/tmp/user/swe"),
        workload_config=Path("/workload.yaml"),
        local_root=Path("/var/tmp/user/colocated"), requests=500,
        concurrency=128, cpus_per_node=224, worker_port=24000,
        router_port=24050, metrics_tunnel_port=24100, tp_size=2,
        groups_per_node=4, node0_groups=1, node1_groups=4,
    )
    manifest = module.build_manifest(
        args, ["node0", "node1"], [{"ip": "10.0.0.1"}, {"ip": "10.0.0.2"}],
    )
    workers = [
        value for name, value in manifest["components"].items()
        if name.startswith("collocated-")
    ]
    assert manifest["settings"]["replicas"] == 5
    assert manifest["settings"]["node0_groups"] == 1
    assert manifest["settings"]["node1_groups"] == 4
    assert [worker["gpus"] for worker in workers] == [
        [0, 1], [0, 1], [2, 3], [4, 5], [6, 7]
    ]
    workload = manifest["components"]["workload"]["command"]
    assert workload[workload.index("--prefill-ports") + 1] == (
        "24000,24100,24110,24120,24130"
    )


def test_component_scrubs_external_sglang_pd_and_nccl_environment(tmp_path):
    local_dir = tmp_path / "worker"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "method": "native_colocated",
        "run_id": "clean-env",
        "components": {
            "collocated-0": {
                "engine_id": "collocated-0",
                "local_dir": str(local_dir),
                "environment": {"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "true"},
                "command": ["/bin/true"],
            }
        },
    }))
    dirty = {
        "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        "PD_UNDECLARED": "1",
        "NCCL_ALGO": "Tree",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    with patch.dict(os.environ, dirty, clear=False), \
         patch.object(module, "supervise", return_value=0) as supervise:
        assert module.component_action(
            manifest_path, "collocated-0", "start"
        ) == 0
    environment = supervise.call_args.args[1]
    assert environment["SGLANG_ENABLE_METRICS_DEVICE_TIMER"] == "true"
    for key in dirty:
        assert key not in environment


def test_start_health_failure_rolls_back_every_started_component(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "method": "native_colocated",
        "run_id": "rollback-test",
        "job_id": "7",
        "settings": {"groups_per_node": 1},
        "components": {
            "collocated-0": {"health": "http://one/model_info"},
            "collocated-1": {"health": "http://two/model_info"},
            "router": {"health": "http://one/router/health"},
            "metrics-tunnel": {},
            "workload": {},
        },
    }
    (run_dir / "colocated-manifest.json").write_text(json.dumps(manifest))
    args = SimpleNamespace(
        action="start", job_id="7", run_dir=run_dir, run_id="",
        manifest=None, component=None, component_action=None,
        tp_size=8, groups_per_node=1,
    )
    stopped = []
    with patch.object(module, "parse_args", return_value=args), \
         patch.object(module, "detached_srun") as detached, \
         patch.object(module, "wait_http", side_effect=RuntimeError("health failed")), \
         patch.object(module, "remote_control", side_effect=lambda a, p, c, x: stopped.append(c) or 0):
        with pytest.raises(RuntimeError, match="health failed"):
            module.main()
    assert [call.args[2] for call in detached.call_args_list] == [
        "collocated-0", "collocated-1"
    ]
    assert stopped == ["collocated-1", "collocated-0"]


def test_worker_srun_exposes_full_node_cpuset_for_numa_binding(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "method": "native_colocated",
        "run_id": "numa-cpuset",
        "job_id": "7",
        "run_dir": str(tmp_path),
        "settings": {"groups_per_node": 4},
        "components": {
            "collocated-0-3": {"node": "node0"},
        },
    }))
    args = SimpleNamespace(job_id="7", cpus_per_node=224)
    process = SimpleNamespace(pid=9)
    with patch.object(module.subprocess, "Popen", return_value=process) as popen:
        module.detached_srun(args, manifest_path, "collocated-0-3")
    command = popen.call_args.args[0]
    assert command[command.index("--cpus-per-task") + 1] == "224"
