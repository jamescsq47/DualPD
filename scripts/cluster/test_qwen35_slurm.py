import unittest
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import qwen35_slurm as s


class Qwen35SlurmTests(unittest.TestCase):
    def test_default_model_is_qwen38_27b(self):
        self.assertEqual(s.DEFAULT_MODEL.name, "Qwen3.8-27B")

    def test_build_config_selects_minimax_tp8_ep8_parsers(self):
        args = SimpleNamespace(
            run_id="minimax-m27", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/minimax-m27"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=8,
            concurrency=4, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-p",
            decode_node="node-d", ib_device="mlx5_1", tp_size=8,
            groups_per_node=1, d2p_host_gib_per_rank=64,
            p2d_host_gib_per_rank=32, model_family="minimax_m2",
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.2"},
            Path("/results/minimax-m27"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )
        self.assertEqual(cfg["model_family"], "minimax_m2")
        self.assertEqual(cfg["ep_size"], 8)
        self.assertEqual(cfg["reasoning_parser"], "minimax-append-think")
        self.assertEqual(cfg["tool_call_parser"], "minimax-m2")
        self.assertEqual(cfg["tp_size"], 8)

    def test_allocation_requires_owned_running_two_node_job(self):
        values = [
            ("JobId=7 JobState=RUNNING UserId=tester(1) NodeList=node[1-2] EndTime=later "
             "OverSubscribe=NO TresPerNode=gres/gpu:h100:8"),
            "node1\nnode2",
        ]
        with patch.object(s, "output", side_effect=values), \
             patch.object(s.getpass, "getuser", return_value="tester"):
            fields, nodes = s.allocation("7")
        self.assertEqual(nodes, ["node1", "node2"])
        self.assertEqual(fields["EndTime"], "later")

    def test_node_inventory_requires_tp8_ib_and_unlimited_memlock(self):
        report = ("host=node1700\ngpus=8\ngpu_name=NVIDIA H100 80GB HBM3\n"
                  "gpu_min_mib=81559\nmem_kib=2113372736\nmemlock=unlimited\n"
                  "docker=\npodman=\napptainer=\nslurm_oci=disabled\n"
                  "rdma_state=4: ACTIVE\nip=172.16.17.0")
        with patch.object(s, "on_node", return_value=report):
            result = s.node_inventory("7", "node1700", "ibp41s0f0", "mlx5_1")
        self.assertEqual(result["ip"], "172.16.17.0")
        self.assertEqual(result["gpus"], "8")

    def test_node_inventory_rejects_partial_gpu_node(self):
        report = ("host=node1700\ngpus=4\ngpu_name=NVIDIA H100 80GB HBM3\n"
                  "gpu_min_mib=81559\nmem_kib=1\nmemlock=unlimited\n"
                  "docker=\npodman=\napptainer=\nslurm_oci=disabled\n"
                  "rdma_state=4: ACTIVE\nip=172.16.17.0")
        with patch.object(s, "on_node", return_value=report):
            with self.assertRaisesRegex(RuntimeError, "exactly eight"):
                s.node_inventory("7", "node1700", "ibp41s0f0", "mlx5_1")

    def test_transport_probe_is_explicitly_not_a_bandwidth_claim(self):
        with patch.object(s, "on_node", return_value="") as on_node:
            result = s.transport_probe("7", "node1700", "172.16.27.0",
                                       "/env/bin/python", "mlx5_1")
        self.assertTrue(result["peer_route_present"])
        self.assertTrue(result["nixl_ucx_loads"])
        self.assertIn("not an RDMA bandwidth", result["note"])
        self.assertIn("UCX_NET_DEVICES=mlx5_1:1", on_node.call_args.args[2][-1])
        self.assertNotIn("/dev/tcp", on_node.call_args.args[2][-1])

    def test_runtime_restores_prepared_physical_placement(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "multinode-v3.json"
            config.write_text('{"slurm_nodes":{"prefill":"node2","decode":"node1"}}')
            args = SimpleNamespace(prefill_node=None, decode_node=None)
            value = s.restore_prepared_placement(args, config, ["node1", "node2"])
            self.assertEqual(value["slurm_nodes"]["prefill"], "node2")
            self.assertEqual(args.prefill_node, "node2")
            self.assertEqual(args.decode_node, "node1")

            conflicting = SimpleNamespace(prefill_node="node1", decode_node=None)
            with self.assertRaisesRegex(RuntimeError, "conflicts with prepared"):
                s.restore_prepared_placement(
                    conflicting, config, ["node1", "node2"]
                )

    def test_triton_uses_system_compiler_not_site_spack_path(self):
        self.assertEqual(s.DEFAULT_CC, "/usr/bin/gcc")
        self.assertEqual(s.DEFAULT_CXX, "/usr/bin/g++")
        self.assertEqual(s.DEFAULT_NVCC_CCBIN, "/usr/bin/g++")
        self.assertEqual(s.DEFAULT_SYSTEM_BIN, "/usr/bin")

    def test_large_mutable_state_defaults_to_var_tmp_and_pool(self):
        self.assertTrue(str(s.DEFAULT_LOCAL_ROOT).startswith("/var/tmp/"))
        self.assertTrue(str(s.DEFAULT_SIF_DIR).startswith("/orcd/pool/004/"))

    def test_build_config_keeps_upstream_v3_and_site_settings_separate(self):
        args = SimpleNamespace(
            run_id="site-v3-test", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=128, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-p",
            decode_node="node-d", ib_device="mlx5_1", tp_size=8,
            groups_per_node=1, d2p_host_gib_per_rank=64,
            p2d_host_gib_per_rank=32,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.2"},
            Path("/results/site-v3-test"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertEqual(cfg["model_family"], "qwen35_moe")
        self.assertEqual(cfg["slurm_nodes"], {"prefill": "node-p", "decode": "node-d"})
        self.assertEqual(cfg["tp_size"], 8)
        self.assertEqual(cfg["d2p_host_gib_per_rank"], 64)
        self.assertEqual(cfg["p2d_host_gib_per_rank"], 32)
        self.assertEqual(cfg["direct_lanes"], 2)
        self.assertEqual(cfg["d2p_direct_lanes"], 2)
        self.assertEqual(cfg["p2d_direct_lanes"], 2)
        self.assertEqual(cfg["host_lanes"], 2)
        self.assertEqual(cfg["host_load_workers"], 1)
        self.assertEqual(cfg["nixl_progress_threads"], 2)
        self.assertEqual(cfg["forward_stream_priority"], -5)
        self.assertEqual(cfg["flashinfer_allreduce_max_tokens"], 8192)
        self.assertEqual(cfg["prefill_mamba_full_memory_ratio"], 0.5)
        self.assertEqual(cfg["direct_admission_seconds"], 1)
        self.assertEqual(cfg["system_bin"], "/usr/bin")
        self.assertEqual(cfg["nodes"][0]["numa_nodes"], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(
            cfg["workload_environment"]["PD_SWE_APPTAINER_IMAGE_DIR"],
            "/images",
        )

    def test_build_config_accepts_directional_direct_lane_limits(self):
        args = SimpleNamespace(
            run_id="directional-lanes", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=8,
            concurrency=4, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-p",
            decode_node="node-d", ib_device="mlx5_1", tp_size=8,
            groups_per_node=1, d2p_host_gib_per_rank=64,
            p2d_host_gib_per_rank=32, direct_lanes=2,
            d2p_direct_lanes=2, p2d_direct_lanes=1,
        )

        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.2"},
            Path("/results/directional-lanes"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertEqual(cfg["direct_lanes"], 2)
        self.assertEqual(cfg["d2p_direct_lanes"], 2)
        self.assertEqual(cfg["p2d_direct_lanes"], 1)
        self.assertNotIn("control_root", cfg)

    def test_three_node_tp8_host_capacity_and_placement(self):
        args = SimpleNamespace(
            run_id="three-node", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen35"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=128, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-p0",
            extra_prefill_node="node-p1", decode_node="node-d",
            job_id="10", extra_job_id="11", ib_device="mlx5_1",
            tp_size=8, groups_per_node=1, d2p_host_gib_per_rank=64,
            p2d_host_gib_per_rank=32,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.3"},
            Path("/results/three-node"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"), extra_prefill={"ip": "10.0.0.2"},
        )
        self.assertEqual([n["engine_id"] for n in cfg["nodes"]],
                         ["prefill-0", "prefill-1", "decode-0"])
        self.assertEqual(cfg["slurm_node_jobs"],
                         {"node-p0": "10", "node-p1": "10", "node-d": "11"})
        self.assertEqual(cfg["d2p_host_gib_per_rank"] * 8, 512)
        self.assertEqual(cfg["p2d_host_gib_per_rank"] * 8, 256)
        command = cfg["workload_command"]
        self.assertEqual(command[command.index("--prefill-endpoints") + 1],
                         "http://10.0.0.1:23900,http://10.0.0.2:23900")

    def test_build_config_expands_four_nonoverlapping_tp2_groups_per_node(self):
        args = SimpleNamespace(
            run_id="site-tp2-test", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen38"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=128, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-p",
            decode_node="node-d", ib_device="mlx5_1", tp_size=2,
            groups_per_node=4, d2p_host_gib_per_rank=64,
            p2d_host_gib_per_rank=32,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.2"},
            Path("/results/site-tp2-test"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertEqual(cfg["tp_size"], 2)
        self.assertEqual(cfg["d2p_host_gib_per_rank"] * 8, 512)
        self.assertEqual(cfg["p2d_host_gib_per_rank"] * 8, 256)
        self.assertEqual(len(cfg["nodes"]), 8)
        self.assertEqual([node["node_id"] for node in cfg["nodes"]],
                         ["p0", "p1", "p2", "p3", "d0", "d1", "d2", "d3"])
        self.assertEqual([node["gpus"] for node in cfg["nodes"][:4]],
                         [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertEqual([node["port"] for node in cfg["nodes"][:4]],
                         [23900, 23910, 23920, 23930])
        command = cfg["workload_command"]
        self.assertEqual(command[command.index("--prefill-ports") + 1],
                         "23900,23910,23920,23930")
        self.assertEqual(command[command.index("--decode-ports") + 1],
                         "23900,23910,23920,23930")

    def test_build_config_places_one_p_and_three_d_on_each_of_two_nodes(self):
        args = SimpleNamespace(
            run_id="mixed-2p6d", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen38"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=192, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-a",
            decode_node="node-b", ib_device="mlx5_1", tp_size=2,
            groups_per_node=1, prefill_groups_per_node=1,
            decode_groups_per_node=3, mixed_one_prefill_per_node=True,
            d2p_host_gib_per_rank=64, p2d_host_gib_per_rank=32,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.2"},
            Path("/results/mixed-2p6d"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertTrue(cfg["mixed_one_prefill_per_node"])
        self.assertEqual(
            [(n["node_id"], n["role"], n["host_ip"], n["gpus"])
             for n in cfg["nodes"]],
            [
                ("p0", "prefill", "10.0.0.1", [0, 1]),
                ("d0", "decode", "10.0.0.1", [2, 3]),
                ("d1", "decode", "10.0.0.1", [4, 5]),
                ("d2", "decode", "10.0.0.1", [6, 7]),
                ("p1", "prefill", "10.0.0.2", [0, 1]),
                ("d3", "decode", "10.0.0.2", [2, 3]),
                ("d4", "decode", "10.0.0.2", [4, 5]),
                ("d5", "decode", "10.0.0.2", [6, 7]),
            ],
        )
        command = cfg["workload_command"]
        self.assertEqual(
            command[command.index("--prefill-endpoints") + 1],
            "http://10.0.0.1:23900,http://10.0.0.2:23900",
        )
        self.assertEqual(
            command[command.index("--decode-endpoints") + 1],
            "http://10.0.0.1:23910,http://10.0.0.1:23920,"
            "http://10.0.0.1:23930,http://10.0.0.2:23910,"
            "http://10.0.0.2:23920,http://10.0.0.2:23930",
        )

    def test_build_config_places_tp1_2p6d_on_one_node(self):
        args = SimpleNamespace(
            run_id="single-2p6d", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen35-9b"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=256, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-a",
            decode_node="node-a", ib_device="mlx5_1", tp_size=1,
            groups_per_node=1, prefill_groups_per_node=2,
            decode_groups_per_node=6, single_node_2p6d=True,
            d2p_host_gib_per_rank=64, p2d_host_gib_per_rank=32,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.1"},
            Path("/results/single-2p6d"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertTrue(cfg["single_node_2p6d"])
        self.assertEqual([n["gpus"] for n in cfg["nodes"]],
                         [[0], [4], [1], [2], [3], [5], [6], [7]])
        self.assertEqual(len({n["port"] for n in cfg["nodes"]}), 8)
        self.assertEqual(
            sum(n.get("d2p_host_gib_per_rank", 0) for n in cfg["nodes"]), 512
        )
        self.assertEqual(
            sum(n.get("p2d_host_gib_per_rank", 0) for n in cfg["nodes"]), 256
        )
        self.assertEqual(cfg["d2p_host_gib_total"], 512)
        self.assertEqual(cfg["p2d_host_gib_total"], 256)
        command = cfg["workload_command"]
        self.assertEqual(command[command.index("--prefill-ports") + 1],
                         "23900,23910")
        self.assertEqual(command[command.index("--decode-ports") + 1],
                         "24000,24010,24020,24030,24040,24050")
        self.assertEqual(
            command[command.index("--decode-endpoints") + 1],
            "http://10.0.0.1:24000,http://10.0.0.1:24010,"
            "http://10.0.0.1:24020,http://10.0.0.1:24030,"
            "http://10.0.0.1:24040,http://10.0.0.1:24050",
        )

    def test_build_config_places_tp2_2p2d_on_one_node(self):
        args = SimpleNamespace(
            run_id="single-2p2d", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen38"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=192, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-a",
            decode_node="node-a", ib_device="mlx5_1", tp_size=2,
            groups_per_node=2, prefill_groups_per_node=2,
            decode_groups_per_node=2, single_node_2p2d=True,
            d2p_host_gib_per_rank=128, p2d_host_gib_per_rank=64,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.1"},
            Path("/results/single-2p2d"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertTrue(cfg["single_node_2p2d"])
        self.assertEqual(
            [(node["node_id"], node["role"], node["gpus"])
             for node in cfg["nodes"]],
            [
                ("p0", "prefill", [0, 1]),
                ("d0", "decode", [2, 3]),
                ("p1", "prefill", [4, 5]),
                ("d1", "decode", [6, 7]),
            ],
        )
        self.assertEqual(cfg["d2p_host_gib_total"], 512)
        self.assertEqual(cfg["p2d_host_gib_total"], 256)
        command = cfg["workload_command"]
        self.assertEqual(command[command.index("--prefill-ports") + 1],
                         "23900,23910")
        self.assertEqual(command[command.index("--decode-ports") + 1],
                         "24000,24010")
        self.assertEqual(
            command[command.index("--prefill-endpoints") + 1],
            "http://10.0.0.1:23900,http://10.0.0.1:23910",
        )
        self.assertEqual(
            command[command.index("--decode-endpoints") + 1],
            "http://10.0.0.1:24000,http://10.0.0.1:24010",
        )

    def test_build_config_places_tp2_1p3d_on_one_node(self):
        args = SimpleNamespace(
            run_id="single-1p3d", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen38"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=192, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-a",
            decode_node="node-a", ib_device="mlx5_1", tp_size=2,
            groups_per_node=1, prefill_groups_per_node=1,
            decode_groups_per_node=3, single_node_1p3d=True,
            decode_growth_tokens=131072,
            d2p_host_gib_per_rank=85, p2d_host_gib_per_rank=128,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.1"},
            Path("/results/single-1p3d"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertTrue(cfg["single_node_1p3d"])
        self.assertEqual(
            [(node["node_id"], node["role"], node["gpus"])
             for node in cfg["nodes"]],
            [
                ("p0", "prefill", [0, 1]),
                ("d0", "decode", [2, 3]),
                ("d1", "decode", [4, 5]),
                ("d2", "decode", [6, 7]),
            ],
        )
        self.assertEqual(
            sum(
                node.get("d2p_host_gib_per_rank", 0) * len(node["gpus"])
                for node in cfg["nodes"]
            ),
            512,
        )
        self.assertEqual(
            sum(
                node.get("p2d_host_gib_per_rank", 0) * len(node["gpus"])
                for node in cfg["nodes"]
            ),
            256,
        )
        self.assertEqual(cfg["d2p_host_gib_total"], 512)
        self.assertEqual(cfg["p2d_host_gib_total"], 256)
        self.assertEqual(cfg["decode_growth_tokens"], 131072)
        command = cfg["workload_command"]
        self.assertEqual(command[command.index("--prefill-ports") + 1], "23900")
        self.assertEqual(command[command.index("--decode-ports") + 1],
                         "24000,24010,24020")
        self.assertEqual(
            command[command.index("--prefill-endpoints") + 1],
            "http://10.0.0.1:23900",
        )
        self.assertEqual(
            command[command.index("--decode-endpoints") + 1],
            "http://10.0.0.1:24000,http://10.0.0.1:24010,"
            "http://10.0.0.1:24020",
        )

    def test_build_config_supports_one_prefill_and_four_decode_tp2_groups(self):
        args = SimpleNamespace(
            run_id="site-asymmetric-test", control_token="x" * 32,
            python="/env/bin/python", model=Path("/models/qwen38"),
            sglang_root=Path("/src/sglang"), data_root=Path("/data"),
            local_root=Path("/tmp/tester/dualpd-v3"), requests=500,
            concurrency=128, swe_sandbox_backend="apptainer",
            swe_sif_dir=Path("/images"), prefill_node="node-p",
            decode_node="node-d", ib_device="mlx5_1", tp_size=2,
            groups_per_node=4, prefill_groups_per_node=1,
            decode_groups_per_node=4, d2p_host_gib_per_rank=64,
            p2d_host_gib_per_rank=32,
        )
        cfg = s.build_config(
            args, {"ip": "10.0.0.1"}, {"ip": "10.0.0.2"},
            Path("/results/site-asymmetric-test"), Path("/config/workload.yaml"),
            Path("/tmp/tester/pd-swe"),
        )

        self.assertEqual([node["node_id"] for node in cfg["nodes"]],
                         ["p0", "d0", "d1", "d2", "d3"])
        command = cfg["workload_command"]
        self.assertEqual(command[command.index("--prefill-ports") + 1], "23900")
        self.assertEqual(command[command.index("--decode-ports") + 1],
                         "23900,23910,23920,23930")

    def test_detached_srun_owns_durable_stdio_and_process_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "multinode-v3.json"
            config_path.write_text("{}\n")
            args = SimpleNamespace(
                job_id="7", prefill_node="node-p", cpus_per_node=224,
                python="/env/bin/python", detach=True, groups_per_node=1,
            )
            process = SimpleNamespace(pid=1234)
            with patch.dict(s.os.environ, {
                "SGLANG_BAD": "1", "PD_BAD": "1", "NCCL_ALGO": "Tree"
            }, clear=False), patch.object(
                s.subprocess, "Popen", return_value=process
            ) as popen:
                result = s.invoke_multinode(
                    args, config_path, "start-control", component="control"
                )

            self.assertEqual(result, 0)
            command = popen.call_args.args[0]
            options = popen.call_args.kwargs
            self.assertEqual(command[:4], ["srun", "--jobid", "7", "--overlap"])
            self.assertEqual(
                command[command.index("--cpus-per-task") + 1], "64"
            )
            self.assertIs(options["stdin"], subprocess.DEVNULL)
            self.assertIs(options["stderr"], subprocess.STDOUT)
            self.assertTrue(options["start_new_session"])
            self.assertTrue(options["close_fds"])
            self.assertNotIn("SGLANG_BAD", options["env"])
            self.assertNotIn("PD_BAD", options["env"])
            self.assertNotIn("NCCL_ALGO", options["env"])
            self.assertEqual(
                Path(options["stdout"].name),
                Path(temporary) / "launcher/control-node-p.log",
            )

    def test_worker_srun_exposes_full_node_cpuset_for_numa_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "multinode-v3.json"
            config_path.write_text("{}\n")
            args = SimpleNamespace(
                job_id="7", prefill_node="node-p", cpus_per_node=224,
                python="/env/bin/python", detach=True, groups_per_node=4,
            )
            process = SimpleNamespace(pid=1234)
            with patch.object(s.subprocess, "Popen", return_value=process) as popen:
                s.invoke_multinode(
                    args, config_path, "start-worker", node_id="p3",
                    slurm_node="node-p",
                )
            command = popen.call_args.args[0]
            self.assertEqual(
                command[command.index("--cpus-per-task") + 1], "224"
            )

    def test_multigroup_prefill_establishes_coordinator_before_other_workers(self):
        args = SimpleNamespace()
        config_path = Path("/results/run/multinode-v3.json")
        cfg = {
            "nodes": [
                {"node_id": "p0", "host_ip": "10.0.0.1", "port": 23900},
                {"node_id": "p1", "host_ip": "10.0.0.1", "port": 23910},
                {"node_id": "p2", "host_ip": "10.0.0.1", "port": 23920},
            ]
        }
        events = []

        def invoke(*_args, node_id=None, **_kwargs):
            events.append(("start", node_id))
            return 0

        def ready(node):
            events.append(("ready", node["node_id"]))

        with patch.object(s, "invoke_multinode", side_effect=invoke), \
             patch.object(s, "wait_model_endpoint", side_effect=ready):
            result = s.start_worker_set(
                args, config_path, ["p0", "p1", "p2"], "node-p", cfg,
                establish_coordinator=True,
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            events,
            [("start", "p0"), ("ready", "p0"),
             ("start", "p1"), ("start", "p2")],
        )

    def test_decode_workers_do_not_wait_for_prefill_coordinator(self):
        args = SimpleNamespace()
        events = []
        with patch.object(
            s, "invoke_multinode",
            side_effect=lambda *_args, node_id=None, **_kwargs:
                events.append(node_id) or 0,
        ), patch.object(s, "wait_model_endpoint") as wait:
            result = s.start_worker_set(
                args, Path("/results/run/multinode-v3.json"),
                ["d0", "d1"], "node-d", {"nodes": []},
            )
        self.assertEqual(result, 0)
        self.assertEqual(events, ["d0", "d1"])
        wait.assert_not_called()

    def test_single_prefill_group_waits_for_coordinator_readiness(self):
        args = SimpleNamespace()
        cfg = {
            "nodes": [
                {"node_id": "p0", "host_ip": "10.0.0.1", "port": 23900}
            ]
        }
        with patch.object(s, "invoke_multinode", return_value=0) as invoke, \
             patch.object(s, "wait_model_endpoint") as wait:
            result = s.start_worker_set(
                args, Path("/results/run/multinode-v3.json"),
                ["p0"], "node-p", cfg,
                establish_coordinator=True,
            )
        self.assertEqual(result, 0)
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(invoke.call_args.kwargs["node_id"], "p0")
        wait.assert_called_once_with(cfg["nodes"][0])

    def test_coordinator_readiness_failure_does_not_start_other_prefills(self):
        args = SimpleNamespace()
        started = []
        cfg = {
            "nodes": [
                {"node_id": "p0", "host_ip": "10.0.0.1", "port": 23900}
            ]
        }
        with patch.object(
            s, "invoke_multinode",
            side_effect=lambda *_args, node_id=None, **_kwargs:
                started.append(node_id) or 0,
        ), patch.object(
            s, "wait_model_endpoint", side_effect=RuntimeError("not ready")
        ):
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                s.start_worker_set(
                    args, Path("/results/run/multinode-v3.json"),
                    ["p0", "p1", "p2"], "node-p", cfg,
                    establish_coordinator=True,
                )
        self.assertEqual(started, ["p0"])


if __name__ == "__main__":
    unittest.main()
