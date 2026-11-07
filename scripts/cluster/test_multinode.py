import copy
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("multinode", ROOT / "multinode.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MultiNodeV2PlanTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "multinode.example.json").read_text())

    def validate(self, cfg=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg or self.cfg))
            return m.load_config(path)

    def test_tp8_plan_is_cpu_only_and_has_tcp_relay(self):
        cfg = self.validate()
        plan = m.plan(cfg)
        self.assertEqual(plan["control"]["environment"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(any("agentic_group_protocol" in arg for arg in plan["control"]["command"]))
        self.assertIn("--links", plan["control"]["command"])
        self.assertEqual(len(plan["workers"]), 2)
        self.assertEqual(plan["workers"][1]["environment"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertEqual(
            plan["workers"][1]["environment"][
                "SGLANG_AGENTIC_MULTINODE_DECODE_GROWTH_TOKENS"
            ],
            "512",
        )

    def test_host_prepare_profile_is_explicit_and_off_by_default(self):
        key = "SGLANG_AGENTIC_PROFILE_HOST_PREPARE"
        for worker in m.plan(self.validate())["workers"]:
            self.assertEqual(worker["environment"][key], "0")
        self.cfg["profile_host_prepare"] = True
        for worker in m.plan(self.validate())["workers"]:
            self.assertEqual(worker["environment"][key], "1")

    def test_compact_handoff_payload_is_explicit_for_all_workers(self):
        key = "SGLANG_AGENTIC_COMPACT_HANDOFF_PAYLOAD"
        for worker in m.plan(self.validate())["workers"]:
            self.assertEqual(worker["environment"][key], "0")
        self.cfg["compact_handoff_payload"] = True
        for worker in m.plan(self.validate())["workers"]:
            self.assertEqual(worker["environment"][key], "1")

    def test_no_shared_control_or_runtime_ledger_environment(self):
        result = m.plan(self.validate())
        legacy_v1_keys = {
            "SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY",
            "SGLANG_AGENTIC_KV_D_HOSTLESS",
            "SGLANG_AGENTIC_KV_HOST_STAGING",
            "SGLANG_AGENTIC_KV_P2D_HOST_STAGING",
            "SGLANG_AGENTIC_KV_RELAY_ENABLED",
            "SGLANG_PD_P_READY_BACKPRESSURE_MODE",
            "SGLANG_PD_P_READY_REQUEST_CAP",
            "SGLANG_PD_LATE_BIND_GLOBAL_DECODE",
        }
        for worker in result["workers"]:
            env = worker["environment"]
            self.assertFalse(any("CONTROL_ROOT" in key for key in env))
            self.assertFalse(any("LEDGER_PATH" in key for key in env))
            self.assertFalse(any("P_READY_DIR" in key for key in env))
            self.assertFalse(any("METADATA_DIR" in key for key in env))
            self.assertTrue(legacy_v1_keys.isdisjoint(env))
            self.assertNotIn("SGLANG_AGENTIC_GROUP_RANK", env)
        self.assertEqual(result["router"]["command"][0], self.cfg["python"])
        self.assertEqual(result["router"]["command"][1:3], [
            "-m", "sglang.srt.disaggregation.dualpd.router.global_pd_router"])
        self.assertIn("--mini-lb", result["router"]["command"])
        self.assertNotIn(
            "launch_late_binding_router.py", result["router"]["command"]
        )
        self.assertEqual(
            result["router"]["environment"][
                "DUALPD_ROUTER_DECODE_RESERVATION_SECONDS"
            ],
            "3600.0",
        )
        router_env = result["router"]["environment"]
        self.assertEqual(
            {
                key: router_env[key]
                for key in (
                    "SGLANG_AGENTIC_GROUP_ENDPOINT",
                    "SGLANG_AGENTIC_GROUP_RUN_ID",
                    "SGLANG_AGENTIC_GROUP_GROUP_ID",
                    "SGLANG_AGENTIC_GROUP_TOKEN",
                )
            },
            {
                "SGLANG_AGENTIC_GROUP_ENDPOINT": "tcp://28.49.24.105:62080",
                "SGLANG_AGENTIC_GROUP_RUN_ID": "qwen3-32b-tp8-smoke-CHANGE-ME",
                "SGLANG_AGENTIC_GROUP_GROUP_ID": "prefill-0--decode-0",
                "SGLANG_AGENTIC_GROUP_TOKEN": "REPLACE-WITH-RANDOM-SECRET",
            },
        )
        for worker in result["workers"]:
            self.assertTrue(
                worker["environment"][
                    "SGLANG_AGENTIC_DECODE_RESERVATION_CALLBACK_URL"
                ].endswith("/dualpd/decode_materialized")
            )
            self.assertTrue(
                worker["environment"]["PD_AGENTIC_FINAL_CALLBACK_URL"].endswith(
                    "/dualpd/application_final"
                )
            )
            self.assertEqual(
                worker["environment"][
                    "SGLANG_AGENTIC_MULTINODE_D2P_SHARED_NETWORK_LANES"
                ],
                worker["environment"][
                    "SGLANG_AGENTIC_MULTINODE_SHARED_NETWORK_LANES"
                ],
            )
            self.assertEqual(
                worker["environment"][
                    "SGLANG_AGENTIC_MULTINODE_P2D_SHARED_NETWORK_LANES"
                ],
                worker["environment"][
                    "SGLANG_AGENTIC_MULTINODE_SHARED_NETWORK_LANES"
                ],
            )

    def test_same_parent_env_yields_eight_explicit_rank_identities(self):
        worker = m.worker_plan(self.validate(), self.cfg["nodes"][0])
        rank_envs = worker["rank_environments"]
        self.assertEqual([entry["SGLANG_AGENTIC_GROUP_RANK"] for entry in rank_envs], list(map(str, range(8))))
        for entry in rank_envs:
            self.assertEqual(entry["SGLANG_AGENTIC_GROUP_GROUP_ID"], "prefill-0--decode-0")
            self.assertEqual(entry["SGLANG_AGENTIC_GROUP_ENDPOINT_GROUP"], "prefill-0")
            self.assertEqual(entry["SGLANG_AGENTIC_GROUP_ENDPOINT_ROLE"], "prefill")

    def test_worker_native_and_jit_thread_pools_are_bounded(self):
        env = m.worker_plan(self.validate(), self.cfg["nodes"][0])["environment"]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TORCHINDUCTOR_COMPILE_THREADS",
        ):
            self.assertEqual(env[name], "1")
        self.assertEqual(env["SGLANG_STACK_SOFT_LIMIT_KIB"], "8192")

    def test_component_open_file_limit_is_raised_without_exceeding_hard_limit(self):
        calls = []
        original_getrlimit = m.resource.getrlimit
        original_setrlimit = m.resource.setrlimit
        try:
            m.resource.getrlimit = lambda kind: (1024, 32768)
            m.resource.setrlimit = lambda kind, limits: calls.append((kind, limits))
            self.assertEqual(m.raise_nofile_soft_limit(), 32768)
        finally:
            m.resource.getrlimit = original_getrlimit
            m.resource.setrlimit = original_setrlimit
        self.assertEqual(calls, [(m.resource.RLIMIT_NOFILE, (32768, 32768))])

    def test_link_contains_both_endpoint_groups_atomically(self):
        links = m.control_links(self.validate())
        spec = links["prefill-0--decode-0"]
        self.assertEqual(spec["coordinator"], {"endpoint_group": "prefill-0", "rank": 0})
        self.assertEqual({e["endpoint_group"] for e in spec["endpoints"]}, {"prefill-0", "decode-0"})
        self.assertEqual(sum(e["size"] for e in spec["endpoints"]), 16)

    def test_control_token_may_begin_with_dash(self):
        cfg = self.validate()
        cfg["group_control"]["token"] = "-leading-token-is-valid"
        command = m.control_plan(cfg)["command"]
        self.assertIn("--token=-leading-token-is-valid", command)

    def test_local_root_is_only_output_location(self):
        cfg = self.validate()
        plan = m.plan(cfg)
        for component in [plan["control"], plan["router"], *plan["workers"]]:
            self.assertTrue(component["local_run_dir"].startswith(cfg["local_root"] + "/"))
            self.assertFalse(any(cfg["local_root"] in value for value in component["environment"].values()))

    def test_legacy_shared_mode_rejected_and_old_control_root_ignored(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["control_root"] = "/some/shared/path"
        loaded = self.validate(cfg)
        self.assertNotIn("control_root", m.common_env(loaded))
        self.assertEqual(
            m.common_env(loaded)["SGLANG_FLASHINFER_ALLREDUCE_MAX_TOKENS"],
            "2048",
        )
        cfg["legacy_shared_control"] = True
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.validate(cfg)

    def test_group_control_required_and_port_collision_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        del cfg["group_control"]
        with self.assertRaisesRegex(ValueError, "group_control"):
            self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["group_control"]["port"] = cfg["nodes"][0]["port"]
        with self.assertRaisesRegex(ValueError, "collides"):
            self.validate(cfg)

    def test_tp1_tp2_tp4_command_generation(self):
        for tp in (1, 2, 4):
            cfg = copy.deepcopy(self.cfg)
            cfg["tp_size"] = tp
            for node in cfg["nodes"]:
                node["gpus"] = list(range(tp))
            loaded = self.validate(cfg)
            self.assertEqual(len(m.plan(loaded)["workers"][0]["rank_environments"]), tp)

    def test_qwen35_plan_restores_hybrid_runtime_contract(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(
            {
                "model_family": "qwen35_moe",
                "prefill_mamba_full_memory_ratio": 0.75,
                "decode_mamba_full_memory_ratio": 0.5,
                "direct_lanes": 8,
                "d2p_direct_lanes": 4,
                "p2d_direct_lanes": 8,
                "host_lanes": 6,
                "shared_network_lanes": 24,
                "d2p_shared_network_lanes": 16,
                "p2d_shared_network_lanes": 20,
                "d2p_direct_network_reserve": 4,
                "d2p_host_network_reserve": 8,
                "nixl_progress_threads": 16,
                "forward_stream_priority": -1,
                "p2d_direct_network_reserve": 2,
                "p2d_late_bind_grace_seconds": 2.5,
                "seed": 2026,
            }
        )
        result = m.plan(self.validate(cfg))
        for node, worker in zip(cfg["nodes"], result["workers"]):
            env = worker["environment"]
            command = worker["command"]
            self.assertEqual(env["SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID"], "1")
            self.assertEqual(env["SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED"], "true")
            self.assertEqual(env["SGLANG_AGENTIC_MULTINODE_DIRECT_LANES"], "8")
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_D2P_DIRECT_LANES"], "4"
            )
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_P2D_DIRECT_LANES"], "8"
            )
            self.assertEqual(env["SGLANG_AGENTIC_MULTINODE_HOST_LANES"], "6")
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_SHARED_NETWORK_LANES"], "24"
            )
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_D2P_SHARED_NETWORK_LANES"], "16"
            )
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_P2D_SHARED_NETWORK_LANES"], "20"
            )
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_D2P_DIRECT_NETWORK_RESERVE"], "4"
            )
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_D2P_HOST_NETWORK_RESERVE"], "8"
            )
            self.assertEqual(env["SGLANG_AGENTIC_NIXL_PROGRESS_THREADS"], "16")
            self.assertEqual(env["SGLANG_AGENTIC_FORWARD_STREAM_PRIORITY"], "-1")
            self.assertEqual(
                env["SGLANG_AGENTIC_MULTINODE_P2D_DIRECT_NETWORK_RESERVE"], "2"
            )
            self.assertEqual(
                env["SGLANG_AGENTIC_P2D_LATE_BIND_GRACE_SECONDS"], "2.5"
            )
            self.assertIn("--mamba-full-memory-ratio", command)
            expected = "0.75" if node["role"] == "prefill" else "0.5"
            self.assertEqual(command[command.index("--mamba-full-memory-ratio") + 1], expected)
            self.assertIn("--tool-call-parser", command)
            self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_coder")
            self.assertEqual(
                env["PATH"].split(m.os.pathsep)[0], str(Path(cfg["python"]).parent)
            )

    def test_minimax_m2_plan_uses_tp8_ep8_without_mamba_state(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(
            {
                "model_family": "minimax_m2",
                "ep_size": 8,
                "reasoning_parser": "minimax-append-think",
                "tool_call_parser": "minimax-m2",
            }
        )
        result = m.plan(self.validate(cfg))
        for worker in result["workers"]:
            env = worker["environment"]
            command = worker["command"]
            self.assertNotIn("SGLANG_AGENTIC_MULTINODE_QWEN35_HYBRID", env)
            self.assertNotIn("SGLANG_AGENTIC_KV_MAMBA_REQUEST_OWNED", env)
            self.assertEqual(command[command.index("--tp-size") + 1], "8")
            self.assertEqual(command[command.index("--ep-size") + 1], "8")
            self.assertEqual(
                command[command.index("--reasoning-parser") + 1],
                "minimax-append-think",
            )
            self.assertEqual(
                command[command.index("--tool-call-parser") + 1],
                "minimax-m2",
            )
            self.assertNotIn("--mamba-full-memory-ratio", command)
            self.assertNotIn("--quantization", command)

    def test_minimax_m2_rejects_non_ep8_topology(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update({"model_family": "minimax_m2", "ep_size": 1})
        with self.assertRaisesRegex(ValueError, "TP=8 and EP=8"):
            self.validate(cfg)

    def test_worker_plan_honors_per_group_directional_host_capacity(self):
        cfg = copy.deepcopy(self.cfg)
        prefill = next(node for node in cfg["nodes"] if node["role"] == "prefill")
        decode = next(node for node in cfg["nodes"] if node["role"] == "decode")
        prefill["p2d_host_gib_per_rank"] = 17
        decode["d2p_host_gib_per_rank"] = 19
        loaded = self.validate(cfg)

        p_plan = m.worker_plan(loaded, prefill)
        d_plan = m.worker_plan(loaded, decode)
        self.assertEqual(
            p_plan["environment"]["SGLANG_AGENTIC_MULTINODE_P2D_HOST_GIB"],
            "17",
        )
        self.assertEqual(
            d_plan["environment"]["SGLANG_AGENTIC_MULTINODE_D2P_HOST_GIB"],
            "19",
        )
        self.assertEqual(
            p_plan["host_capacity_gib_per_direction"]["p2d_source"],
            17 * cfg["tp_size"],
        )
        self.assertEqual(
            d_plan["host_capacity_gib_per_direction"]["d2p_source"],
            19 * cfg["tp_size"],
        )

    def test_site_compiler_cache_and_workload_environment_are_explicit(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(
            {
                "local_triton_cache": True,
                "python": "/env/bin/python",
                "system_bin": "/usr/bin",
                "cc": "/usr/bin/gcc",
                "cxx": "/usr/bin/g++",
                "nvcc_ccbin": "/usr/bin/g++",
                "workload_command": ["/env/bin/python", "workload.py"],
                "workload_environment": {
                    "PD_SWE_APPTAINER_IMAGE_DIR": "/images",
                    "PD_SWE_APPTAINER_WORK_ROOT": "/tmp/user/swe",
                },
            }
        )
        loaded = self.validate(cfg)
        worker = m.worker_plan(loaded, loaded["nodes"][0])
        env = worker["environment"]
        self.assertEqual(env["CC"], "/usr/bin/gcc")
        self.assertEqual(env["CXX"], "/usr/bin/g++")
        self.assertEqual(env["NVCC_CCBIN"], "/usr/bin/g++")
        self.assertEqual(env["PATH"].split(m.os.pathsep)[:2], ["/env/bin", "/usr/bin"])
        self.assertTrue(env["FLASHINFER_WORKSPACE_BASE"].endswith("/flashinfer"))
        workload = m.workload_plan(loaded)
        self.assertEqual(workload["environment"]["PD_SWE_APPTAINER_IMAGE_DIR"], "/images")
        self.assertEqual(
            workload["environment"]["PD_AGENTIC_FINAL_CALLBACK_URL"],
            "http://28.49.24.105:30002/dualpd/application_final",
        )

        cfg["workload_environment"]["UNSAFE_ENV"] = "value"
        with self.assertRaisesRegex(ValueError, "unsupported workload environment"):
            self.validate(cfg)

    def test_unknown_model_family_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model_family"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported multi-node model family"):
            self.validate(cfg)

    def test_runtime_scrubs_another_experiment_environment(self):
        worker = m.worker_plan(self.validate(), self.cfg["nodes"][0])
        with patch.dict(m.os.environ, {"SGLANG_AGENTIC_KV_LEDGER_PATH": "/old", "HOST_IP": "127.0.0.1"}):
            env = m.runtime_env(worker)
        self.assertNotIn("SGLANG_AGENTIC_KV_LEDGER_PATH", env)
        self.assertNotIn("HOST_IP", env)
        self.assertEqual(env["SGLANG_AGENTIC_GROUP_ENDPOINT"], "tcp://28.49.24.105:62080")

    def test_listening_port_conflict_is_fail_closed(self):
        cfg = self.validate()
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            cfg["nodes"][0]["port"] = occupied.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                m.check_listen_ports(cfg, m.worker_plan(cfg, cfg["nodes"][0]), "worker")

    def test_protocol_check_never_launches_on_import_failure(self):
        cfg = self.validate()
        result = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "missing"})()
        with patch.object(m.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                m.protocol_check(cfg, {})


if __name__ == "__main__":
    unittest.main()
