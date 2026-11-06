import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "qwen35_9b_h100_matrix_autopilot",
    ROOT / "qwen35_9b_h100_matrix_autopilot.py",
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Qwen35H100AutopilotTests(unittest.TestCase):
    def test_dynamic_l2norm_token_dimension_is_not_a_triton_compile_key(self):
        relative = "python/sglang/srt/layers/attention/fla/l2norm.py"
        for tree in (a.PD_ENGINE, a.BASELINE_ENGINE):
            source = (tree / relative).read_text()
            self.assertNotIn("NB: tl.constexpr", source)
            self.assertNotRegex(source, r"(?m)^\s+T:\s*tl\.constexpr,")
            self.assertIn('@triton.jit(do_not_specialize=["T"])', source)

    def test_agentic_uses_stock_batch_invariant_matmul(self):
        relative = "python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py"
        source = (a.PD_ENGINE / relative).read_text()
        self.assertNotIn("_DEEPGEMM_FAILED_SHAPES", source)
        self.assertIn(
            "return _matmul_persistent_deepgemm(a=a, b=b, bias=bias)",
            source,
        )

    def test_baseline_uses_validated_hybrid_mooncake_compatibility_layer(self):
        source = (a.MOONCAKE_COMPAT / "sitecustomize.py").read_text()
        self.assertIn("def _register_mem_host_pool_v2", source)
        self.assertIn("def _batch_get_v2", source)
        self.assertIn("def _batch_set_v2", source)
        self.assertIn("DecodeKVCacheOffloadManager.__init__", source)

    def test_baseline_excludes_internal_nodes_from_host_leaf_eviction(self):
        source = (
            a.BASELINE_ENGINE
            / "python/sglang/srt/mem_cache/hi_mamba_radix_cache.py"
        ).read_text()
        self.assertIn("or len(node.children) > 0", source)

    def test_matrix_is_serial_and_locked_to_expected_order(self):
        self.assertEqual(len(a.ALL_EXPERIMENTS), 12)
        self.assertEqual(len(a.EXPERIMENTS), 9)
        self.assertEqual(
            [(row["concurrency"], row["method"]) for row in a.EXPERIMENTS],
            [
                (concurrency, method)
                for concurrency in (128, 192)
                for method in ("agentic", "colocated", "native")
            ] + [(concurrency, "hicache") for concurrency in (128, 192, 256)],
        )
        self.assertEqual(a.NODE, "node2800")
        self.assertEqual(a.JOB_ID, "25734937")
        self.assertGreater(a.HICACHE_ATTENTION_GB_PER_RANK * 1e9, 30.14 * 2**30)
        self.assertGreater(a.HICACHE_MAMBA_GB_PER_RANK * 1e9, 21.40 * 2**30)
        self.assertEqual(
            a.run_name(256, "agentic"),
            "q35-9b-h100-agentic-2p6d-swe500-miles-c256-20261010-r2",
        )
        self.assertIn("stock256", a.run_name(256, "hicache"))
        self.assertEqual(a.run_name(256, "hicache").rsplit("-", 1)[-1], "r22")
        self.assertEqual(a.run_name(128, "agentic").rsplit("-", 1)[-1], "r15")
        self.assertEqual(a.run_name(192, "agentic").rsplit("-", 1)[-1], "r16")
        self.assertEqual(a.run_name(128, "hicache").rsplit("-", 1)[-1], "r17")
        self.assertTrue(all(len(row["run_id"]) <= 64 for row in a.ALL_EXPERIMENTS))
        self.assertTrue(
            {
                23902, 23903, 23904,  # Agentic router/metrics/control.
                24002, 24003,         # Colocated router/metrics.
                24400, 24401,         # Native router/metrics.
                25103, 25151, 25152, 25180, 25190,  # Mooncake services.
            }.issubset(a.MATRIX_PORTS)
        )

    def test_hard_admission_gate_includes_apptainer_helpers(self):
        counts = {name: 0 for name in a.SANDBOX_HELPER_NAMES}
        counts["squashfuse_ll"] = 1
        with patch.object(a, "gpu_process_count", return_value=0), patch.object(
            a, "occupied_matrix_ports", return_value=[]
        ), patch.object(a, "sandbox_helper_counts", return_value=counts):
            with self.assertRaisesRegex(RuntimeError, "squashfuse_ll"):
                a.assert_clean_for_start()

    def test_dirty_post_run_gate_rebuilds_before_cleanup_returns(self):
        experiment = {"concurrency": 394, "method": "agentic", "run_id": "x"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            a, "RUNS", Path(temporary)
        ), patch.object(a, "scoped_process_count", return_value=0), patch.object(
            a, "workload_process_count", return_value=0
        ), patch.object(
            a,
            "sandbox_helper_counts",
            return_value={name: 0 for name in a.SANDBOX_HELPER_NAMES},
        ), patch.object(
            a, "wait_clean", side_effect=a.DirtyAllocationError("residue")
        ), patch.object(a, "rebuild_allocation") as rebuild:
            a.cleanup_experiment(experiment)

        rebuild.assert_called_once()
        self.assertIn("DirtyAllocationError", rebuild.call_args.args[0])

    def test_runtime_pressure_gate_stops_before_node_exhaustion(self):
        with patch.object(
            a,
            "node_resource_snapshot",
            return_value={
                "user_threads": a.MAX_USER_THREADS,
                "mem_available_kib": a.MIN_MEM_AVAILABLE_KIB * 2,
            },
        ):
            self.assertIn("thread safety limit", a.runtime_pressure_issue())
        with patch.object(
            a,
            "node_resource_snapshot",
            return_value={
                "user_threads": 100,
                "mem_available_kib": a.MIN_MEM_AVAILABLE_KIB,
            },
        ):
            self.assertIn("available-memory safety limit", a.runtime_pressure_issue())

        with patch.object(
            a,
            "node_resource_snapshot",
            return_value={
                "user_threads": 100,
                "mem_available_kib": a.MIN_MEM_AVAILABLE_KIB * 2,
                "commit_ratio": 1.25,
            },
        ):
            self.assertEqual(a.runtime_pressure_issue(), "")

    def test_runtime_launch_transports_use_validated_ssh_path(self):
        with patch.object(a, "command") as command:
            a.prepare_and_start(
                {"concurrency": 128, "method": "agentic", "run_id": "ssh-agentic"}
            )
            rendered = [str(value) for call in command.call_args_list for value in call.args[0]]
            self.assertIn("ssh", rendered)
            self.assertNotIn("srun", rendered)

        with patch.object(a, "command") as command:
            a.prepare_and_start(
                {"concurrency": 256, "method": "hicache", "run_id": "fixed-evict"}
            )
            rendered = [str(value) for call in command.call_args_list for value in call.args[0]]
            self.assertIn(str(a.FIXED_MOONCAKE_MASTER), rendered)
            self.assertIn("--disable-mlir-multithreading", rendered)

        with patch.object(a, "command") as command:
            a.prepare_and_start(
                {"concurrency": 128, "method": "native", "run_id": "ssh-native"}
            )
            rendered = [str(value) for call in command.call_args_list for value in call.args[0]]
            self.assertIn("ssh", rendered)
            self.assertNotIn("srun", rendered)

    def test_runtime_health_detects_a_single_dead_worker(self):
        experiment = {"concurrency": 256, "method": "hicache", "run_id": "x"}
        with patch.object(a, "gpu_process_count", return_value=7), patch.object(
            a, "occupied_matrix_ports", return_value=list(a.MATRIX_PORTS)
        ):
            self.assertEqual(
                a.runtime_health_issue(experiment),
                "expected 8 GPU processes, found 7",
            )

    def test_runtime_health_detects_a_missing_worker_port(self):
        experiment = {"concurrency": 256, "method": "hicache", "run_id": "x"}
        ports = set(a.MATRIX_PORTS) - {24210}
        with patch.object(a, "gpu_process_count", return_value=8), patch.object(
            a, "occupied_matrix_ports", return_value=sorted(ports)
        ):
            self.assertEqual(
                a.runtime_health_issue(experiment),
                "missing service ports 24210",
            )

    def test_runtime_health_detects_a_dead_mooncake_master(self):
        experiment = {"concurrency": 256, "method": "hicache", "run_id": "x"}
        ports = set(a.MATRIX_PORTS) - {25151}
        with patch.object(a, "gpu_process_count", return_value=8), patch.object(
            a, "occupied_matrix_ports", return_value=sorted(ports)
        ):
            self.assertEqual(
                a.runtime_health_issue(experiment),
                "missing service ports 25151",
            )

    def test_t450_uses_workload_schedule_origin_for_clean_run(self):
        experiment = {
            "concurrency": 256,
            "method": "agentic",
            "run_id": "timing-test",
        }
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            a, "RUNS", Path(temporary)
        ):
            directory = a.run_dir(experiment)
            workload = directory / "workload"
            workload.mkdir(parents=True)
            start = 1_700_000_000.0
            progress = [
                {
                    "instance_id": f"case-{index}",
                    "completed_at": start + index + 1,
                    "reward": index % 2,
                    "environment_error": None,
                }
                for index in range(500)
            ]
            (directory / "episode_progress.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in progress)
            )
            (workload / "requests.jsonl").write_text(
                "".join(
                    json.dumps({"scheduled_ts": start}) + "\n"
                    for _ in range(500)
                )
            )
            (workload / "summary.json").write_text(
                json.dumps(
                    {
                        "requests": 500,
                        "completed": 500,
                        "failed": 0,
                        # Deliberately different: engine sampling is not the
                        # workload/T450 clock origin.
                        "measurement_elapsed_seconds": 499.125,
                    }
                )
            )

            result = a.completed_result(experiment)

        self.assertIsNotNone(result)
        self.assertEqual(result["normal_completed"], 500)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["t50_seconds"], 50)
        self.assertEqual(result["t250_seconds"], 250)
        self.assertEqual(result["t450_seconds"], 450)
        self.assertEqual(result["t500_seconds"], 500)
        self.assertIsNone(result["t450_decode_tps"])

    def test_t450_decode_is_interpolated_and_summed_per_endpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workload = directory / "workload"
            workload.mkdir()
            rows = []
            for timestamp, first, second in ((100.0, 10.0, 20.0), (200.0, 1010.0, 2020.0)):
                rows.append(
                    {
                        "ts": timestamp,
                        "role": "decode",
                        "engine_count": 2,
                        "endpoint_metrics": [
                            {
                                "endpoint": "http://d0",
                                "metrics": {a.DECODE_COUNTER: first},
                            },
                            {
                                "endpoint": "http://d1",
                                "metrics": {a.DECODE_COUNTER: second},
                            },
                        ],
                    }
                )
            (workload / "engine_metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            tokens, throughput = a.t450_decode_metrics(directory, 125.0, 175.0)

        self.assertAlmostEqual(tokens, 1500.0)
        self.assertAlmostEqual(throughput, 30.0)

    def test_completed_result_counts_terminal_environment_failure(self):
        experiment = {"concurrency": 256, "method": "hicache", "run_id": "bad"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            a, "RUNS", Path(temporary)
        ):
            directory = a.run_dir(experiment)
            workload = directory / "workload"
            workload.mkdir(parents=True)
            rows = [
                {
                    "instance_id": f"case-{index}",
                    "completed_at": 1_700_000_000.0 + index,
                    "environment_error": "fork EAGAIN" if index == 7 else None,
                }
                for index in range(500)
            ]
            (directory / "episode_progress.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (workload / "requests.jsonl").write_text(
                "".join(
                    json.dumps({"scheduled_ts": 1_700_000_000.0}) + "\n"
                    for _ in range(500)
                )
            )
            (workload / "summary.json").write_text(
                json.dumps({"requests": 500, "completed": 499, "failed": 1})
            )

            result = a.completed_result(experiment)

        self.assertIsNotNone(result)
        self.assertEqual(result["normal_completed"], 499)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["environment_errors"], 1)

    def test_completed_result_rejects_explicitly_invalid_run(self):
        experiment = {"concurrency": 256, "method": "hicache", "run_id": "bad"}
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            a, "RUNS", Path(temporary)
        ):
            directory = a.run_dir(experiment)
            directory.mkdir(parents=True)
            (directory / "INVALID.md").write_text("runtime health failure\n")
            self.assertIsNone(a.completed_result(experiment))


if __name__ == "__main__":
    unittest.main()
