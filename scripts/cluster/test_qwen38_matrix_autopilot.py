import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "qwen38_matrix_autopilot",
    ROOT / "qwen38_matrix_autopilot.py",
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Qwen38MatrixAutopilotTests(unittest.TestCase):
    def test_completed_result_accepts_terminal_failure_within_full_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workload = directory / "workload"
            workload.mkdir()
            (workload / "summary.json").write_text(
                json.dumps(
                    {
                        "requests": 500,
                        "completed": 499,
                        "failed": 1,
                        "measurement_elapsed_seconds": 100.0,
                    }
                )
            )
            with (directory / "episode_progress.jsonl").open("w") as stream:
                for index in range(500):
                    stream.write(
                        json.dumps(
                            {
                                "instance_id": str(index),
                                "completed_at": 1_000.0 + index,
                                "environment_error": index == 499,
                                "reward": 0,
                            }
                        )
                        + "\n"
                    )

            with patch.object(
                a, "t450_decode_metrics", return_value=(None, None, None)
            ):
                result = a.completed_result(64, "agentic", directory)

            self.assertIsNotNone(result)
            self.assertEqual(result["completed"], 499)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["environment_errors"], 1)

    def test_baseline_dynamic_l2norm_shape_is_not_compile_key(self):
        source = (
            a.BASELINE_ENGINE
            / "python/sglang/srt/layers/attention/fla/l2norm.py"
        ).read_text()
        self.assertNotIn("NB: tl.constexpr", source)
        self.assertNotRegex(source, r"(?m)^\s+T:\s*tl\.constexpr,")
        self.assertIn('@triton.jit(do_not_specialize=["T"])', source)

    def test_c256_hicache_uses_sufficient_mamba_host_capacity(self):
        experiment = {
            "concurrency": 256,
            "method": "hicache",
            "run_id": a.run_name(256, "hicache"),
        }
        with patch.object(a, "command") as command:
            a.prepare_and_start(experiment)

        prepare = [str(value) for value in command.call_args_list[0].args[0]]
        self.assertEqual(prepare[prepare.index("--mamba-full-memory-ratio") + 1], "0.9")
        self.assertEqual(prepare[prepare.index("--mamba-hicache-size-gb") + 1], "32")
        self.assertEqual(prepare[prepare.index("--prefill-hicache-size-gb") + 1], "32")
        self.assertEqual(prepare[prepare.index("--decode-hicache-size-gb") + 1], "32")
        self.assertTrue(experiment["run_id"].endswith("-r10"))

    def test_hicache_reclaims_host_before_deferring_internal_prefix(self):
        source = (a.MOONCAKE_COMPAT / "sitecustomize.py").read_text()
        reclaim = source.index("self.evict_host(int(kv_shortfall.item()))")
        retired_guard = source.index(
            "if not is_structurally_attached(node, self.get_child_key_fn):",
            reclaim,
        )
        admission = source.index("if not bool(torch.all(ready).item())")
        self.assertLess(reclaim, retired_guard)
        self.assertLess(retired_guard, admission)
        self.assertIn(
            "return _original_himamba_evict_device_leaf(self, node)",
            source[retired_guard:admission],
        )
        self.assertIn("ReduceOp.MAX", source[reclaim - 700 : reclaim])

    def test_hicache_protects_restore_sources_and_falls_back_on_stale_hit(self):
        source = (a.MOONCAKE_COMPAT / "sitecustomize.py").read_text()
        wrapper = source.index("def _load_back_with_host_source_protection")
        protect = source.index("self._protect_host_node", wrapper)
        original = source.index("_original_himamba_load_back", protect)
        retained = source.index('pending[node.id] = protected', original)
        fence_wrapper = source.index("def _loading_check_releasing_host_sources")
        fence = source.index("_original_himamba_loading_check(self)", fence_wrapper)
        release = source.index("self._release_host_node", fence)
        self.assertLess(protect, original)
        self.assertLess(original, retained)
        self.assertLess(retained, fence)
        self.assertLess(fence, release)
        self.assertIn("ReduceOp.MIN", source[wrapper:protect])
        self.assertIn("return None", source[wrapper:protect])


if __name__ == "__main__":
    unittest.main()
