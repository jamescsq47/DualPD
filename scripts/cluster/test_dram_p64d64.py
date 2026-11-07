import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import browsecomp_qwen3_8b_dram_p64d64_h100 as followup
import browsecomp_qwen3_8b_dram_p64d64_watchdog as monitor


class DramP64D64Tests(unittest.TestCase):
    def test_only_host_p_capacity_differs_from_completed_native_case(self):
        env, command = followup.case_environment(followup.CASE, followup.run_dir(followup.CASE))
        prior = followup.SOURCE_CONTROL / "browsecomp8b-hicache-full-c512-w300-m1200-r2/launch_config.json"
        old = json.loads(prior.read_text())["env"]
        for key in ["MODEL_PATH", "MAX_INFLIGHT", "CLOSED_LOOP", "REQUESTS", "SCHEDULE_FILE",
                    "WARMUP_SECONDS", "MAX_WARMUP_SECONDS", "MEASURE_SECONDS", "PAGE_SIZE",
                    "PREFILL_GPUS", "DECODE_GPUS", "PREFILL_TP_SIZE", "DECODE_TP_SIZE",
                    "PREFILL_MEM_FRACTION_STATICS", "DECODE_MEM_FRACTION_STATICS",
                    "P_HICACHE_WRITE_POLICY", "CASE_MODE", "TP1_LOCAL_NUMA", "PYTHONPATH"]:
            self.assertEqual(env.get(key), old.get(key), key)
        self.assertEqual(env["P_HICACHE_SIZE"], "64")
        self.assertEqual(env["D_HICACHE_SIZE"], "64")
        self.assertEqual(command, ["bash", str(followup.dram.base.PD / "scripts/baseline/run_pd_case.sh")])
        self.assertEqual(env["MAX_INFLIGHT"], "512")
        self.assertEqual(env["PD_DRAM_SAMPLE_SECONDS"], "2")
        self.assertNotIn("HF_TOKEN", env)

    def test_native_capacity_strictly_exceeds_gpu_pool(self):
        tokens = (int(64 * 1e9 // 147456) // 64 + 1) * 64
        self.assertGreater(tokens, 345920)

    def test_isolated_state_and_only_one_case(self):
        self.assertNotEqual(followup.CONTROL, followup.SOURCE_CONTROL)
        self.assertEqual(followup.dram.base.CASES, [followup.CASE])
        self.assertEqual(followup.dram.base.CONTROL, followup.CONTROL)
        self.assertEqual(followup.dram.run_dir(followup.CASE), followup.run_dir(followup.CASE))

    def test_append_replace_preserves_original_report(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            original = "Original four completed cases and verification.\n"
            report.write_text(original)
            with patch.object(followup, "REPORT", report):
                followup.render_report({})
                first = report.read_text()
                followup.render_report({})
            final = report.read_text()
            self.assertTrue(final.startswith(original))
            self.assertEqual(first, final)
            self.assertEqual(final.count(followup.START), 1)
            self.assertIn("P64/D64", final)

    def test_partial_report_marker_fails_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.md"
            original = "Original four.\n" + followup.START
            report.write_text(original)
            with patch.object(followup, "REPORT", report):
                with self.assertRaisesRegex(RuntimeError, "incomplete managed"):
                    followup.render_report({})
            self.assertEqual(report.read_text(), original)

    def test_cleanup_fails_closed_on_gpu_or_helpers(self):
        with patch.object(followup.dram.base, "wait_ports_free"), \
                patch.object(followup.dram.base, "gpu_processes", return_value=["leftover"]):
            with self.assertRaisesRegex(RuntimeError, "GPU cleanup gate failed"):
                followup.dram.cleanup_gate(followup.CASE)

    def test_borrowed_two_node_allocation_is_pinned_to_selected_node(self):
        with patch.object(monitor, "original_allocation", return_value=("RUNNING", "node[1700-1701]")), \
                patch.dict(monitor.os.environ, {"PD_DRAM_WATCH_NODE": "node1701"}), \
                patch.object(monitor.subprocess, "check_output", return_value="node1700\nnode1701\n"):
            self.assertEqual(monitor.allocation("25647805"), ("RUNNING", "node1701"))

    def test_selected_node_must_belong_to_borrowed_allocation(self):
        with patch.object(monitor, "original_allocation", return_value=("RUNNING", "node1700")), \
                patch.dict(monitor.os.environ, {"PD_DRAM_WATCH_NODE": "node1701"}), \
                patch.object(monitor.subprocess, "check_output", return_value="node1700\n"):
            self.assertEqual(monitor.allocation("job"), ("NODE_NOT_IN_ALLOCATION", "node1701"))

    def test_unselected_allocation_keeps_original_behavior(self):
        with patch.object(monitor, "original_allocation", return_value=("RUNNING", "node2700")), \
                patch.dict(monitor.os.environ, {}, clear=True), \
                patch.object(monitor.subprocess, "check_output") as probe:
            self.assertEqual(monitor.allocation("job"), ("RUNNING", "node2700"))
            probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
