import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import browsecomp_qwen3_8b_dram_d64p128_h100 as run
import browsecomp_qwen3_8b_dram_d64p128_watchdog as monitor


class ConfigTests(unittest.TestCase):
    def test_only_requested_capacity_and_output_paths_change(self):
        old = json.loads(run.PRIOR.read_text())
        env, command = run.case_environment(run.CASE, run.run_dir(run.CASE))
        changed = {key for key in set(old['env']) | set(env) if old['env'].get(key) != env.get(key)}
        self.assertEqual(changed, {'RUN_DIR', 'PD_DRAM_TRACE_DIR', 'D2P_HOST_ARENA_GIB_PER_P',
                                  'P2D_HOST_ARENA_GIB_PER_P', 'SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB',
                                  'SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB'})
        self.assertEqual(command, old['command'])
        self.assertEqual(env['D2P_HOST_ARENA_GIB_PER_P'], '64')
        self.assertEqual(env['P2D_HOST_ARENA_GIB_PER_P'], '128')
        self.assertEqual(env['MAX_INFLIGHT'], '512')
        self.assertEqual(env['WARMUP_SECONDS'], '300')
        self.assertEqual(env['MEASURE_SECONDS'], '1200')

    def test_frozen_engine_matches_original(self):
        run.validate_inputs()

    def test_state_is_separate_and_single_case(self):
        self.assertNotEqual(run.CONTROL, run.SOURCE_CONTROL)
        self.assertEqual(run.dram.base.CASES, [run.CASE])
        self.assertEqual(run.dram.base.CONTROL, run.CONTROL)

    def test_preserve_existing_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            original = 'All five earlier results.\n'
            path.write_text(original)
            with patch.object(run, 'REPORT', path):
                run.render_report({})
                first = path.read_text()
                run.render_report({})
            self.assertEqual(first, path.read_text())
            self.assertTrue(first.startswith(original))
            self.assertEqual(first.count(run.START), 1)

    def test_partial_marker_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            original = 'Existing results.\n' + run.START
            path.write_text(original)
            with patch.object(run, 'REPORT', path):
                with self.assertRaisesRegex(RuntimeError, 'incomplete managed'):
                    run.render_report({})
            self.assertEqual(path.read_text(), original)

    def test_cleanup_rejects_gpu_leftovers(self):
        with patch.object(run.dram.base, 'wait_ports_free'), \
                patch.object(run.dram.base, 'gpu_processes', return_value=['live']):
            with self.assertRaisesRegex(RuntimeError, 'GPU cleanup gate failed'):
                run.dram.cleanup_gate(run.CASE)

    def test_monitor_uses_readonly_ssh_without_creating_job_step(self):
        from types import SimpleNamespace
        with patch.object(monitor.watch, 'allocation', return_value=('RUNNING', 'node1701')), \
                patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout='{"gpu_processes":8}')) as call:
            self.assertEqual(monitor.ssh_resource_probe('27105768', 'node1701')['gpu_processes'], 8)
        self.assertEqual(call.call_args.args[0][0], 'ssh')
        self.assertEqual(call.call_args.args[0][-2], 'node1701')

    def test_monitor_refuses_unallocated_target(self):
        with patch.object(monitor.watch, 'allocation', return_value=('RUNNING', 'other-node')), \
                patch.object(monitor.subprocess, 'run') as call:
            with self.assertRaisesRegex(RuntimeError, 'no longer allocated'):
                monitor.ssh_resource_probe('27105768', 'node1701')
        call.assert_not_called()

    def test_monitor_detects_quarantine_even_if_logs_and_gpu_are_alive(self):
        from types import SimpleNamespace
        payload = json.dumps(dict(gpu_processes=8, transport_errors=['prefill-0.log: quarantine']))
        with patch.object(monitor.watch, 'allocation', return_value=('RUNNING', 'node1701')), \
                patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout=payload)):
            with self.assertRaisesRegex(RuntimeError, 'transport failure'):
                monitor.ssh_resource_probe('job', 'node1701')

    def test_monitor_detects_no_token_progress(self):
        from types import SimpleNamespace
        payload = json.dumps(dict(gpu_processes=8, decode_tail_span_seconds=100, decode_tail_tokens=0))
        with patch.object(monitor.watch, 'allocation', return_value=('RUNNING', 'node1701')), \
                patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout=payload)):
            with self.assertRaisesRegex(RuntimeError, 'no Decode counter growth'):
                monitor.ssh_resource_probe('job', 'node1701')

    def test_probe_timeout_message_does_not_dump_probe_program(self):
        import subprocess
        with patch.object(monitor.watch, 'allocation', return_value=('RUNNING', 'node1701')), \
                patch.object(monitor.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['secret-long-code'], 40)):
            with self.assertRaisesRegex(RuntimeError, '^SSH resource probe timed out after 40 seconds on node1701$'):
                monitor.ssh_resource_probe('job', 'node1701')

    def test_completed_window_does_not_require_live_engine_counters(self):
        from types import SimpleNamespace
        payload = json.dumps(dict(gpu_processes=0, measurement_complete=True,
                                  decode_counter_age_seconds=200, decode_tail_span_seconds=100,
                                  decode_tail_tokens=0))
        with patch.object(monitor.watch, 'allocation', return_value=('RUNNING', 'node1701')), \
                patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout=payload)):
            self.assertTrue(monitor.ssh_resource_probe('job', 'node1701')['measurement_complete'])


if __name__ == '__main__':
    unittest.main()
