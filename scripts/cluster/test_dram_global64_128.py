import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import browsecomp_qwen3_8b_dram_global64_128_h100 as run


class GlobalCapacityTests(unittest.TestCase):
    def test_global_not_per_worker(self):
        env, command = run.case_environment(run.CASE, run.run_dir(run.CASE))
        count = len(env['PREFILL_GPUS'].split())
        self.assertEqual(count * int(env['D2P_HOST_ARENA_GIB_PER_P']), 64)
        self.assertEqual(count * int(env['P2D_HOST_ARENA_GIB_PER_P']), 128)
        prior = json.loads(run.prior.PRIOR.read_text())
        changed = {k for k in set(prior['env']) | set(env) if prior['env'].get(k) != env.get(k)}
        self.assertEqual(changed, {'RUN_DIR','PD_DRAM_TRACE_DIR','D2P_HOST_ARENA_GIB_PER_P',
                                  'P2D_HOST_ARENA_GIB_PER_P','SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB',
                                  'SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB'})
        self.assertEqual(command, prior['command'])

    def test_reject_previous_mistake(self):
        env, _ = run.case_environment(run.CASE, run.run_dir(run.CASE))
        env['D2P_HOST_ARENA_GIB_PER_P'] = '64'
        env['P2D_HOST_ARENA_GIB_PER_P'] = '128'
        with self.assertRaisesRegex(RuntimeError, 'GLOBAL 64/128'):
            run.check_global_capacities(env)

    def test_environment_and_window_unchanged(self):
        env, _ = run.case_environment(run.CASE, run.run_dir(run.CASE))
        self.assertEqual((env['MAX_INFLIGHT'], env['WARMUP_SECONDS'], env['MEASURE_SECONDS']), ('512','300','1200'))
        self.assertEqual(env['SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB'], '768')
        run.prior.validate_inputs()

    def test_isolated_current_control(self):
        self.assertEqual(run.dram.base.CONTROL, run.CONTROL)
        self.assertEqual(run.dram.base.CASES, [run.CASE])
        self.assertIn('global64-128', str(run.run_dir(run.CASE)))
        self.assertNotIn('dram-d64p128', str(run.CONTROL))

    def test_report_preserves_all_old_results_and_explicitly_labels_global(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.md'
            original = 'Old results preserved.\n'
            path.write_text(original)
            with patch.object(run, 'REPORT', path):
                run.render_report({})
                first = path.read_text()
                run.render_report({})
            self.assertEqual(first, path.read_text())
            self.assertTrue(first.startswith(original))
            self.assertIn('总计192 GiB', first)
            self.assertIn('每P分别16/32 GiB', first)

    def test_formal_capacity_gate_rejects_wrong_observed_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'closed_loop_boundaries.json').write_text(json.dumps(dict(measurement_start_wall=1, measurement_end_wall=3)))
            row = dict(ts=2, valid_sample=True, pools=[dict(kind='arena', direction='d2p', capacity_bytes=64*1024**3)]*4)
            (path/'host_resources_2s.jsonl').write_text(json.dumps(row)+'\n')
            with patch.object(run, 'original_summarize', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'observed GLOBAL d2p'):
                    run.summarize(path, {})

    def test_formal_capacity_gate_accepts_correct_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/'closed_loop_boundaries.json').write_text(json.dumps(dict(measurement_start_wall=1, measurement_end_wall=3)))
            pools = [dict(kind='arena', direction=d, capacity_bytes=c*1024**3) for d,c in [('d2p',16)]*4+[('p2d',32)]*4]
            (path/'host_resources_2s.jsonl').write_text(json.dumps(dict(ts=2, valid_sample=True, pools=pools))+'\n')
            with patch.object(run, 'original_summarize', return_value={}):
                result = run.summarize(path, {})
            self.assertEqual(result['global_host_capacity_gib'], dict(d2p=64,p2d=128,total=192))


if __name__ == '__main__':
    unittest.main()
