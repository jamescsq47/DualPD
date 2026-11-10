"""Whole-file replay reconstruction and destructive mutation checks."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from dualpd.scripts import verify_replay_layout as proof

ROOT = Path(__file__).resolve().parents[1]


def copy_fixture(root):
    manifest = json.loads((ROOT / 'scripts/replay/manifest.json').read_text())
    files = {row['path'] for row in manifest['files']} | set(proof.PINNED) | set(proof.POST_ARCHIVE_HELPERS) | {
        'scripts/tools/run_swe_aiohttp_matrix.py', 'tests/test_qwen35_fused_swe_topology.py',
        'scripts/replay/README.md',
    }
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
        path.chmod((ROOT / relative).stat().st_mode & 0o777)


class ReplayLayoutTests(unittest.TestCase):
    def test_complete_immutable_inventory_and_sources(self):
        result = proof.verify()
        self.assertEqual(result['inventory_entries_verified'], 201)
        self.assertEqual(result['whole_replay_files_reconstructed'], 142)
        self.assertEqual(result['byte_identical_retained_code'], 52)
        self.assertEqual(result['post_archive_whole_shared_tools_reconstructed'], 2)
        self.assertEqual(result['post_archive_whole_router_entries_reconstructed'], 2)
        self.assertEqual(result['post_archive_helper_hashes_verified'], 1)
        self.assertEqual(result['finite_special_edit_files'], 24)
        self.assertEqual(result['complete_outside_callers_verified'], 2)

    def test_destructive_source_and_proof_data_mutations_fail(self):
        cases = [
            ('scripts/new_method/internal/launch_late_binding_router.py',
             '["launch_router"]("current")', '["launch_router"]("legacy_mamba")'),
            ('scripts/new_method/internal/launch_mamba_late_binding_router.py',
             'parents[3]', 'parents[4]'),
            ('runtime/engine_entry.py',
             'if "DUALPD_ENGINE_ROOT" in os.environ or engine.exists():', 'if engine.exists():'),
            ('scripts/replay/cluster/matrix_controller.py',
             'FAST_TOOL_THRESHOLD_SECONDS="1"', 'FAST_TOOL_THRESHOLD_SECONDS="7"'),
            ('scripts/replay/cluster/matrix_controller.py',
             'SLIME = Path(__file__).resolve().parents[3]',
             'SLIME = Path(__file__).resolve().parents[2]'),
            ('scripts/replay/cluster/matrix_controller.py',
             'if consecutive_free >= 3:', 'if consecutive_free >= 1:'),
            ('scripts/replay/cluster/browsecomp_qwen3_8b_lowc_h100_matrix.py',
             'FAST_TOOL_THRESHOLD_SECONDS="1"', 'FAST_TOOL_THRESHOLD_SECONDS="30"'),
            ('scripts/replay/cluster/browsecomp_qwen3_8b_dram_h100_matrix.py',
             'OBSERVER = ROOT / "scripts/cluster/dram_observer"',
             'OBSERVER = Path(__file__).with_name("dram_observer")'),
            ('scripts/replay/cluster/browsecomp_qwen3_8b_dram_d64p128_h100.py',
             'Path(__file__).with_name("_core.py")', 'Path(__file__).with_name("missing.py")'),
            ('scripts/replay/cluster/minimax_m27_three_node_autopilot.py',
             "native = _load_core('qwen35_native_pd_slurm')", 'native = None'),
            ('scripts/replay/cluster/qwen38_27b_hicache_ratio12_single_run.py',
             'if __package__:', 'if False:'),
            ('scripts/replay/new_method/run_qwen3_32b_tp2_terminal_bench_2p6d.sh',
             '${SCRIPT_DIR}/../../../runs-host/', '${SCRIPT_DIR}/../../runs-host/'),
            ('scripts/replay/new_method/run_qwen3_8b_tp1_browsecomp_4p4d.sh',
             'FAST_TOOL_THRESHOLD_SECONDS:-2', 'FAST_TOOL_THRESHOLD_SECONDS:-1'),
            ('scripts/replay/new_method/test_qwen35_27b_sequence.py',
             "'MAMBA_FULL_MEMORY_RATIO=0.5'", "'MAMBA_FULL_MEMORY_RATIO=0.9'"),
            ('scripts/cluster/qwen35_slurm.py',
             'DEFAULT_DIRECT_LANES = 2', 'DEFAULT_DIRECT_LANES = 4'),
            ('scripts/cluster/dram_resource_sampler.py', 'GIB = 1024**3', 'GIB = 1024**2'),
            ('scripts/cluster/dram_complete_frame_analysis.py',
             '"--run-dir", str(view)', '"--run-dir", str(directory)'),
            ('scripts/cluster/dram_complete_frame_analysis.py',
             'boundary["measurement_seconds"] < 1200', 'boundary["measurement_seconds"] < 120'),
            ('scripts/cluster/run_swe_sif_shards_in_step.sh',
             'SLURM_PROCID + SLURM_NTASKS * local_worker', 'SLURM_PROCID + local_worker'),
            ('scripts/cluster/run_swe_sif_shards_in_step.sh',
             'status=1', 'status=0'),
            ('scripts/cluster/run_swe_sif_shards_in_step.sh',
             '! -f "${policy}"', 'false'),
            ('scripts/tools/run_swe_aiohttp_matrix.py',
             "PD_FUSED_CONGESTION_RECOMPUTE='false'", "PD_FUSED_CONGESTION_RECOMPUTE='true'"),
            ('tests/test_qwen35_fused_swe_topology.py',
             "assert env['MEM_FRACTION_STATIC'] == '0.80'", 'assert True'),
            ('scripts/replay/cluster/_core.py',
             'if len({id(module) for module in existing}) > 1:', 'if False:'),
            ('scripts/replay/layout_rules.py', 'return current', 'return source'),
            ('scripts/replay/address-edits.json', 'MAMBA_FULL_MEMORY_RATIO=0.5', 'MAMBA_FULL_MEMORY_RATIO=1'),
            ('scripts/replay/manifest.json', 'historical-replay', 'maintained-core'),
            ('scripts/replay/cluster/browsecomp_qwen3_8b_p2d_only_h100.sbatch',
             'DualPD/scripts/replay/cluster/browsecomp_qwen3_8b_p2d_only_h100.py',
             'slime/scripts/cluster/browsecomp_qwen3_8b_p2d_only_h100.py'),
            ('scripts/replay/cluster/ppd45423_browsecomp_smoke.sh',
             'TOOLS=${ROOT}/DualPD/scripts/replay/cluster',
             'TOOLS=${ROOT}/slime/scripts/cluster'),
            ('scripts/replay/cluster/ppd45423_browsecomp_aligned_matrix.sh',
             '${ROOT}/DualPD/scripts/tools/summarize_agentic_steady.py',
             '${ROOT}/slime/scripts/tools/summarize_agentic_steady.py'),
        ]
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-source-proof-') as directory:
            root = Path(directory)
            copy_fixture(root)
            proof.verify(root)
            for relative, before, after in cases:
                with self.subTest(file=relative, mutation=before):
                    path = root / relative
                    source = path.read_text()
                    self.assertIn(before, source)
                    path.write_text(source.replace(before, after, 1))
                    try:
                        with self.assertRaises(AssertionError):
                            proof.verify(root)
                    finally:
                        path.write_text(source)

    def test_old_path_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-old-path-') as directory:
            root = Path(directory)
            copy_fixture(root)
            old = root / 'scripts/cluster/multinode_smoke.py'
            old.write_bytes((root / 'scripts/replay/cluster/multinode_smoke.py').read_bytes())
            with self.assertRaisesRegex(AssertionError, 'old source remains'):
                proof.verify(root)

    def test_archived_shell_syntax_and_precise_helper_addresses(self):
        edits = json.loads((ROOT / 'scripts/replay/address-edits.json').read_text())
        helper_repairs = [row for row in edits['files']
                          if any('slime/scripts/cluster' in edit['before']
                                 for edit in row['edits'])]
        self.assertEqual(len(helper_repairs), 14)
        self.assertEqual(sum(row['path'].endswith('.sbatch') for row in helper_repairs), 8)
        replay = ROOT / 'scripts/replay'
        shells = sorted(path for path in replay.rglob('*') if path.suffix in {'.sh', '.sbatch'})
        self.assertEqual(len(shells), 50)
        for path in shells:
            with self.subTest(shell=str(path.relative_to(ROOT))):
                # Parsing only: never submit a batch job or run a controller.
                result = subprocess.run(['/bin/bash', '-n', str(path)],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
        for row in helper_repairs:
            source = (ROOT / row['path']).read_text()
            self.assertNotIn('slime/scripts/cluster', source)
            self.assertNotIn('${ROOT}/slime/scripts/tools/summarize_agentic_steady.py', source)
            if row['path'].endswith('.sbatch'):
                helper = ROOT / 'scripts/replay/cluster' / (Path(row['path']).stem + '.py')
                self.assertTrue(helper.is_file())
            else:
                self.assertIn('TOOLS=${ROOT}/DualPD/scripts/replay/cluster', source)
                self.assertTrue((replay / 'cluster/ppd45423_browsecomp_smoke.sh').is_file())
                self.assertTrue((replay / 'cluster/ppd45423_affinity_proxy.py').is_file())
        self.assertTrue((ROOT / 'scripts/tools/summarize_agentic_steady.py').is_file())

    def test_lost_executable_mode_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-mode-') as directory:
            root = Path(directory)
            copy_fixture(root)
            path = root / 'scripts/replay/new_method/run_qwen3_8b_tp1_browsecomp_4p4d.sh'
            for mode in (0o644, 0o654, 0o645):
                with self.subTest(mode=oct(mode)):
                    path.chmod(mode)
                    with self.assertRaisesRegex(AssertionError, 'executable mode differs'):
                        proof.verify(root)

    def test_unreviewed_extra_runtime_source_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-extra-source-') as directory:
            root = Path(directory)
            copy_fixture(root)
            for name in ('rogue_helper.py', 'rogue_helper', 'rogue.so', 'rogue.pth', 'rogue.md'):
                with self.subTest(name=name):
                    path = root / 'scripts/replay/cluster' / name
                    path.write_text('raise RuntimeError\n')
                    try:
                        with self.assertRaisesRegex(AssertionError, 'source inventory'):
                            proof.verify(root)
                    finally:
                        path.unlink()

    def test_symlinks_including_cache_shaped_links_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-symlink-') as directory:
            root = Path(directory)
            copy_fixture(root)
            for name in ('rogue.py', 'missing-link', '__pycache__'):
                with self.subTest(name=name):
                    path = root / 'scripts/replay/cluster' / name
                    path.symlink_to(root / 'nonexistent')
                    try:
                        with self.assertRaisesRegex(AssertionError, 'replay symlink'):
                            proof.verify(root)
                    finally:
                        path.unlink()
