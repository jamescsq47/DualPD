"""Reject changes outside the complete immutable benchmark extraction."""
from pathlib import Path
import tempfile
import unittest

from dualpd.scripts import verify_benchmark_components as proof

ROOT = Path(__file__).resolve().parents[1]
FILES = (proof.RECIPE, proof.MEASUREMENT, proof.RUNNER, *proof.REPLAY_PATHS.values())


class BenchmarkStructure(unittest.TestCase):
    def test_complete_modules_and_relocated_historical_sources(self):
        result = proof.verify()
        self.assertEqual(result["complete_runtime_modules_verified"], 3)
        self.assertEqual(result["measurement_functions_verified"], 7)
        self.assertEqual(result["historical_modules_verified"], 2)
        self.assertEqual(result["unchanged_historical_modules_verified"], 1)
        self.assertEqual(result["historical_address_only_relocations_verified"], 1)
        self.assertTrue(result["controller_monitor_stop_cleanup_bodies_unchanged"])

    def test_destructive_mutations_are_rejected(self):
        cases = [
            (proof.RECIPE, "MEASURE_SECONDS='1200'", "MEASURE_SECONDS='1'"),
            (proof.RECIPE, "FAST_TOOL_THRESHOLD_SECONDS='1'", "FAST_TOOL_THRESHOLD_SECONDS='2'"),
            (proof.RECIPE, "DIRECT_WAIT_SECONDS='1'", "DIRECT_WAIT_SECONDS='2'"),
            (proof.RECIPE, "D2P_HOST_ARENA_GIB_PER_P='128'", "D2P_HOST_ARENA_GIB_PER_P='64'"),
            (proof.RECIPE, "P2D_HOST_ARENA_GIB_PER_P='64'", "P2D_HOST_ARENA_GIB_PER_P='32'"),
            (proof.RECIPE, "env.pop(\"HF_TOKEN\", None)", "env.pop(\"NOT_HF_TOKEN\", None)"),
            (proof.RECIPE, ":/usr/local/cuda/bin:/usr/local/bin", ":/usr/local/bin:/usr/local/cuda/bin"),
            (proof.RECIPE, "import os\n", "import os\nimport torch\n"),
            (proof.MEASUREMENT, 'if event_delta.get("oom", 0) or event_delta.get("oom_kill", 0):', "if False:"),
            (proof.MEASUREMENT, "end - 120", "end - 1"),
            (proof.MEASUREMENT, "cwd=root, check=True, timeout=600", "cwd=root, check=False, timeout=600"),
            (proof.MEASUREMENT, "--query-compute-apps=pid,process_name", "--query-compute-apps=pid"),
            (proof.RUNNER, "if free >= 3:", "if free >= 2:"),
            (proof.RUNNER, "    validate_component_sources()\n", ""),
            (proof.RUNNER, "if Path(__file__).resolve() != SCRIPT.resolve():", "if False:"),
            (proof.RUNNER, "signal.SIGTERM", "signal.SIGKILL"),
            (proof.MATRIX, '"Decode made no token progress in final 120 seconds"', '"removed guard"'),
            (proof.LOWC, 'FAST_TOOL_THRESHOLD_SECONDS="1"', 'FAST_TOOL_THRESHOLD_SECONDS="2"'),
            (proof.MATRIX, 'SLIME = Path(__file__).resolve().parents[3]',
             'SLIME = Path(__file__).resolve().parents[2]'),
            (proof.MATRIX, 'scripts/replay/new_method/run_qwen3_32b_tp2_browsecomp_2p6d.sh',
             'scripts/new_method/run_qwen3_32b_tp2_browsecomp_2p6d.sh'),
        ]
        with tempfile.TemporaryDirectory(prefix="dualpd-benchmark-proof-") as temp:
            root = Path(temp)
            for relative in FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text((ROOT / relative).read_text())
            proof.verify(root)
            for relative, before, after in cases:
                with self.subTest(file=relative, mutation=before):
                    path = root / proof.REPLAY_PATHS.get(relative, relative)
                    original = path.read_text()
                    self.assertIn(before, original)
                    path.write_text(original.replace(before, after, 1))
                    try:
                        with self.assertRaises(AssertionError):
                            proof.verify(root)
                    finally:
                        path.write_text(original)
