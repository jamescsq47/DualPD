import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "qwen35_122b_c64_autopilot",
    ROOT / "qwen35_122b_c64_autopilot.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Qwen35122BC64AutopilotTests(unittest.TestCase):
    def test_hicache_precedes_native_and_uses_old_baseline_cli_spelling(self):
        self.assertEqual(
            [item["method"] for item in module.core.EXPERIMENTS],
            ["agentic", "colocated", "hicache", "native"],
        )
        experiment = next(
            item for item in module.core.EXPERIMENTS if item["method"] == "hicache"
        )
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            module.core, "run_dir", return_value=Path(temporary) / "run"
        ), patch.object(module.core, "command") as command:
            module.prepare_and_start(experiment)

        prepare = [str(value) for value in command.call_args_list[0].args[0]]
        self.assertEqual(
            prepare[prepare.index("--mamba-cache-strategy-flag") + 1],
            "scheduler",
        )

    def test_colocated_uses_current_pd_source_and_compatible_environment(self):
        experiment = next(
            item
            for item in module.core.EXPERIMENTS
            if item["method"] == "colocated"
        )
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            module.core, "run_dir", return_value=Path(temporary) / "run"
        ), patch.object(module.core, "command") as command:
            module.prepare_and_start(experiment)

        prepare = [str(value) for value in command.call_args_list[0].args[0]]
        self.assertEqual(
            prepare[prepare.index("--sglang-root") + 1],
            str(module.core.PD_ENGINE),
        )
        self.assertEqual(
            prepare[prepare.index("--python") + 1],
            str(module.COLOCATED_PYTHON),
        )
        self.assertNotIn(str(module.core.BASELINE_ENGINE), prepare)


if __name__ == "__main__":
    unittest.main()
