"""Mutate temporary candidate fixtures only, never a serving checkout."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("common_case_candidate_proof", ROOT / "scripts/verify_case_configuration.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def copy_fixture(root):
    relatives = ["scripts/new_method/" + name for name in proof.NAMES] + [
        "scripts/new_method/internal/agentic_case_environment.sh",
        "scripts/new_method/internal/apply_experiment_profile.sh",
        "scripts/tools/load_experiment_profile.py", "scripts/cluster/multinode.py"]
    for relative in relatives:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


class FullCaseSourceProof(unittest.TestCase):
    def test_actual_candidate(self):
        self.assertEqual(proof.verify()["exact_common_bindings"], 16)

    def test_complete_source_rejects_semantic_and_structural_mutations(self):
        prefix = "scripts/new_method/"
        cases = [
            (prefix + "internal/agentic_case_environment.sh", "FAST_TOOL_THRESHOLD_SECONDS:-2", "FAST_TOOL_THRESHOLD_SECONDS:-30"),
            (prefix + "internal/agentic_case_environment.sh", "  DUALPD_AGENTIC_COMMON_ENV=(", "  export EXTRA_STATE=1\n  DUALPD_AGENTIC_COMMON_ENV=("),
            (prefix + "run_4p4d_numa_case.sh", '"${DUALPD_AGENTIC_COMMON_ENV[@]}"', '"${DUALPD_AGENTIC_COMMON_ENV[*]}"'),
            (prefix + "run_4p4d_numa_case.sh", "PREFILL_GPUS:-0 2 4 6", "PREFILL_GPUS:-0 1 2 3"),
            (prefix + "run_4p4d_numa_case.sh", "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS:-1", "SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS:-0"),
            (prefix + "run_4p4d_numa_case.sh", "P2D_HOST_ARENA_GIB_PER_P:-64", "P2D_HOST_ARENA_GIB_PER_P:-32"),
            (prefix + "run_4p4d_numa_case.sh", "dualpd_build_agentic_case_environment\n", ":\n"),
            (prefix + "run_4p4d_numa_case.sh", "BOOTSTRAP_PORT=28300", "BOOTSTRAP_PORT=28301"),
            (prefix + "run_1p3d_case.sh", "MAX_INFLIGHT:-256", "MAX_INFLIGHT:-512"),
            (prefix + "internal/apply_experiment_profile.sh", '"${profile_path}")" || return', '"${profile_path}")"'),
            ("scripts/tools/load_experiment_profile.py", '    "FAST_TOOL_THRESHOLD_SECONDS",\n', ""),
            ("scripts/cluster/multinode.py", 'n["host_ip"] == router_node["host_ip"]', 'n["node_id"] == router_node["node_id"]'),
            ("scripts/cluster/multinode.py", 'n["host_ip"] == relay_node["host_ip"]', 'n["node_id"] == relay_node["node_id"]'),
            ("scripts/cluster/multinode.py", 'if relay_node["host_ip"] == router_node["host_ip"]:', 'if control["node_id"] == router["node_id"]:'),
        ]
        with tempfile.TemporaryDirectory(prefix="dualpd-common-case-proof-") as directory:
            root = Path(directory) / "candidate"
            copy_fixture(root)
            for relative, before, after in cases:
                with self.subTest(path=relative, mutation=before):
                    path = root / relative
                    source = path.read_text()
                    self.assertIn(before, source)
                    try:
                        path.write_text(source.replace(before, after))
                        with self.assertRaises(AssertionError):
                            proof.verify(root)
                    finally:
                        path.write_text(source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
