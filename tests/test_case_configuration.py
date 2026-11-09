"""Capture complete Bash argv only: no environment/native/GPU acceptance.

Fingerprint checks are explicitly stubbed; the real pipeline is never executed.
The actual Bash wrappers and real YAML loader still construct every argument.
"""
import json
import os
from pathlib import Path
from functools import lru_cache
import re
import subprocess
import sys
import tempfile
import unittest

FRONTEND = Path(__file__).resolve().parents[1]
CANDIDATE = FRONTEND
PYTHON = Path(sys.executable)
NAMES = ("run_1p3d_case.sh", "run_1p5d_case.sh", "run_2p6d_numa_case.sh", "run_4p4d_numa_case.sh")
INPUTS = {"LATE_BIND_POLL_INTERVAL_SECONDS": "0.125", "P_ACCEPT_TIMEOUT_SECONDS": "17",
          "P_QUEUE_TIMEOUT_SECONDS": "18", "P_READY_TIMEOUT_SECONDS": "19",
          "WARMUP_SECONDS": "300", "MAX_WARMUP_SECONDS": "420", "MEASURE_SECONDS": "1200",
          "FAST_TOOL_THRESHOLD_SECONDS": "1", "DIRECT_WAIT_SECONDS": "1",
          "SELECTED_IO_CAP": "3", "P_H2D_MAX_INFLIGHT": "2"}
BASE = "0ed608d2906ed0f3b1b0b27f33e38b9ebda75495"


@lru_cache(maxsize=None)
def original_relative(relative):
    return subprocess.check_output(
        ["git", "-C", str(FRONTEND), "show", BASE + ":" + relative], text=True)


def frozen_source(name):
    return original_relative("scripts/new_method/" + name)


def normalize(value, reference_root):
    if isinstance(value, str) and value.startswith(str(reference_root)):
        return str(FRONTEND) + value[len(str(reference_root)): ]
    if isinstance(value, dict):
        return {k: normalize(v, reference_root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, reference_root) for v in value]
    return value


def compare(name, overrides, *, profile=False):
    with tempfile.TemporaryDirectory(prefix="dualpd-case-argv-") as directory:
        root = Path(directory)
        old_tree = root / "immutable-old"
        relatives = ["scripts/new_method/" + name for name in NAMES] + [
            "scripts/new_method/internal/apply_experiment_profile.sh",
            "scripts/tools/load_experiment_profile.py"]
        for relative in relatives:
            path = old_tree / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(original_relative(relative))
        stub = root / "bin/python"
        stub.parent.mkdir()
        stub.write_text('''#!/bin/bash
case "$1" in
  */check_environments.py) exit 0;;
  */load_experiment_profile.py) exec "$DUALPD_CAPTURE_PYTHON" "$@";;
  *) echo "unexpected fixture Python invocation: $1" >&2; exit 91;;
esac
''')
        stub.chmod(0o700)
        capture = root / "capture.py"
        capture.write_text('''import json,re,sys
bindings={}; duplicates=[]
for index,arg in enumerate(sys.argv[1:]):
 if not re.match(r"^[A-Z][A-Z0-9_]*=",arg):
  command=sys.argv[index+1:]
  break
 key,value=arg.split("=",1)
 if key in bindings: duplicates.append(key)
 bindings[key]=value
else: raise AssertionError("missing pipeline command")
assert command[0]=="bash" and len(command)==2 and command[1].endswith("/internal/run_agentic_pipeline.sh")
print(json.dumps({"bindings":bindings,"duplicates":duplicates,"command":command},sort_keys=True))
''')
        env = dict(os.environ)
        # Clear all case source knobs, not just the sixteen shared settings.
        sources = "\n".join(frozen_source(case) for case in NAMES)
        knobs = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)", sources))
        for key in knobs | set(INPUTS) | {"EXPERIMENT_CONFIG", "BASH_ENV", "ENV"}:
            env.pop(key, None)
        for key in list(env):
            if key.startswith("BASH_FUNC_"):
                env.pop(key)
        env.update(PD_ENV_BIN=str(stub.parent), RUN_DIR=str(root / "run"),
                   DUALPD_CAPTURE_PYTHON=str(PYTHON), DUALPD_CAPTURE_SCRIPT=str(capture))
        env.update(overrides)
        if profile:
            path = root / "profile.yaml"
            path.write_text("environment:\n  FAST_TOOL_THRESHOLD_SECONDS: 3\n  DIRECT_WAIT_SECONDS: 4\n  MAX_INFLIGHT: 71\n")
            env["EXPERIMENT_CONFIG"] = str(path)
        code = '''env() { "$DUALPD_CAPTURE_PYTHON" "$DUALPD_CAPTURE_SCRIPT" "$@"; }
export -f env
exec bash "$1"
'''
        results = []
        for tree in (old_tree, CANDIDATE):
            result = subprocess.run(["bash", "-c", code, "case-argv-fixture",
                                     str(tree / "scripts/new_method" / name)],
                                    env=env, text=True, capture_output=True, timeout=30)
            assert result.returncode == 0, result.stdout + result.stderr
            # A valid profile logs its provenance before the final capture line.
            results.append(normalize(json.loads(result.stdout.splitlines()[-1]), old_tree))
        assert not results[0]["duplicates"] and not results[1]["duplicates"]
        assert results[0] == results[1], (name, overrides, results)
        return results[1]


class CommonCaseArguments(unittest.TestCase):
    def test_all_four_topologies_default_arguments(self):
        for name in NAMES:
            with self.subTest(case=name):
                compare(name, {})

    def test_all_four_topologies_explicit_common_overrides(self):
        for name in NAMES:
            with self.subTest(case=name):
                data = compare(name, INPUTS)
                self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD"], "1")
                self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT"], "1")

    def test_empty_input_knobs_retain_original_fallbacks(self):
        for name in NAMES:
            with self.subTest(case=name):
                compare(name, {key: "" for key in INPUTS})

    def test_raw_target_variables_do_not_change_case_knob_precedence(self):
        for name in NAMES:
            with self.subTest(case=name):
                compare(name, {"SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT": "99",
                               "SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD": "99"})

    def test_shared_value_stays_one_literal_argument(self):
        literal = "literal spaces $HOME $(printf INJECTED) \\ *"
        for name in NAMES:
            with self.subTest(case=name):
                data = compare(name, {"LATE_BIND_POLL_INTERVAL_SECONDS": literal})
                self.assertEqual(data["bindings"]["SGLANG_PD_LATE_BIND_POLL_INTERVAL_S"], literal)

    def test_other_case_entries_keep_existing_profile_scope(self):
        # Only4P4D currently applies this profile at the case layer. Do not add
        # new profile semantics to other topologies as an extraction side effect.
        for name in NAMES[:-1]:
            with self.subTest(case=name):
                data = compare(name, {}, profile=True)
                self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD"], "2")

    def test_real_profile_precedes_common_array_construction(self):
        data = compare("run_4p4d_numa_case.sh", {}, profile=True)
        self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD"], "3")
        self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT"], "4")
        self.assertEqual(data["bindings"]["MAX_INFLIGHT"], "71")

    def test_explicit_values_override_profile_before_construction(self):
        data = compare("run_4p4d_numa_case.sh", INPUTS, profile=True)
        self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_FAST_TOOL_THRESHOLD"], "1")
        self.assertEqual(data["bindings"]["SGLANG_AGENTIC_KV_DIRECT_HANDSHAKE_TIMEOUT"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
