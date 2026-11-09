"""Profile validation and immutable historical success parity; no model launch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT
FUNCTION = ROOT / "scripts/new_method/internal/apply_experiment_profile.sh"
LOADER = FRONTEND / "scripts/tools/load_experiment_profile.py"
PYTHON = Path(sys.executable)
BASE = "0ed608d2906ed0f3b1b0b27f33e38b9ebda75495"
KEYS = ("TEMPERATURE", "TOP_P", "EXPERIMENT_CONFIG", "WORKLOAD_CONFIG",
        "PROFILE_ORIGIN_TEMPERATURE", "P2D_HOST_STAGING")


def run_case(profile, *, initial=None, strict=False, loader=LOADER,
             function=FUNCTION, python_bin=PYTHON, missing_parent=False,
             conditional=False, symlink_relative=False):
    with tempfile.TemporaryDirectory(prefix="dualpd-profile-candidate-") as directory:
        path = Path(directory) / "profile.yaml"
        path.write_text(profile)
        canonical = path
        if symlink_relative:
            link = Path(directory) / "linked-profile.yaml"
            link.symlink_to(path.name)
            path = link
        env = dict(os.environ)
        for key in KEYS:
            env.pop(key, None)
        env.update(initial or {})
        code = r'''
source "$1"
apply_experiment_profile "$2" "$3" "$4" >/dev/null
status=$?
# Origin labels intentionally remain shell-local in the original implementation.
# Inspect their value explicitly rather than requiring a new exported variable.
"$5" -c 'import json,os,sys; print(json.dumps({"status":int(sys.argv[1]),"values":{**{k:os.environ.get(k) for k in sys.argv[3:]},"PROFILE_ORIGIN_TEMPERATURE":sys.argv[2] or None}}))' "$status" "${PROFILE_ORIGIN_TEMPERATURE-}" TEMPERATURE TOP_P EXPERIMENT_CONFIG WORKLOAD_CONFIG P2D_HOST_STAGING
'''
        if conditional:
            code = code.replace('apply_experiment_profile "$2" "$3" "$4" >/dev/null\nstatus=$?',
                'if apply_experiment_profile "$2" "$3" "$4" >/dev/null; then status=0; else status=$?; fi')
        if missing_parent:
            path = Path(directory) / "nonexistent-directory/profile.yaml"
        result = subprocess.run(
            ["bash", *( ["-e"] if strict else []), "-c", code,
             "profile-candidate", str(function), path.name if symlink_relative else str(path), str(loader),
             str(python_bin), str(PYTHON)],
            env=env, text=True, capture_output=True, timeout=30, cwd=directory)
        return result, canonical if symlink_relative else path


class AtomicProfileContract(unittest.TestCase):
    def test_invalid_profile_stops_real_wrapper_before_fingerprint_or_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="dualpd-profile-launch-guard-") as directory:
            root = Path(directory)
            profile = root / "invalid.yaml"
            profile.write_text("environment:\n  TEMPERATURE: 0\n  TOP_P: null\n")
            fake_python = root / "bin/python"
            fake_python.parent.mkdir()
            fake_python.write_text('''#!/bin/bash
case "$1" in
  */load_experiment_profile.py) exec "$PROFILE_TEST_PYTHON" "$@";;
  *) "$PROFILE_TEST_PYTHON" -c 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("unexpected fingerprint invocation")' "$PROFILE_TEST_MARKER"; exit 99;;
esac
''')
            fake_python.chmod(0o700)
            marker = root / "fingerprint-was-called"
            env = dict(os.environ)
            for key in list(env):
                if key.startswith("BASH_FUNC_") or key in {"BASH_ENV", "ENV", "TEMPERATURE", "TOP_P"}:
                    env.pop(key)
            env.update(EXPERIMENT_CONFIG=str(profile), PD_ENV_BIN=str(fake_python.parent),
                       RUN_DIR=str(root / "run"), PROFILE_TEST_PYTHON=str(PYTHON),
                       PROFILE_TEST_MARKER=str(marker))
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/new_method/run_4p4d_numa_case.sh")],
                env=env, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("TypeError", result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((root / "run").exists())

    def test_symlink_profile_keeps_canonical_selection(self):
        result, canonical = run_case("environment:\n  TEMPERATURE: 0\n", symlink_relative=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], 0)
        self.assertEqual(data["values"]["EXPERIMENT_CONFIG"], str(canonical))

    def test_success_then_failure_in_same_shell_preserves_first_selection(self):
        with tempfile.TemporaryDirectory(prefix="dualpd-profile-sequence-") as directory:
            first = Path(directory) / "valid.yaml"
            second = Path(directory) / "invalid.yaml"
            first.write_text("environment:\n  TEMPERATURE: 0.25\n")
            second.write_text("environment:\n  TEMPERATURE: 0.5\n  TOP_P: null\n")
            env = dict(os.environ)
            for key in KEYS:
                env.pop(key, None)
            code = r'''
source "$1"
apply_experiment_profile "$2" "$4" "$5" >/dev/null
if apply_experiment_profile "$3" "$4" "$5" >/dev/null; then status=0; else status=$?; fi
"$5" -c 'import json,os,sys; print(json.dumps([int(sys.argv[1]),os.environ.get("TEMPERATURE"),os.environ.get("EXPERIMENT_CONFIG"),os.environ.get("TOP_P"),sys.argv[2]]))' "$status" "${PROFILE_ORIGIN_TEMPERATURE-}"
'''
            result = subprocess.run(["bash", "-e", "-c", code, "profile-sequence",
                                     str(FUNCTION), str(first), str(second), str(LOADER), str(PYTHON)],
                                    env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            status, temperature, selected, top_p, origin = json.loads(result.stdout)
            self.assertNotEqual(status, 0)
            self.assertEqual((temperature, selected, top_p, origin),
                             ("0.25", str(first), None, "profile"))

    def test_successful_inputs_match_original_function(self):
        with tempfile.TemporaryDirectory(prefix="dualpd-old-profile-") as directory:
            original = Path(directory) / "apply_experiment_profile.sh"
            original.write_text(subprocess.check_output(["git", "-C", str(ROOT), "show",
                BASE + ":scripts/new_method/internal/apply_experiment_profile.sh"], text=True))
            cases = [("environment:\n  TEMPERATURE: 0\n  TOP_P: 1\n", {}),
                     ("environment:\n  TEMPERATURE: 0\n", {"TEMPERATURE": ""}),
                     ("environment:\n  TEMPERATURE: ''\n", {}),
                     ("environment:\n  TEMPERATURE: 'literal $(printf INJECTED) $HOME \\ * spaces'\n", {})]
            for profile, initial in cases:
                with self.subTest(profile=profile, initial=initial):
                    actual, actual_path = run_case(profile, initial=initial)
                    previous, previous_path = run_case(profile, initial=initial, function=original)
                    self.assertEqual(actual.returncode, 0, actual.stderr)
                    self.assertEqual(previous.returncode, 0, previous.stderr)
                    actual_data = json.loads(actual.stdout)
                    previous_data = json.loads(previous.stdout)
                    self.assertEqual(actual_data["values"].pop("EXPERIMENT_CONFIG"), str(actual_path))
                    self.assertEqual(previous_data["values"].pop("EXPERIMENT_CONFIG"), str(previous_path))
                    self.assertEqual(actual_data, previous_data)

    def test_conditional_call_still_returns_failure_and_preserves_origin(self):
        result, _ = run_case("environment:\n  TEMPERATURE: 0\n  TOP_P: null\n",
                             strict=True, conditional=True,
                             initial={"PROFILE_ORIGIN_TEMPERATURE": "earlier"})
        self.assertEqual(result.returncode, 0)
        data = json.loads(result.stdout)
        self.assertNotEqual(data["status"], 0)
        self.assertIsNone(data["values"]["TEMPERATURE"])
        self.assertIsNone(data["values"]["EXPERIMENT_CONFIG"])
        self.assertEqual(data["values"]["PROFILE_ORIGIN_TEMPERATURE"], "earlier")

    def test_path_or_python_failure_cannot_select_profile(self):
        for extra in ({"missing_parent": True}, {"python_bin": ROOT / "missing-python"}):
            with self.subTest(extra=extra):
                result, _ = run_case("environment:\n  TEMPERATURE: 0\n", **extra)
                data = json.loads(result.stdout)
                self.assertNotEqual(data["status"], 0)
                self.assertTrue(all(value is None for value in data["values"].values()))

    def test_valid_values_and_relative_path(self):
        result, path = run_case(
            "schema_version: 1\nenvironment:\n  TEMPERATURE: 0\n  TOP_P: 1\n"
            "  P2D_HOST_STAGING: true\n  WORKLOAD_CONFIG: ../task.yaml\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], 0)
        self.assertEqual(data["values"]["TEMPERATURE"], "0")
        self.assertEqual(data["values"]["TOP_P"], "1")
        self.assertEqual(data["values"]["P2D_HOST_STAGING"], "true")
        self.assertEqual(data["values"]["PROFILE_ORIGIN_TEMPERATURE"], "profile")
        self.assertEqual(data["values"]["WORKLOAD_CONFIG"], str((path.parent / "../task.yaml").resolve()))

    def test_existing_explicit_override_is_preserved(self):
        result, _ = run_case("environment:\n  TEMPERATURE: 0\n", initial={"TEMPERATURE": "0.25"})
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], 0)
        self.assertEqual(data["values"]["TEMPERATURE"], "0.25")
        self.assertEqual(data["values"]["PROFILE_ORIGIN_TEMPERATURE"], "override")

    def test_bad_profiles_cannot_partially_apply_or_select(self):
        cases = ["environment:\n  TEMPERATURE: 0\n  TOP_P: null\n",
                 "environment:\n  UNREVIEWED_INJECTION: 1\n",
                 "schema_version: 99\nenvironment:\n  TEMPERATURE: 0\n",
                 "environment: {}\n", "[]\n", "environment: [\n"]
        initial = {"TEMPERATURE": "0.5", "EXPERIMENT_CONFIG": "original-profile"}
        for profile in cases:
            with self.subTest(profile=profile):
                result, _ = run_case(profile, initial=initial)
                self.assertEqual(result.returncode, 0)
                data = json.loads(result.stdout)
                self.assertNotEqual(data["status"], 0)
                self.assertEqual(data["values"]["TEMPERATURE"], "0.5")
                self.assertEqual(data["values"]["EXPERIMENT_CONFIG"], "original-profile")
                self.assertIsNone(data["values"]["PROFILE_ORIGIN_TEMPERATURE"])
                self.assertIsNone(data["values"]["TOP_P"])

    def test_strict_launcher_stops_after_rejected_profile(self):
        result, _ = run_case("environment:\n  TEMPERATURE: 0\n  TOP_P: null\n", strict=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_missing_loader_propagates_failure(self):
        result, _ = run_case("environment:\n  TEMPERATURE: 0\n", loader=ROOT / "missing-loader.py")
        data = json.loads(result.stdout)
        self.assertNotEqual(data["status"], 0)
        self.assertTrue(all(value is None for value in data["values"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
