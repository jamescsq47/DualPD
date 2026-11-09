"""Whole environment/argv A/B against immutable pre-extraction controllers."""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dualpd.scripts.benchmarks import browsecomp_qwen3_8b_h100 as recipe
from dualpd.scripts.cluster import engine_cleanup_quickcheck as runner

ROOT = Path(__file__).resolve().parents[1]
BASE = "152a620edbefef4b53a65fb9150c0d6b3fb10aa4"


@lru_cache(maxsize=None)
def original(name):
    return subprocess.check_output([
        "git", "-C", str(ROOT), "show", BASE + ":scripts/cluster/" + name,
    ], text=True)


@contextmanager
def frozen_runner(root, engine, python):
    with tempfile.TemporaryDirectory(prefix="dualpd-benchmark-old-") as temp:
        directory = Path(temp) / "scripts/cluster"
        directory.mkdir(parents=True)
        for name in ("engine_cleanup_quickcheck.py", "matrix_controller.py",
                     "browsecomp_qwen3_8b_lowc_h100_matrix.py"):
            (directory / name).write_text(original(name))
        path = directory / "engine_cleanup_quickcheck.py"
        spec = importlib.util.spec_from_file_location("immutable_quickcheck_fixture", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT, module.ENGINE, module.PYTHON = root, engine, python
        # Historical dynamic imports install module aliases. Restore only this
        # process's fixture state after the comparison; never touch serving.
        with patch.dict(sys.modules):
            yield module


def compare(overrides, *, root=ROOT, cached=(), directory=Path("/fixture/run")):
    engine, python = root.parent / "selected-engine", root / "bin/custom-python"
    with frozen_runner(root, engine, python) as old, \
            patch.dict(os.environ, overrides, clear=True), \
            patch.object(Path, "glob", side_effect=lambda *args: iter(cached)) as glob:
        before = dict(os.environ)
        expected = old.environment(directory)
        actual = recipe.environment(root, engine, python, directory)
        assert actual == expected, "complete environment/argv differs"
        assert list(actual[0]) == list(expected[0]), "environment insertion order differs"
        assert dict(os.environ) == before, "recipe mutated caller environment"
        assert "HF_TOKEN" not in actual[0]
        assert glob.call_count == 2, "preserve eager cache-fallback evaluation"
        with patch.object(runner, "ROOT", root), patch.object(runner, "ENGINE", engine), \
                patch.object(runner, "PYTHON", python):
            assert runner.environment(directory) == expected
        return actual


class RecipeParity(unittest.TestCase):
    def test_direct_script_without_installation_or_working_directory(self):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "DUALPD_ROOT": str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1"}
        with tempfile.TemporaryDirectory(prefix="dualpd-direct-cwd-") as temp:
            result = subprocess.run([
                sys.executable, "-I", "-S", str(ROOT / "scripts/cluster/engine_cleanup_quickcheck.py"), "--help",
            ], cwd=temp, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("usage:", result.stdout)

    def test_foreign_selected_root_fails_before_loading_its_components(self):
        with tempfile.TemporaryDirectory(prefix="dualpd-foreign-root-") as temp:
            env = {"DUALPD_ROOT": temp, "PYTHONDONTWRITEBYTECODE": "1"}
            result = subprocess.run([
                sys.executable, "-I", "-S", str(ROOT / "scripts/cluster/engine_cleanup_quickcheck.py"), "--help",
            ], cwd=temp, env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected checkout does not match controller source", result.stderr)

    def test_source_manifest_rejects_replaced_component_origin_before_git_probe(self):
        for component in (runner.recipe, runner.measurement):
            with self.subTest(component=component.__name__), \
                    patch.object(component, "__file__", "/foreign/changed.py"), \
                    patch.object(runner, "command") as command:
                with self.assertRaisesRegex(RuntimeError, "outside selected checkout"):
                    runner.source_state()
                command.assert_not_called()

    def test_empty_caller_and_missing_cache_fallback(self):
        env, _ = compare({})
        self.assertEqual(env["SEARCH_SERVER_EMBEDDING_CACHE"],
                         str(ROOT / "data/browsecomp/artifacts/search/corpus_embeddings.pkl"))

    def test_fixed_protocol_ignores_foreign_case_overrides(self):
        env, command = compare({
            "PATH": "/fixture/host-path", "FAST_TOOL_THRESHOLD_SECONDS": "99",
            "DIRECT_WAIT_SECONDS": "99", "MEASURE_SECONDS": "1",
            "MAX_INFLIGHT": "1", "EXPERIMENT_CONFIG": "/fixture/foreign.yaml",
            "SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE": "true",
            "HF_TOKEN": "fixture-not-a-real-token",
        })
        self.assertEqual(env["FAST_TOOL_THRESHOLD_SECONDS"], "1")
        self.assertEqual(env["DIRECT_WAIT_SECONDS"], "1")
        self.assertEqual(env["MAX_INFLIGHT"], "256")
        self.assertEqual(env["WARMUP_SECONDS"], "300")
        self.assertEqual(env["MEASURE_SECONDS"], "1200")
        self.assertEqual(command, ["bash", str(ROOT / "scripts/new_method/run_4p4d_numa_case.sh")])

    def test_all_retained_caller_values_and_literal_metacharacters(self):
        values = {key: "literal $() ; * \\\\ 中文 space" for key in (
            "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "HF_TOKEN",
        )}
        compare(values)

    def test_empty_retained_values_preserve_setdefault_semantics(self):
        env, _ = compare({key: "" for key in (
            "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        )})
        self.assertEqual(env["HOME"], "")
        self.assertEqual(env["LANG"], "")

    def test_selected_roots_python_and_run_paths_are_explicit(self):
        for root in (Path("/fixture/root with spaces"), Path("/another/NFS-alias/DualPD")):
            with self.subTest(root=str(root)):
                compare({}, root=root, directory=Path("/fixture/run $literal 中文"))

    def test_first_cached_embedding_path_is_preserved(self):
        paths = (Path("/fixture/cache-first.pkl"), Path("/fixture/cache-second.pkl"))
        env, _ = compare({}, cached=paths)
        self.assertEqual(env["SEARCH_SERVER_EMBEDDING_CACHE"], str(paths[0]))

    def test_explicit_cache_override_including_empty(self):
        for value in ("", "/fixture/user-cache.pkl", "literal $() path"):
            with self.subTest(value=value):
                env, _ = compare({"SEARCH_SERVER_EMBEDDING_CACHE": value},
                                 cached=(Path("/fixture/not-selected.pkl"),))
                self.assertEqual(env["SEARCH_SERVER_EMBEDDING_CACHE"], value)

    def test_fresh_import_does_not_load_historical_or_native_modules(self):
        code = '''
import importlib.abc, importlib.util, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch', 'sglang', 'slime'}:
   raise AssertionError('forbidden dependency: '+fullname)
sys.meta_path.insert(0, Block())
original=importlib.util.spec_from_file_location
def guarded(name, location, *args, **kwargs):
 if pathlib.Path(location).name in {'matrix_controller.py','browsecomp_qwen3_8b_lowc_h100_matrix.py'}:
  raise AssertionError('historical controller dependency')
 return original(name,location,*args,**kwargs)
importlib.util.spec_from_file_location=guarded
from dualpd.scripts.cluster import engine_cleanup_quickcheck
assert not any(name.startswith(('cleanup_browsecomp_recipe','browsecomp8b_lowc_base')) for name in sys.modules)
print('standalone benchmark imports passed')
'''
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("standalone benchmark imports passed", result.stdout)
