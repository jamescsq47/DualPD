
# Resolve only maintained core/shared tools; historical siblings stay local.
if __package__:
    from ._core import load_core as _load_core
else:
    from _core import load_core as _load_core

import importlib.util
from pathlib import Path
import types
import unittest
import tempfile
import threading
import time
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sampler = _load_core('dram_resource_sampler')
if __package__:
    from . import browsecomp_qwen3_8b_dram_h100_watchdog as watchdog
else:
    import browsecomp_qwen3_8b_dram_h100_watchdog as watchdog


spec = importlib.util.spec_from_file_location(
    "dram_observer_test_module", Path(__file__).resolve().parents[3] / "scripts/cluster/dram_observer/sitecustomize.py"
)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class DramObserverTests(unittest.TestCase):
    def test_hicache_counts_free_slots_without_mutating_pool(self):
        obj = types.SimpleNamespace(size=100, free_slots=[1, 2, 3], size_per_token=4096)
        original = obj.free_slots
        row = observer.pool_sample(obj, {"kind": "hicache"})
        self.assertEqual(row["used_bytes"], 97 * 4096)
        self.assertEqual(row["capacity_bytes"], 100 * 4096)
        self.assertIs(obj.free_slots, original)

    def test_arena_used_is_distinct_from_committed(self):
        obj = types.SimpleNamespace(used_bytes=128, capacity_bytes=1024,
                                    committed_bytes=1024, _closed=False)
        row = observer.pool_sample(obj, {"kind": "arena", "direction": "d2p"})
        self.assertEqual(row["used_bytes"], 128)
        self.assertEqual(row["committed_bytes"], 1024)

    def test_observation_does_not_lock_allocator(self):
        class ForbiddenLock:
            def acquire(self, *args, **kwargs):
                raise AssertionError("observer must never acquire allocator locks")
        obj = types.SimpleNamespace(used_bytes=0, capacity_bytes=1024, _lock=ForbiddenLock())
        self.assertEqual(observer.pool_sample(obj, {"kind": "arena"})["used_bytes"], 0)

    def test_invalid_observation_is_not_reported_as_zero(self):
        obj = types.SimpleNamespace(used_bytes=2048, capacity_bytes=1024)
        with self.assertRaises(ValueError):
            observer.pool_sample(obj, {"kind": "arena"})

    def test_constructor_hook_preserves_exception_and_arguments(self):
        class Original:
            def __init__(self, value, *, reject=False):
                if reject:
                    raise ValueError("original capacity rejection")
                self.value = value
        module = types.SimpleNamespace(__name__="sglang.srt.mem_cache.memory_pool_host", HostKVCache=Original)
        observer.instrument(module)
        with patch.object(observer, "register") as register, \
                patch.object(observer.sys, "argv", ["launch", "--disaggregation-mode", "decode"]):
            obj = Original(123)
            self.assertEqual(obj.value, 123)
            self.assertEqual(register.call_count, 1)
            with self.assertRaisesRegex(ValueError, "original capacity rejection"):
                Original(123, reject=True)
            self.assertEqual(register.call_count, 1)

    def test_full_prometheus_names_are_not_double_counted(self):
        text = '# TYPE master_allocated_bytes gauge\nmaster_allocated_bytes 100\nsegment_allocated_bytes{segment="a"} 100\n'
        self.assertEqual(sampler.metric_value(text, "master_allocated_bytes"), 100)
        self.assertIsNone(sampler.metric_value(text, "master_total_capacity_bytes"))

    def test_observer_setup_failure_cannot_fail_successful_constructor(self):
        class Original:
            def __init__(self):
                self.initialized = True
        module = types.SimpleNamespace(__name__="sglang.srt.mem_cache.memory_pool_host", HostKVCache=Original)
        observer.instrument(module)
        with patch.object(observer, "register", side_effect=OSError("trace disk unavailable")), \
                patch.object(observer.sys, "argv", ["launch", "--disaggregation-mode", "prefill"]):
            self.assertTrue(Original().initialized)

    def test_slow_pss_never_blocks_core_sample_or_shutdown(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_scan(_):
            entered.set()
            release.wait(10)
            return [dict(pid=1, pss_bytes=100, private_bytes=20)]
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(sampler, "process_memory", side_effect=blocked_scan):
            instance = sampler.Sampler(directory, "new_method", [])
            instance.start()
            try:
                self.assertTrue(entered.wait(2))
                with ThreadPoolExecutor(max_workers=1) as pool:
                    started = time.monotonic()
                    row, _ = instance.sample(pool)
                self.assertLess(time.monotonic() - started, 1)
                self.assertIsNone(row["process_pss_bytes"])
                self.assertFalse(row["process_sample_valid"])
                started = time.monotonic()
                instance.stop()
                self.assertLess(time.monotonic() - started, 2)
            finally:
                release.set()
                instance.process_thread.join(2)

    def test_pss_staleness_and_error_are_explicit_not_zero(self):
        instance = sampler.Sampler(Path("/unused"), "new_method", [])
        instance.process_snapshot = dict(ts=100, duration_seconds=20,
                                        processes=[dict(pid=1)], process_pss_bytes=42,
                                        process_private_bytes=12)
        self.assertEqual(instance.cached_process_memory(101)["process_pss_bytes"], 42)
        self.assertIsNone(instance.cached_process_memory(221)["process_pss_bytes"])
        instance.process_failure = "injected reader failure"
        self.assertEqual(instance.cached_process_memory(221)["process_sample_error"], "injected reader failure")
        self.assertIsNone(instance.cached_process_memory(101)["process_pss_bytes"])

    def test_watchdog_writer_node_probe_tolerates_partial_append(self):
        real_run = subprocess.run
        def local_ssh(args, **kwargs):
            self.assertEqual(args[0], "ssh")
            self.assertEqual(args[-2], "node2700")
            return real_run(shlex.split(args[-1]), **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "engine_metrics.jsonl").touch()
            row = dict(ts=time.time(), pool_count=8, valid_sample=True, pool_errors=[],
                       process_sample_valid=True, process_sample_age_seconds=30,
                       process_sample_error=None)
            (directory / "host_resources_2s.jsonl").write_text(json.dumps(row) + '\n{"ts":')
            with patch.object(watchdog.subprocess, "run", side_effect=local_ssh):
                result = watchdog.observation_probe("node2700", directory)
            self.assertTrue(result["dram_sample_valid"])
            self.assertEqual(result["dram_pool_count"], 8)
            self.assertLess(result["dram_sample_age_seconds"], 5)

    def test_watchdog_keeps_strict_age_and_pss_gates(self):
        state = {"case": {"status": "running", "run_dir": "/unused", "started": 1}}
        observation = dict(dram_sample_age_seconds=16, dram_sample_valid=True,
                           dram_pool_count=8, dram_pool_errors=[], pss_sample_valid=True)
        with patch.object(watchdog, "original_probe", return_value={}), \
                patch.object(watchdog.watch, "read_json", return_value=state), \
                patch.object(watchdog, "observation_probe", return_value=observation):
            with self.assertRaisesRegex(RuntimeError, "DRAM observation unhealthy"):
                watchdog.node_probe("job", "node2700")
            observation.update(dram_sample_age_seconds=1, pss_sample_valid=False)
            with self.assertRaisesRegex(RuntimeError, "PSS observation unhealthy"):
                watchdog.node_probe("job", "node2700")


if __name__ == "__main__":
    unittest.main()
