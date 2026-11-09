"""Maintained measurement/controller contracts from frozen old-behavior fixtures.

All GPU/socket/cgroup/analyzer observations are synthetic and isolated. These
are not real hardware readiness, native ownership or throughput acceptance.
"""
import copy
import io
import json
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
from dualpd.scripts.benchmarks import measurement as BASE
from dualpd.scripts.benchmarks import browsecomp_qwen3_8b_h100 as RECIPE
from dualpd.scripts.cluster import engine_cleanup_quickcheck as QUICK
ANALYSIS_PYTHON = RECIPE.ANALYSIS_ENV / "bin/python"
ZERO = {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0}
DIRECTORY = Path("/fixture/not-a-live-run")


def fixtures():
    summary = {
        "status": "steady_churn", "warmup_seconds": 300.5,
        "measurement_seconds": 1200.5, "completed_agents": 99,
        "completion_rps": 0.082, "prefill_compute_tps": 1234.0,
        "decode_tps": 567.0, "decode_tps_per_gpu": 141.75,
        "prefill_gpu_busy_fraction": 2.0, "prefill_gpus": 2,
        "decode_gpu_busy_fraction": 3.0, "decode_gpus": 2,
        "prefill_compute_tokens_per_completed_agent": 14960.0,
        "decode_tokens_per_completed_agent": 6876.0,
        "completion_weighted_reuse": {
            "page_aligned_parent_prefix_reuse_fraction": 1.0,
        },
    }
    boundary = {
        "measurement_start_wall": 1000.0, "measurement_end_wall": 2200.5,
        "state_at_measurement_start": {"failures": 2},
        "state_at_measurement_end": {"failures": 2},
    }
    launch = {"env": {"PREFILL_TP_SIZE": "2", "DECODE_TP_SIZE": "2"}}
    metrics = [
        {"ts": t, "role": role, "endpoint_metrics": [{"metrics": {
            "sglang_token_usage": 0.25 if role == "prefill" else 0.75,
            "sglang_realtime_tokens_total|mode=decode": value,
        }}]}
        for t, role, value in [
            (1000.0, "prefill", 0), (2090.0, "decode", 100),
            (2199.0, "decode", 1200), (2201.0, "decode", 999999),
        ]
    ]
    return summary, boundary, launch, metrics


def summarize(values=None, after=None):
    summary, boundary, launch, metrics = copy.deepcopy(values or fixtures())
    files = {
        "offload_analysis_summary.json": summary,
        "closed_loop_boundaries.json": boundary,
        "launch_config.json": launch,
    }
    def read(path, *args, **kwargs):
        return json.dumps(files[path.name])
    def opened(path, *args, **kwargs):
        if path.name != "engine_metrics.jsonl":
            raise AssertionError("unexpected input " + str(path))
        return io.StringIO("".join(json.dumps(row) + "\n" for row in metrics))
    with patch.object(Path, "read_text", read), patch.object(Path, "open", opened), \
            patch.object(BASE, "memory_events", return_value=after or ZERO):
        return BASE.summarize(DIRECTORY, ZERO)


class SummaryCharacterization(unittest.TestCase):
    def test_success_scope_tp_divisor_and_window(self):
        result = summarize()
        self.assertEqual(len(result), 25)
        self.assertEqual(result["prefill_forward_per_gpu"], 0.5)
        self.assertEqual(result["decode_forward_per_gpu"], 0.75)
        self.assertEqual(result["tail_120s_decode_tokens"], 1100)
        self.assertEqual(result["p_kv"], 0.25)
        self.assertEqual(result["d_kv"], 0.75)
        self.assertEqual(result["parent_kv_reuse"], 1.0)
        self.assertEqual(result["failures"], 0)

    def test_rejects_status_and_short_windows(self):
        for key, value, message in [
            ("status", "warmup_timeout_without_steady_churn", "unexpected status"),
            ("warmup_seconds", 298.9, "warmup shorter"),
            ("measurement_seconds", 1198.9, "measurement shorter"),
        ]:
            with self.subTest(key=key):
                values = fixtures()
                values[0][key] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    summarize(values)

    def test_rejects_new_request_failures(self):
        values = fixtures()
        values[1]["state_at_measurement_end"]["failures"] = 3
        with self.assertRaisesRegex(RuntimeError, "measurement failures=1"):
            summarize(values)

    def test_rejects_missing_or_stagnant_tail(self):
        for kind in ("missing", "stagnant", "decreasing"):
            with self.subTest(kind=kind):
                values = fixtures()
                if kind == "missing":
                    values[3].pop(2)
                    message = "missing Decode liveness"
                else:
                    values[3][2]["endpoint_metrics"][0]["metrics"][
                        "sglang_realtime_tokens_total|mode=decode"
                    ] = 100 if kind == "stagnant" else 99
                    message = "Decode made no token progress"
                with self.assertRaisesRegex(RuntimeError, message):
                    summarize(values)

    def test_rejects_new_oom_or_kill(self):
        for key in ("oom", "oom_kill"):
            with self.subTest(counter=key):
                with self.assertRaisesRegex(RuntimeError, "new cgroup OOM event"):
                    summarize(after={**ZERO, key: 1})


class AnalysisCharacterization(unittest.TestCase):
    def test_existing_summary_is_a_real_short_circuit(self):
        with patch.object(Path, "exists", return_value=True), \
                patch.object(BASE.subprocess, "run") as run:
            self.assertIsNone(BASE.ensure_analysis(DIRECTORY, python=ANALYSIS_PYTHON, root=ROOT))
            run.assert_not_called()

    def test_missing_raw_inputs_fail_before_subprocess(self):
        with patch.object(Path, "exists", return_value=False), \
                patch.object(BASE.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "cannot postprocess incomplete run"):
                BASE.ensure_analysis(DIRECTORY, python=ANALYSIS_PYTHON, root=ROOT)
            run.assert_not_called()

    def test_complete_raw_input_selects_exact_analyzer_python(self):
        def exists(path):
            return path.name != "offload_analysis_summary.json"
        with patch.object(Path, "exists", exists), \
                patch.object(BASE.subprocess, "run") as run:
            BASE.ensure_analysis(DIRECTORY, python=ANALYSIS_PYTHON, root=ROOT)
            run.assert_called_once_with([
                str(ANALYSIS_PYTHON),
                str(ROOT / "scripts/tools/analyze_pd_offload.py"),
                "--run-dir", str(DIRECTORY),
            ], cwd=ROOT, check=True, timeout=600)

    def test_analyzer_failure_is_not_swallowed(self):
        error = subprocess.CalledProcessError(17, ["fixture-only"])
        def exists(path):
            return path.name != "offload_analysis_summary.json"
        with patch.object(Path, "exists", exists), \
                patch.object(BASE.subprocess, "run", side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                BASE.ensure_analysis(DIRECTORY, python=ANALYSIS_PYTHON, root=ROOT)
            self.assertIs(caught.exception, error)


class ResourceCharacterization(unittest.TestCase):
    def test_cleanup_requires_three_consecutive_clean_observations(self):
        samples = [{"gpu": []}, {"gpu": [123]}, {"gpu": []}, {"gpu": []}, {"gpu": []}]
        with patch.object(QUICK, "resources", side_effect=samples) as resources, \
                patch.object(QUICK.time, "sleep") as sleep, \
                patch.object(QUICK.os, "killpg") as killpg, \
                patch.object(QUICK.os, "kill") as kill:
            self.assertEqual(QUICK.cleanup_gate({}), {"gpu": []})
            self.assertEqual(resources.call_count, 5)
            self.assertEqual(sleep.call_count, 4)
            killpg.assert_not_called()
            kill.assert_not_called()

    def test_unknown_helpers_stop_without_any_signal(self):
        with patch.object(QUICK, "resources", return_value={"helpers": ["foreign"]}), \
                patch.object(QUICK.time, "sleep") as sleep, \
                patch.object(QUICK.os, "killpg") as killpg, \
                patch.object(QUICK.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "no unknown processes killed"):
                QUICK.cleanup_gate({}, timeout=0)
            sleep.assert_not_called()
            killpg.assert_not_called()
            kill.assert_not_called()

    def test_configured_ports_union_and_numeric_filter(self):
        self.assertEqual(BASE.configured_ports({
            "PREFILL_PORTS": "27300, 27320 27300", "ROUTER_PORT": "27510",
            "IGNORE_ME": "9999", "EMPTY_PORT": "", "BAD_PORT": "invalid",
        }), [27300, 27320, 27510])

    def test_bind_failures_are_busy_and_all_sockets_close(self):
        instances = []
        class Socket:
            def __init__(self, *args):
                self.closed = False
                instances.append(self)
            def bind(self, address):
                self.address = address
                if address[1] == 27320:
                    raise OSError("fixture busy")
            def close(self):
                self.closed = True
        with patch.object(BASE.socket, "socket", Socket):
            self.assertEqual(BASE.occupied_ports([27300, 27320, 27510]), [27320])
        self.assertEqual(len(instances), 3)
        self.assertTrue(all(sock.closed for sock in instances))

    def test_gpu_probe_exact_command_and_empty_lines(self):
        with patch.object(BASE.subprocess, "run", return_value=types.SimpleNamespace(
            stdout="123, python3\n\n 456, other-python \n",
        )) as run:
            self.assertEqual(BASE.gpu_processes(), ["123, python3", "456, other-python"])
            run.assert_called_once_with([
                "nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits",
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=True, timeout=30)

    def test_gpu_probe_failure_propagates(self):
        error = subprocess.CalledProcessError(9, ["fixture-only"])
        with patch.object(BASE.subprocess, "run", side_effect=error):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                BASE.gpu_processes()
            self.assertIs(caught.exception, error)

    def test_cgroup_events_path_and_absence(self):
        seen = []
        def read(path):
            seen.append(str(path))
            if str(path) == "/proc/self/cgroup":
                return "0::/owned/step/\n"
            return "oom 2\noom_kill 1\n"
        with patch.object(Path, "read_text", read), \
                patch.object(Path, "exists", return_value=True):
            self.assertEqual(BASE.memory_events(), {"oom": 2, "oom_kill": 1})
        self.assertEqual(seen, [
            "/proc/self/cgroup", "/sys/fs/cgroup/owned/step/memory.events",
        ])
        with patch.object(Path, "read_text", read), \
                patch.object(Path, "exists", return_value=False):
            self.assertEqual(BASE.memory_events(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
