"""CPU-only gates for the allocation keeper and staged GPU runner."""
import importlib.util
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cleanup_quickcheck_test", ROOT / "scripts/cluster/engine_cleanup_quickcheck.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_environment_retains_proven_protocol_and_isolates_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "never-persist-me")
    monkeypatch.setenv("SGLANG_AGENTIC_KV_FORCE_SLOW_PATH", "true")
    monkeypatch.setenv("FAST_TOOL_THRESHOLD_SECONDS", "999")
    env, launch = runner.environment(tmp_path)
    assert "HF_TOKEN" not in env
    assert "sglang-src" not in env["PYTHONPATH"]
    assert env["PD_ENV_BIN"] == str(ROOT / ".venv/bin")
    assert env["DUALPD_ENGINE_ROOT"] == str(ROOT.parent / "sglang_pd")
    assert env["FAST_TOOL_THRESHOLD_SECONDS"] == env["DIRECT_WAIT_SECONDS"] == "1"
    assert env["SGLANG_AGENTIC_KV_FORCE_SLOW_PATH"] == "false"
    assert env["WARMUP_SECONDS"] == "300"
    assert env["MEASURE_SECONDS"] == "1200"
    assert env["MAX_INFLIGHT"] == "256"
    assert env["D2P_HOST_ARENA_GIB_PER_P"] == "128"
    assert env["P2D_HOST_ARENA_GIB_PER_P"] == "64"
    assert env["PREFILL_GPUS"] == "0 2 4 6"
    assert env["DECODE_GPUS"] == "1 3 5 7"
    assert env["DECODE_MEM_FRACTION_STATICS"] == "0.80 0.80 0.80 0.60"
    assert Path(launch[1]).is_file()


def plan_fixture(tmp_path, monkeypatch):
    env = {"RUN_DIR": str(tmp_path), "WARMUP_SECONDS": "300", "MEASURE_SECONDS": "1200"}
    monkeypatch.setattr(runner, "source_state", lambda: {"revision": "abc"})
    monkeypatch.setattr(runner, "environment", lambda _: (env, ["bash", "owned.sh"]))
    path = tmp_path / "plan.json"
    runner.write_json(path, {"directory": str(tmp_path), "source": {"revision": "abc"},
                            "env": env, "command": ["bash", "owned.sh"], "input_sha256": {}})
    return path


def audit(path, decision="GO", digest=None):
    runner.write_json(path.parent / "audit.json", {
        "decision": decision, "plan_sha256": digest or runner.sha(path),
    })


def test_exact_audited_plan_required(tmp_path, monkeypatch):
    path = plan_fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        runner.validate_plan(path)
    audit(path, "NO-GO")
    with pytest.raises(RuntimeError, match="independent audit"):
        runner.validate_plan(path)
    audit(path, digest="wrong-digest")
    with pytest.raises(RuntimeError, match="independent audit"):
        runner.validate_plan(path)
    audit(path)
    assert runner.validate_plan(path)["source"] == {"revision": "abc"}
    monkeypatch.setattr(runner, "source_state", lambda: {"revision": "different"})
    with pytest.raises(RuntimeError, match="source changed"):
        runner.validate_plan(path)


def test_no_launch_when_recipe_changes(tmp_path, monkeypatch):
    path = plan_fixture(tmp_path, monkeypatch)
    audit(path)
    monkeypatch.setattr(runner, "environment", lambda _: ({}, ["another.sh"]))
    with pytest.raises(RuntimeError, match="launch recipe changed"):
        runner.validate_plan(path)


def test_cleanup_gate_requires_three_consecutive_clean_probes(monkeypatch):
    samples = iter([{"gpu": []}, {"gpu": [12]}, {"gpu": []}, {"gpu": []}, {"gpu": []}])
    seen = []
    def probe(_):
        value = next(samples)
        seen.append(value)
        return value
    monkeypatch.setattr(runner, "resources", probe)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    assert runner.cleanup_gate({}) == {"gpu": []}
    assert len(seen) == 5


def test_dirty_resource_gate_never_kills_unknown_processes(monkeypatch):
    monkeypatch.setattr(runner, "resources", lambda _: {"helpers": ["foreign"]})
    monkeypatch.setattr(runner.os, "killpg", lambda *_: pytest.fail("must not signal foreign PID"))
    with pytest.raises(RuntimeError, match="no unknown processes killed"):
        runner.cleanup_gate({}, timeout=0)


def row(ts, tokens, role="decode"):
    return json.dumps({"ts": ts, "role": role, "endpoint_metrics": [
        {"metrics": {"sglang_realtime_tokens_total|mode=decode": tokens}},
    ]}) + "\n"


def test_metrics_reader_keeps_partial_record_for_next_poll():
    handle = io.StringIO(row(100, 1) + row(101, 3)[:-2])
    assert runner.read_decode_progress(handle) == (100, 1)
    offset = handle.tell()
    handle.seek(0, 2)
    handle.write("}\n")
    handle.seek(offset)
    assert runner.read_decode_progress(handle) == (101, 3)
    assert runner.read_decode_progress(handle) is None


def test_metrics_reader_ignores_prefill():
    assert runner.read_decode_progress(io.StringIO(row(100, 100, "prefill"))) is None


def test_stop_signals_exact_owned_group_and_waits(monkeypatch):
    actions = []
    process = SimpleNamespace(pid=123456, poll=lambda: None,
                              wait=lambda **kw: actions.append(("wait", kw)))
    monkeypatch.setattr(runner.os, "killpg", lambda *args: actions.append(args))
    monkeypatch.setattr(runner, "owned_group_alive", lambda _: False)
    runner.stop_owned_launcher(process)
    assert actions[0] == (123456, signal.SIGTERM)
    assert actions[1][0] == "wait"
    assert 299 <= actions[1][1]["timeout"] <= 300


def test_stop_does_not_signal_exited_pid(monkeypatch):
    monkeypatch.setattr(runner.os, "killpg", lambda *_: pytest.fail("PID may have been reused"))
    monkeypatch.setattr(runner, "owned_group_alive", lambda _: False)
    runner.stop_owned_launcher(SimpleNamespace(pid=123456, poll=lambda: 0, wait=lambda **_: None))


def test_already_exited_wrapper_still_waits_live_group(monkeypatch):
    alive = iter([True, False])
    waits = []
    monkeypatch.setattr(runner.os, "killpg", lambda *_: pytest.fail("must not signal exited leader"))
    monkeypatch.setattr(runner, "owned_group_alive", lambda _: next(alive))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: waits.append(seconds))
    runner.stop_owned_launcher(SimpleNamespace(pid=123456, poll=lambda: 0, wait=lambda **_: None))
    assert waits == [1]


def test_stop_waits_for_group_after_wrapper_exits(monkeypatch):
    alive = iter([True, True, False])
    waits = []
    monkeypatch.setattr(runner, "owned_group_alive", lambda _: next(alive))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: waits.append(seconds))
    monkeypatch.setattr(runner.os, "killpg", lambda *_: None)
    process = SimpleNamespace(pid=123456, poll=lambda: None, wait=lambda **_: None)
    runner.stop_owned_launcher(process)
    assert waits == [1, 1]


def test_owned_group_checks_threads_and_ignores_zombies(monkeypatch):
    monkeypatch.setattr(runner, "command", lambda _: "10 Z\n11 S\n12 Sl\n")
    assert not runner.owned_group_alive(10)
    assert runner.owned_group_alive(11)
    assert runner.owned_group_alive(12)
    assert not runner.owned_group_alive(13)


@pytest.mark.parametrize("tps,completed", [(0, 3), (-1, 3), (float("nan"), 3), (5, 0)])
def test_empty_or_invalid_result_rejected(tps, completed):
    with pytest.raises(RuntimeError, match="no valid positive"):
        runner.comparison({}, {"decode_tps": tps, "completed_agents": completed})


def test_comparison_requires_same_settings_and_accepted_reference(tmp_path):
    reference = tmp_path / "result.json"
    runner.write_json(reference, {"decode_tps": 100, "acceptance": "baseline", "cleanup_passed": True})
    settings = {"MAX_INFLIGHT": "256", "RUN_DIR": "/old"}
    runner.write_json(tmp_path / "plan.json", {"env": settings, "input_sha256": {}})
    plan = {"reference": {"path": str(reference)}, "env": {**settings, "RUN_DIR": "/new"},
            "input_sha256": {}}
    result = {"decode_tps": 89, "completed_agents": 10}
    assert runner.comparison(plan, result)["acceptance"] == "regression_needs_repeat"
    assert runner.comparison(plan, {**result, "decode_tps": 95})["acceptance"] == "pass"
    plan["env"]["MAX_INFLIGHT"] = "128"
    with pytest.raises(RuntimeError, match="identical settings"):
        runner.comparison(plan, result)


def test_refactor_plan_cannot_skip_before_result(tmp_path):
    with pytest.raises(RuntimeError, match="before-result"):
        runner.prepare(tmp_path, "123", "host-transport")


def test_slurm_record_parser(monkeypatch):
    monkeypatch.setattr(runner, "command", lambda _: "JobId=123 JobState=PENDING Reason=None NodeList= NumNodes=1")
    assert runner.job_state("123") == {
        "JobId": "123", "JobState": "PENDING", "Reason": "None", "NodeList": "", "NumNodes": "1",
    }


def test_frontend_fingerprint_covers_untracked_sources(tmp_path):
    module = tmp_path / "runtime/untracked_extension.py"
    module.parent.mkdir()
    module.write_text("revision = 1\n")
    first = runner.frontend_sources(tmp_path)
    assert "runtime/untracked_extension.py" in first
    module.write_text("revision = 2\n")
    assert runner.frontend_sources(tmp_path) != first
    extra = tmp_path / "data/new_workload.py"
    extra.parent.mkdir()
    extra.write_text("new_source = True\n")
    assert "data/new_workload.py" in runner.frontend_sources(tmp_path)


def test_node_lock_independent_of_plan_directory(monkeypatch):
    monkeypatch.setattr(runner.os, "getuid", lambda: 1234)
    assert runner.node_lock_path("987") == Path("/tmp/dualpd-1234-job-987.experiment.lock")
    assert runner.node_lock_path("987") != runner.node_lock_path("988")


def test_cpu_list_includes_both_numa_ranges():
    assert runner.parse_cpu_list("0-2,56-58,112,170") == {0, 1, 2, 56, 57, 58, 112, 170}


def test_recipe_follows_selected_checkout_instead_of_host_realpath(monkeypatch):
    selected = Path("/deployment-alias/DualPD")
    monkeypatch.setattr(runner, "ROOT", selected)
    recipe = runner.load_recipe()
    assert recipe.base.PD == recipe.base.SLIME == selected
    assert recipe.base.REPO == selected.parent
    assert recipe.base.MODIFIED_SGLANG == runner.ENGINE / "python"


def test_selected_root_spelling_survives_mount_alias(tmp_path):
    import os
    import subprocess
    selected = tmp_path / "checkout-alias"
    selected.symlink_to(ROOT, target_is_directory=True)
    env = {**os.environ, "DUALPD_ROOT": str(selected)}
    code = (
        "import importlib.util; from pathlib import Path; "
        f"s=importlib.util.spec_from_file_location('alias_check',{str(selected / 'scripts/cluster/engine_cleanup_quickcheck.py')!r}); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        "print(m.ROOT);print(m.load_recipe().base.PD)"
    )
    output = subprocess.check_output([str(runner.PYTHON), "-c", code], env=env, text=True)
    assert output.splitlines() == [str(selected), str(selected)]


def test_stopping_wrapper_reaches_nested_cleanup_trap(tmp_path):
    """No GPU: exercise the same wrapper->pipeline->owned-service pattern."""
    import os
    import subprocess
    import time
    marker = tmp_path / "cleaned"
    ready = tmp_path / "ready"
    inner = tmp_path / "inner.sh"
    outer = tmp_path / "outer.sh"
    inner.write_text('trap \'echo clean > "$1"; exit 0\' TERM\n'
                     'touch "$2"\nwhile true; do sleep 1; done\n')
    outer.write_text('bash "$1" "$2" "$3"\n')
    process = subprocess.Popen(["bash", str(outer), str(inner), str(marker), str(ready)],
                               start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        runner.stop_owned_launcher(process)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.read_text().strip() == "clean"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
