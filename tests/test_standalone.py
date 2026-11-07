"""Extraction contracts independent of the original slime checkout."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dualpd.runtime.generation import GenerateState
from dualpd.runtime.types import Sample

ROOT = Path(__file__).resolve().parents[1]


def test_sample_trace_and_dynamic_fields_roundtrip():
    sample = Sample(index=12, status=Sample.Status.COMPLETED,
                    metadata={"turn_metrics": [{"prompt_tokens": 42, "completion_tokens": 8}]})
    sample.trace = {"version": 1, "events": [{"kind": "tool", "duration": 0.2}]}
    sample.tool_time = 0.2
    record = sample.to_dict()
    assert Sample.from_dict(record).to_dict() == record
    assert record["status"] == "completed"


def test_sampling_singleton_preserves_serving_contract():
    args = SimpleNamespace(hf_checkpoint="fake", sglang_server_concurrency=64,
        rollout_num_gpus=2, rollout_num_gpus_per_engine=1, rollout_temperature=0.6,
        rollout_top_p=0.95, rollout_top_k=20, rollout_max_response_len=8192,
        rollout_stop=None, rollout_stop_token_ids=None, rollout_skip_special_tokens=False)
    tokenizer = object()
    GenerateState.clear_instances()
    try:
        with patch("dualpd.runtime.generation.load_tokenizer", return_value=tokenizer), \
             patch("dualpd.runtime.generation.load_processor", return_value=None):
            state = GenerateState(args)
            assert state is GenerateState(args)
            assert state.tokenizer is tokenizer
            assert state.semaphore._value == 128
            assert state.sampling_params == dict(temperature=0.6, top_p=0.95, top_k=20,
                max_new_tokens=8192, stop=None, stop_token_ids=None, skip_special_tokens=False,
                no_stop_trim=True, spaces_between_special_tokens=False)
            assert not state.aborted and state.abort_epoch == 0
    finally:
        GenerateState.clear_instances()


def test_public_engine_module_identity_is_preserved():
    from sglang.srt.disaggregation import agentic_group_protocol
    module = agentic_group_protocol
    assert module.__name__ == "sglang.srt.disaggregation.agentic_group_protocol"
    assert module is sys.modules[module.__name__]
    assert Path(module.__file__).parent.name == "dualpd"


def test_workloads_import_with_slime_blocked(tmp_path):
    code = '''
import importlib.abc, sys
class BlockSlime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "slime" or fullname.startswith("slime."):
            raise AssertionError("standalone import tried to load " + fullname)
sys.meta_path.insert(0, BlockSlime())
import dualpd.inference, dualpd.data.browsecomp, dualpd.data.swe_bench_openenv
import sglang.srt.disaggregation.dualpd.router.late_binding_router
assert not any(n == "slime" or n.startswith("slime.") for n in sys.modules)
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_standalone_root_and_legacy_config_are_compatible(tmp_path):
    spec = importlib.util.spec_from_file_location("standalone_multinode", ROOT / "scripts/cluster/multinode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((ROOT / "scripts/cluster/multinode.example.json").read_text())
    config["dualpd_root"] = config.pop("slime_root")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    loaded = module.load_config(path)
    assert loaded["slime_root"] == loaded["dualpd_root"]
    assert module.plan(loaded)["workers"]


def test_flat_project_has_direct_entrypoints_without_compatibility_layers():
    assert not (ROOT / "examples").exists()
    assert not (ROOT / "src").exists()
    assert (ROOT / "inference.py").is_file()
    assert (ROOT / "runs-host").is_dir()
    for path in ("scripts/new_method/run_4p4d_numa_case.sh", "scripts/baseline/run_pd_case.sh",
                 "configs/experiments/swe_bench_verified_openenv_structured_tool_8k_t64_500_apptainer.yaml"):
        assert (ROOT / path).is_file()


@pytest.mark.parametrize("entry", [
    "scripts/new_method/internal/launch_late_binding_router.py",
    "scripts/new_method/internal/launch_mamba_late_binding_router.py",
])
def test_router_entry_preserves_overlay_without_new_router_namespace(entry, tmp_path):
    code = '''
import importlib.abc, runpy, sys, tempfile
from pathlib import Path
import sglang.srt.disaggregation
overlay = sglang.srt.disaggregation
overlay_paths = list(overlay.__path__)
paths = list(sys.path)
class OldOverlay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sglang.srt.disaggregation.dualpd" or fullname.startswith("sglang.srt.disaggregation.dualpd."):
            raise ModuleNotFoundError("old overlay has no Router package: " + fullname)
sys.meta_path.insert(0, OldOverlay())
entry = sys.argv[1]
sys.argv = [entry, "--help"]
try:
    runpy.run_path(entry, run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("launcher did not process --help")
# Torch may append its own generated /tmp compilation directory on import.
# No original search path may be removed/reordered, and no source checkout
# may be added in place of the selected overlay.
assert [p for p in sys.path if p in paths] == paths
assert all(Path(p).is_relative_to(tempfile.gettempdir()) for p in sys.path if p not in paths)
assert sys.modules["sglang.srt.disaggregation"] is overlay
assert list(overlay.__path__) == overlay_paths
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("DUALPD_ENGINE_ROOT", None)
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / entry)],
                            cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-3000:]
