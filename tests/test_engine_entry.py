"""Router address selection does not replace an explicitly pinned engine."""
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dualpd.runtime import engine_entry


@pytest.mark.parametrize("kind,filename", list(engine_entry.ROUTERS.items()))
def test_existing_source_keeps_exact_file_entry(kind, filename, tmp_path):
    with patch.object(engine_entry, "DEFAULT_ENGINE", tmp_path), \
            patch.dict(engine_entry.os.environ, {}, clear=True), \
            patch.object(engine_entry.importlib.util, "find_spec") as lookup, \
            patch.object(engine_entry.runpy, "run_path") as run:
        engine_entry.launch_router(kind)
    lookup.assert_not_called()
    run.assert_called_once_with(str(tmp_path / "python/sglang/srt/disaggregation/dualpd/router" / filename), run_name="__main__")


@pytest.mark.parametrize("override", ["", "/missing/pinned-engine"])
def test_explicit_invalid_override_never_falls_back(override, tmp_path):
    error = FileNotFoundError("original file error")
    with patch.object(engine_entry, "DEFAULT_ENGINE", tmp_path), \
            patch.dict(engine_entry.os.environ, {"DUALPD_ENGINE_ROOT": override}, clear=True), \
            patch.object(engine_entry.importlib.util, "find_spec") as lookup, \
            patch.object(engine_entry.runpy, "run_path", side_effect=error) as run:
        with pytest.raises(FileNotFoundError) as caught:
            engine_entry.launch_router("current")
    assert caught.value is error
    lookup.assert_not_called()
    run.assert_called_once_with(str(Path(override) / "python/sglang/srt/disaggregation/dualpd/router/launch_late_binding_router.py"), run_name="__main__")


@pytest.mark.parametrize("kind,filename", list(engine_entry.ROUTERS.items()))
def test_wheel_uses_selected_package_location(kind, filename, tmp_path):
    origin = tmp_path / "selected/sglang/srt/disaggregation/__init__.py"
    with patch.object(engine_entry, "DEFAULT_ENGINE", tmp_path / "absent"), \
            patch.dict(engine_entry.os.environ, {}, clear=True), \
            patch.object(engine_entry.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(origin))) as lookup, \
            patch.object(engine_entry.runpy, "run_path") as run:
        engine_entry.launch_router(kind)
    lookup.assert_called_once_with("sglang.srt.disaggregation")
    run.assert_called_once_with(str(origin.parent / "dualpd/router" / filename), run_name="__main__")


@pytest.mark.parametrize("spec", [None, SimpleNamespace(origin=None)])
def test_missing_selected_package_fails_closed(spec, tmp_path):
    with patch.object(engine_entry, "DEFAULT_ENGINE", tmp_path / "absent"), \
            patch.dict(engine_entry.os.environ, {}, clear=True), \
            patch.object(engine_entry.importlib.util, "find_spec", return_value=spec), \
            patch.object(engine_entry.runpy, "run_path") as run:
        with pytest.raises(RuntimeError, match="no disaggregation package"):
            engine_entry.launch_router("current")
    run.assert_not_called()


def test_unknown_kind_does_not_lookup_or_execute():
    with patch.object(engine_entry.importlib.util, "find_spec") as lookup, \
            patch.object(engine_entry.runpy, "run_path") as run:
        with pytest.raises(KeyError):
            engine_entry.launch_router("unrecognized")
    lookup.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("entry,filename", [
    ("launch_late_binding_router.py", "launch_late_binding_router.py"),
    ("launch_mamba_late_binding_router.py", "legacy_mamba_router.py"),
])
def test_direct_script_needs_no_installed_frontend(entry, filename, tmp_path):
    root = Path(__file__).resolve().parents[1]
    selected = tmp_path / "explicit-engine"
    target = selected / "python/sglang/srt/disaggregation/dualpd/router" / filename
    target.parent.mkdir(parents=True)
    target.write_text('import json,sys\nprint(json.dumps({"argv":sys.argv, '
                      '"frontend_loaded":any(n=="dualpd" or n.startswith("dualpd.") for n in sys.modules)}))\n')
    env = dict(os.environ, DUALPD_ENGINE_ROOT=str(selected))
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, '-I', '-S', str(root / 'scripts/new_method/internal' / entry), '--sentinel'],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed == {'argv': [str(target), '--sentinel'], 'frontend_loaded': False}
