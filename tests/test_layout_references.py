"""Keep root cleanup and migrated source references from drifting back."""
import ast
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REMOVED_MODULES = {
    "agentic_kv_request", "pd_metrics", "model_http_transport", "cli",
    "generate_with_hybrid", "generate_with_retool", "browsecomp_agent",
    "browsecomp_env", "tool_sandbox", "search_server", "launch_late_binding_router",
}


def test_root_only_contains_public_entry_and_packaging_files():
    assert {path.name for path in ROOT.glob("*.py")} == {
        "inference.py", "dualpd.py", "__init__.py", "__main__.py", "setup.py",
    }


def test_relocated_modules_resolve_to_their_real_sources():
    from dualpd.runtime import agentic_kv_request, pd_metrics, model_http_transport, cli
    from dualpd.data import generation, dispatch

    for module in (agentic_kv_request, pd_metrics, model_http_transport, cli):
        assert Path(module.__file__).parent == ROOT / "runtime"
    assert Path(generation.__file__).parent == ROOT / "data"
    assert callable(generation.generate_unified)
    assert callable(dispatch.select_samples) and callable(dispatch.exact_counts)


def test_active_imports_do_not_reference_removed_root_modules():
    forbidden = {"dualpd." + name for name in REMOVED_MODULES}
    violations = []
    paths = [*ROOT.glob("*.py")]
    for directory in ("runtime", "data", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    names = [node.module or ""]
                if node.module == "dualpd" and node.level == 0:
                    names.extend("dualpd." + alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Catch importlib module names and monkeypatch target strings.
                names = [node.value]
            for name in names:
                if any(name == old or name.startswith(old + ".") for old in forbidden):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "\n".join(violations)


def test_shell_launchers_do_not_address_deleted_root_services():
    violations = []
    for path in (ROOT / "scripts").rglob("*"):
        if path.suffix not in {".sh", ".sbatch"}:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"\$\{(?:PD_DIR|PD)\}/(?:search_server|launch_late_binding_router)\.py", line):
                violations.append(f"{path.relative_to(ROOT)}:{number}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("arguments", [
    ["-m", "dualpd", "inference", "--help"],
    ["-m", "dualpd", "router", "--help"],
    ["-m", "dualpd", "multinode", "--help"],
    ["-m", "dualpd", "slurm", "--help"],
    ["-m", "dualpd.data.browsecomp.search_server", "--help"],
])
def test_relocated_command_entries_from_outside_checkout(arguments, tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, *arguments], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-3000:]
    assert "usage:" in result.stdout


def test_installed_console_entry_uses_relocated_cli(tmp_path):
    executable = Path(sys.executable).with_name("dualpd")
    assert executable.is_file(), "Install the current checkout with pip install -e ."
    assert "dualpd.runtime.cli" in executable.read_text()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([str(executable), "doctor"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-1000:] + result.stderr[-3000:]
    assert "no slime modules loaded" in result.stdout
