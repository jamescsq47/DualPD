"""Stable entrypoints for the extracted serving project."""
import argparse
import importlib
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(prog="dualpd")
    parser.add_argument("command", choices=("inference", "router", "multinode", "slurm", "doctor"))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    root = Path(__file__).resolve().parents[1]
    if args.command == "doctor":
        modules = ["dualpd.runtime.types", "dualpd.runtime.generation", "dualpd.runtime.workloads.config",
                   "dualpd.data.browsecomp", "dualpd.data.swe_bench_verified"]
        for name in modules:
            module = importlib.import_module(name)
            print(f"{name}: {module.__file__}")
        if any(name == "slime" or name.startswith("slime.") for name in sys.modules):
            raise SystemExit("unexpected slime dependency loaded")
        print("OK: standalone workload imports; no slime modules loaded")
        return
    modules = {"inference": "dualpd.runtime.inference"}
    if args.command in modules:
        sys.argv = [f"dualpd {args.command}", *rest]
        runpy.run_module(modules[args.command], run_name="__main__")
        return
    scripts = {"router": root / "runtime/launch/local/launch_late_binding_router.py",
               "multinode": root / "runtime/launch/cluster/multinode.py",
               "slurm": root / "runtime/launch/cluster/qwen35_slurm.py"}
    path = scripts[args.command]
    if not path.is_file():
        raise SystemExit(f"missing runtime component: {path}; reinstall DualPD")
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *rest]
    runpy.run_path(str(path), run_name="__main__")
