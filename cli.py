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
    root = Path(__file__).resolve().parent
    if args.command == "doctor":
        modules = ["dualpd.runtime.types", "dualpd.runtime.generation", "dualpd.data.config",
                   "dualpd.data.browsecomp", "dualpd.data.swe_bench_openenv"]
        for name in modules:
            module = importlib.import_module(name)
            print(f"{name}: {module.__file__}")
        if any(name == "slime" or name.startswith("slime.") for name in sys.modules):
            raise SystemExit("unexpected slime dependency loaded")
        print("OK: standalone workload imports; no slime modules loaded")
        return
    modules = {"inference": "dualpd.inference", "router": "dualpd.late_binding_router"}
    if args.command in modules:
        sys.argv = [f"dualpd {args.command}", *rest]
        runpy.run_module(modules[args.command], run_name="__main__")
        return
    scripts = {"multinode": root / "scripts/cluster/multinode.py",
               "slurm": root / "scripts/cluster/qwen35_slurm.py"}
    path = scripts[args.command]
    if not path.is_file():
        raise SystemExit("cluster launch commands require an editable DualPD checkout: pip install -e /path/to/DualPD")
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *rest]
    runpy.run_path(str(path), run_name="__main__")
