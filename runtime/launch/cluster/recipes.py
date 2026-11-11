"""Model-specific SWE launch recipes, without historical allocation/watchdog state.

Only argv construction is shared here. Serving and run-scoped shutdown remain
owned by the existing launchers; native baselines never use the custom engine.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = ROOT / "configs" / "model-profiles"
MODULES = {
    "new_method": "dualpd.runtime.launch.cluster.qwen35_slurm",
    "no_reverse": "dualpd.runtime.launch.cluster.qwen35_native_pd_slurm",
    "hicache": "dualpd.runtime.launch.cluster.qwen35_native_pd_slurm",
    "colocated": "dualpd.runtime.launch.cluster.qwen35_colocated_slurm",
}


def option_value(arguments: list[str], flag: str) -> str | None:
    value = None
    for index, argument in enumerate(arguments):
        if argument.startswith(flag + "="):
            value = argument.split("=", 1)[1]
        elif argument == flag and index + 1 < len(arguments):
            value = arguments[index + 1]
    return value


def load_recipe(name: str) -> dict:
    if name not in {path.stem for path in RECIPE_DIR.glob("*.json")}:
        raise ValueError(f"unknown model recipe: {name}")
    return json.loads((RECIPE_DIR / f"{name}.json").read_text())


def build_commands(name: str, method: str, action: str, arguments: list[str]) -> list[list[str]]:
    recipe = load_recipe(name)
    if method not in MODULES:
        raise ValueError(f"unknown method: {method}")
    options = ["--workload-config", str(ROOT / "configs/experiments" / recipe["workload"])]
    options += recipe["common"]
    options += recipe["new_method" if method == "new_method" else "baseline"]
    if method != "new_method":
        options += recipe["colocated" if method == "colocated" else "native"]
        if method == "hicache":
            options += ["--native-mooncake", *recipe["hicache"]]
    options += arguments  # Explicit caller arguments override recipe defaults.
    if not option_value(arguments, "--run-id") and option_value(arguments, "--run-dir"):
        options += ["--run-id", Path(option_value(arguments, "--run-dir")).name]
    prefix = [sys.executable, "-m", MODULES[method]]
    if action != "launch":
        return [[*prefix, *options, action]]
    if method != "new_method":
        return [[*prefix, *options, phase] for phase in ("prepare", "start")]
    phases = (
        ("prepare", False), ("start-control", True), ("control-ready", False),
        ("start-prefill", True), ("start-decode", True), ("wait-ready", False),
        ("start-router", True), ("run-workload", True),
    )
    return [[*prefix, *options, *(["--detach"] if detach else []), phase]
            for phase, detach in phases]


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: recipes MODEL METHOD [launch|prepare|status|stop|...] [options]")
    name, method, *arguments = sys.argv[1:]
    recipe = load_recipe(name)
    if recipe.get("notes"):
        print(recipe["notes"], file=sys.stderr)
    if not arguments or arguments == ["--help"] or arguments == ["-h"]:
        print(f"{recipe['description']}\nMethod: {method}\n")
        print("Usage: bash scripts/<method>/<model-script>.sh [launch|prepare|status|stop|...] [options]")
        print("Baseline entries take no_reverse|hicache|colocated before the action.")
        print("launch prepares and starts one case; it does not monitor, stop at T450, or advance a matrix.")
        print("--dry-run prints commands without accessing Slurm, SSH, GPUs or files in a run directory.")
        print("Supply --job-id, --run-dir, --model, --data-root and --swe-sif-dir.")
        print("For baselines also supply --python and --sglang-root for a separate clean baseline environment.")
        print("Two-node HiCache also needs explicit --workload-job-id/--workload-node and --storage-job-id/--storage-node.")
        print(f"Recipe: configs/model-profiles/{name}.json\nSource: {recipe['source']}")
        print("Pass an action and --help for the underlying launcher's complete options.")
        return 0
    dry_run = "--dry-run" in arguments
    arguments = [arg for arg in arguments if arg != "--dry-run"]
    action = arguments.pop(0) if arguments and not arguments[0].startswith("-") else "launch"
    if action in {"prepare", "launch"} and not dry_run and "--help" not in arguments:
        required = ["--job-id", "--run-dir", "--model", "--data-root", "--swe-sif-dir"]
        if method != "new_method":
            required += ["--python", "--sglang-root"]
        if method == "hicache" and "two-node" in recipe["description"]:
            required += ["--workload-job-id", "--workload-node", "--storage-job-id", "--storage-node"]
        missing = [flag for flag in required
                   if flag not in arguments and not any(arg.startswith(flag + "=") for arg in arguments)]
        if missing:
            raise SystemExit("missing explicit launch paths/options: " + ", ".join(missing))
    for command in build_commands(name, method, action, arguments):
        print(shlex.join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
