"""Deployment entrypoint for the explicitly pinned historical Mamba Router."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    helper = Path(__file__).resolve().parents[3] / "runtime/engine_entry.py"
    runpy.run_path(str(helper))["launch_router"]("legacy_mamba")
