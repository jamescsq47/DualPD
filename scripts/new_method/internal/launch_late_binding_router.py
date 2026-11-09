"""Deployment entrypoint; the Router implementation belongs to sglang_pd."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    helper = Path(__file__).resolve().parents[3] / "runtime/engine_entry.py"
    runpy.run_path(str(helper))["launch_router"]("current")
