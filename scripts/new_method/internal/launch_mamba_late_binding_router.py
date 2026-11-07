"""Deployment entrypoint for the explicitly pinned historical Mamba Router."""
import os
from pathlib import Path
import runpy

if __name__ == "__main__":
    engine = Path(os.environ.get("DUALPD_ENGINE_ROOT", Path(__file__).resolve().parents[4] / "sglang_pd"))
    runpy.run_path(str(engine / "python/sglang/srt/disaggregation/dualpd/router/legacy_mamba_router.py"), run_name="__main__")
