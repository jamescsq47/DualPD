"""Deployment entrypoint; the Router implementation belongs to sglang_pd."""
import os
from pathlib import Path
import runpy

if __name__ == "__main__":
    engine = Path(os.environ.get("DUALPD_ENGINE_ROOT", Path(__file__).resolve().parent.parent / "sglang_pd"))
    runpy.run_path(str(engine / "python/sglang/srt/disaggregation/dualpd/router/launch_late_binding_router.py"), run_name="__main__")
