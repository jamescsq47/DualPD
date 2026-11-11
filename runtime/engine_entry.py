"""Locate engine-owned Router entrypoints without replacing native imports."""
import importlib.util
import os
from pathlib import Path
import runpy


DEFAULT_ENGINE = Path(__file__).resolve().parents[1] / "3rdparty/sglang"
ROUTERS = {
    "current": "launch_late_binding_router.py",
    "legacy_mamba": "legacy_mamba_router.py",
}


def launch_router(kind):
    filename = ROUTERS[kind]
    engine = Path(os.environ.get("DUALPD_ENGINE_ROOT", DEFAULT_ENGINE))
    if "DUALPD_ENGINE_ROOT" in os.environ or engine.exists():
        # Preserve source deployments, explicit pins, and their failure mode:
        # an invalid override must never silently choose a different engine.
        entry = engine / "python/sglang/srt/disaggregation/dualpd/router" / filename
    else:
        # A normal wheel installation has no sibling source checkout. Resolve
        # the engine selected by Python, not a guessed site-packages directory.
        spec = importlib.util.find_spec("sglang.srt.disaggregation")
        if spec is None or spec.origin is None:
            raise RuntimeError("selected SGLang installation has no disaggregation package")
        entry = Path(spec.origin).parent / "dualpd/router" / filename
    # Keep file-entry semantics for both branches. Historical overlays may lack
    # the new Router package; the engine entry handles its own private module
    # loading without changing the selected native SGLang overlay or sys.path.
    runpy.run_path(str(entry), run_name="__main__")
