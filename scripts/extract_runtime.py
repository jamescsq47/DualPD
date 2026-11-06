"""One-time, mechanical extraction of inference utilities from the source snapshot.

Only the new DualPD tree is written. Source provenance is recorded separately.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "slime/slime"
TARGET = ROOT / "src/dualpd/runtime"


def definitions(path, names):
    source = path.read_text()
    lines = source.splitlines(keepends=True)
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            selected.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(selected) + "\n"


def main():
    TARGET.mkdir(exist_ok=True)
    (TARGET / "types.py").write_text(
        '"""Inference sample schema, extracted unchanged from the source snapshot."""\n'
        'from __future__ import annotations\n'
        'from dataclasses import dataclass, field\nfrom enum import Enum\nfrom typing import Any\n\n'
        + definitions(SOURCE / "utils/types.py", {"Sample", "MultimodalType", "MultimodalTypes"})
    )
    data = (SOURCE / "utils/data.py").read_text().split("\ndef get_minimum_num_micro_batch_size", 1)[0]
    data = data.replace("import ray\n", "").replace("from .timer import Timer\n", "")
    (TARGET / "dataset.py").write_text(data)
    for name in ("processing_utils", "trace_utils"):
        (TARGET / f"{name}.py").write_text((SOURCE / f"utils/{name}.py").read_text())
    (TARGET / "http_requests.py").write_text(
        '"""Original local HTTP retry and response lifecycle; no training dispatch."""\n'
        'import asyncio\nimport json\nimport logging\nimport httpx\n'
        'logger = logging.getLogger(__name__)\n\n'
        + definitions(SOURCE / "utils/http_utils.py", {"_post", "get"})
    )
    # get() must observe the same live client as post() and model_http_transport.
    # Its globals remain in http_utils, so include the exact functions there.
    (TARGET / "http_utils.py").write_text((TARGET / "http_requests.py").read_text())
    (TARGET / "http_requests.py").unlink()
    trace = (SOURCE / "dashboard/api.py").read_text().split("\ndef metrics", 1)[0]
    (TARGET / "tracing.py").write_text(trace)

    replacements = {
        "slime.utils.types": "dualpd.runtime.types",
        "slime.utils.data": "dualpd.runtime.dataset",
        "slime.utils.processing_utils": "dualpd.runtime.processing_utils",
        "slime.utils.trace_utils": "dualpd.runtime.trace_utils",
        "slime.utils.http_utils": "dualpd.runtime.http_utils",
        "slime.dashboard.api": "dualpd.runtime.tracing",
        "slime.rollout.sglang_rollout": "dualpd.runtime.generation",
        "from slime.utils import http_utils": "from dualpd.runtime import http_utils",
    }
    # Preserve original flat entrypoints for existing launchers. The project
    # installs both src and the compatibility directory through its .pth file.
    for path in (ROOT / "src/dualpd").rglob("*.py"):
        if "runs-host" in path.parts:
            continue
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)


if __name__ == "__main__":
    main()
