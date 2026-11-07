#!/usr/bin/env python3
"""Add c128 and c384 to the completed Qwen3-30B-A3B matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-30b-a3b-tp2-c128-c384-20261013"
)
ORIGINAL_CONTROL = Path(
    "/orcd/scratch/orcd/003/shangy/dualpd/runs/"
    "h100-browsecomp-qwen3-30b-a3b-tp2-12-matrix-20261012"
)
PARENT = Path(__file__).with_name("qwen3_30b_a3b_browsecomp_h100_matrix.py")
CONCURRENCIES = (128, 384)
MODES = ("new_method", "colocated", "no_reverse", "native_mooncake")
CASES = [("browsecomp32b", mode, concurrency) for mode in MODES for concurrency in CONCURRENCIES]


spec = importlib.util.spec_from_file_location("qwen3_30b_a3b_parent_matrix", PARENT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {PARENT}")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)

parent.CONTROL = CONTROL
parent.base.CONTROL = CONTROL
parent.base.CASES = CASES


def merged_states(continuation: dict[str, object]) -> dict[str, object]:
    original_path = ORIGINAL_CONTROL / "state.json"
    original = json.loads(original_path.read_text()) if original_path.exists() else {}
    original.update(continuation)
    return original


def atomic_json(path: Path, value: object) -> None:
    parent.original_atomic_json(path, value)
    if path == CONTROL / "state.json" and isinstance(value, dict):
        parent.render_report(merged_states(value))


parent.base.atomic_json = atomic_json


if __name__ == "__main__":
    existing_path = CONTROL / "state.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else {}
    parent.render_report(merged_states(existing))
    parent.base.main()
