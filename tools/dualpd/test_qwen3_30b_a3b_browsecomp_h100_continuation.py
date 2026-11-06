from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE = Path(__file__).with_name("qwen3_30b_a3b_browsecomp_h100_continuation.py")


def load_module():
    spec = importlib.util.spec_from_file_location("qwen3_30b_a3b_continuation_test_target", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_eight_case_order() -> None:
    module = load_module()
    assert module.CONCURRENCIES == (128, 384)
    assert module.CASES == [
        ("browsecomp32b", mode, concurrency)
        for mode in module.MODES
        for concurrency in (128, 384)
    ]
    assert len(module.CASES) == 8


def test_report_covers_all_five_concurrencies() -> None:
    module = load_module()
    assert module.parent.REPORT_CONCURRENCIES == (128, 192, 256, 320, 384)
    assert module.parent.MODEL.name == "Qwen3-30B-A3B"


def test_continuation_avoids_our_active_nodes() -> None:
    sbatch = MODULE.with_name("qwen3_30b_a3b_browsecomp_h100_continuation.sbatch").read_text()
    assert "#SBATCH --nodelist=" not in sbatch
    assert "#SBATCH --exclude=node1800,node2700,node2701" in sbatch
    assert "#SBATCH --nodes=1" in sbatch
    assert "#SBATCH --gres=gpu:h100:8" in sbatch
