from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE = Path(__file__).with_name("qwen3_30b_a3b_browsecomp_h100_matrix.py")


def load_module():
    spec = importlib.util.spec_from_file_location("qwen3_30b_a3b_matrix_test_target", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_has_exact_serial_order_and_model_override() -> None:
    module = load_module()
    expected = [
        ("browsecomp32b", mode, concurrency)
        for mode in module.MODES
        for concurrency in module.CONCURRENCIES
    ]
    assert module.CASES == expected
    assert len(module.CASES) == 12
    assert module.CONCURRENCIES == (192, 256, 320)
    assert module.MODES == ("new_method", "colocated", "no_reverse", "native_mooncake")
    assert module.base.MODEL_32B == module.MODEL


def test_common_and_agentic_settings_match_original_matrix(tmp_path: Path) -> None:
    module = load_module()
    case = ("browsecomp32b", "new_method", 192)
    common = module.common_environment(case, tmp_path)
    assert common["MODEL_PATH"] == str(module.MODEL)
    assert common["REQUESTS"] == "680"
    assert common["PRESERVE_SOURCE_ORDER"] == "true"
    assert common["WARMUP_SECONDS"] == "300"
    assert common["MEASURE_SECONDS"] == "1200"
    assert common["MODEL_CONTEXT_LENGTH"] == "40960"
    assert common["MODEL_MAX_RESPONSE_LENGTH"] == "36864"
    assert "/usr/include/x86_64-linux-gnu" in common["CPATH"]
    assert "/usr/lib/x86_64-linux-gnu" in common["LIBRARY_PATH"]

    env, _ = module.base.modified_environment(case, tmp_path)
    assert env["PREFILL_GPU_GROUPS"] == "0,4"
    assert env["DECODE_GPU_GROUPS"] == "1,5;2,6;3,7"
    assert env["FAST_TOOL_THRESHOLD_SECONDS"] == "1"
    assert env["DIRECT_WAIT_SECONDS"] == "1"
    assert env["SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB"] == "256"
    assert env["P2D_HOST_ARENA_GIB_PER_P"] == "128"


def test_sbatch_is_pinned_to_node2801_only() -> None:
    sbatch = MODULE.with_name("qwen3_30b_a3b_browsecomp_h100.sbatch").read_text()
    assert "#SBATCH --nodelist=node2801" in sbatch
    assert "#SBATCH --nodes=1" in sbatch
    assert "#SBATCH --gres=gpu:h100:8" in sbatch
    assert "node180" not in sbatch
    assert "node270" not in sbatch
