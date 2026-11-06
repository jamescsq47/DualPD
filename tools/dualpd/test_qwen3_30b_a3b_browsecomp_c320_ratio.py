from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


SOURCE = Path(__file__).with_name("qwen3_30b_a3b_browsecomp_c320_ratio_matrix.py")


def load(ratio: str):
    os.environ["PD_RATIO"] = ratio
    name = f"ratio_matrix_{ratio}"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ratio_layouts_and_fixed_host_totals():
    expected = {
        "2p2d": ("0,4;1,5", "2,6;3,7", 4),
        "3p1d": ("0,4;1,5;2,6", "3,7", 6),
    }
    for ratio, (p_groups, d_groups, p_ranks) in expected.items():
        module = load(ratio)
        case = ("browsecomp32b", "new_method", 320)
        env, _ = module.case_environment(case, module.run_dir(case))
        assert env["MAX_INFLIGHT"] == "320"
        assert env["PREFILL_GPU_GROUPS"] == p_groups
        assert env["DECODE_GPU_GROUPS"] == d_groups
        assert float(env["SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB"]) * p_ranks == pytest.approx(512)
        assert float(env["SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB"]) * p_ranks == pytest.approx(256)
        assert env["FAST_TOOL_THRESHOLD_SECONDS"] == "1"
        assert env["DIRECT_WAIT_SECONDS"] == "1"
        assert env["PD_LATE_BIND_NUMA_DOMAINS"] == "1"
        assert env["SGLANG_PD_LATE_BIND_DYNAMIC_PREFILL_DOMAINS"] == "1"


def test_colocated_remains_four_tp2_replicas():
    for ratio in ("2p2d", "3p1d"):
        module = load(ratio)
        case = ("browsecomp32b", "colocated", 320)
        env, _ = module.case_environment(case, module.run_dir(case))
        assert env["MODEL_GPU_GROUPS"] == "0,4;1,5;2,6;3,7"
        assert env["MODEL_TP_SIZE"] == "2"


def test_no_reverse_uses_requested_ratio_layout():
    for ratio, expected in {
        "2p2d": ("0,4;1,5", "2,6;3,7"),
        "3p1d": ("0,4;1,5;2,6", "3,7"),
    }.items():
        module = load(ratio)
        case = ("browsecomp32b", "no_reverse", 320)
        env, command = module.case_environment(case, module.run_dir(case))
        assert env["CASE_MODE"] == "no_reverse"
        assert env["PREFILL_GPU_GROUPS"] == expected[0]
        assert env["DECODE_GPU_GROUPS"] == expected[1]
        assert command[-1].endswith("run_pd_case.sh")


def test_agentic_launcher_accepts_more_p_than_d():
    launcher = (
        Path(__file__).parents[2]
        / "examples/pd/scripts/new_method/internal/run_pd_servers.sh"
    ).read_text()
    assert "DECODE GPU groups must split evenly" not in launcher
    assert 'domain="$((index % ${#prefill_gpu_groups[@]}))"' in launcher

    router = (
        Path(__file__).parents[2] / "examples/pd/late_binding_router.py"
    ).read_text()
    assert "and not self.global_decode" in router
    assert "to share arbitrary D counts across P domains" in router


import pytest
