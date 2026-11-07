import importlib.util
from pathlib import Path
import sys


MODULE = Path(__file__).with_name(
    "qwen35_35b_a3b_single_node_1p3d_matrix_autopilot.py"
)


def load_module():
    sys.path.insert(0, str(MODULE.parent))
    spec = importlib.util.spec_from_file_location("qwen35_35b_a3b_matrix", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_identity_and_order():
    module = load_module()
    assert module.NODE == "node2800"
    assert module.JOB_ID == "25734937"
    assert module.MODEL.name == "Qwen3.5-35B-A3B"
    assert module.CONCURRENCIES == (64, 128, 192)
    assert len(module.EXPERIMENTS) == 12
    assert [item["method"] for item in module.EXPERIMENTS[:4]] == [
        "agentic",
        "colocated",
        "native",
        "hicache",
    ]
    assert [item["concurrency"] for item in module.EXPERIMENTS] == [
        64,
        64,
        64,
        64,
        128,
        128,
        128,
        128,
        192,
        192,
        192,
        192,
    ]
    assert all("q35-35b-a3b" in item["run_id"] for item in module.EXPERIMENTS)


def test_configure_isolated_from_node2700_matrix():
    module = load_module()
    module.configure()
    assert module.base.NODE == "node2800"
    assert module.base.CONTROL_DIR == module.CONTROL_DIR
    assert module.base.REPORT == module.REPORT
    assert module.base.EXPERIMENTS == module.EXPERIMENTS
    assert module.base.RESULT_EXPERIMENTS == module.EXPERIMENTS
    assert module.base.BASELINE_ENGINE.name == "pd_baseline_b1ef6_q38fix"
    assert module.base.HICACHE_ENGINE.name == "pd_baseline_b1ef6"
