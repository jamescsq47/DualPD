"""Actual old/candidate config parsers; no sockets, workers or GPU processes."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "scripts/cluster"
BASE = "0ed608d2906ed0f3b1b0b27f33e38b9ebda75495"
sys.path.insert(0, str(CLUSTER))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD_SOURCE = subprocess.check_output(["git", "-C", str(ROOT), "show",
                                      BASE + ":scripts/cluster/multinode.py"], text=True)
with tempfile.TemporaryDirectory(prefix="dualpd-old-port-parser-") as directory:
    path = Path(directory) / "multinode.py"
    path.write_text(OLD_SOURCE)
    OLD = load("frozen_port_parser", path)
NEW = load("maintained_port_parser", CLUSTER / "multinode.py")
EXAMPLE = json.loads((CLUSTER / "multinode.example.json").read_text())


def fixture(*, samehost=True):
    cfg = copy.deepcopy(EXAMPLE)
    cfg["tp_size"] = 2
    p, d = cfg["nodes"]
    p.update(host_ip="10.0.0.1", gpus=[0, 1], port=30000,
             bootstrap_port=8998, reverse_bootstrap_port=61000)
    d.update(host_ip="10.0.0.1" if samehost else "10.0.0.2",
             gpus=[2, 3] if samehost else [0, 1], port=30001,
             bootstrap_port=8999, reverse_bootstrap_port=61001)
    cfg["group_control"]["advertise_host"] = "10.0.0.1"
    return cfg


def parse(module, cfg):
    # Real load_config, with only its JSON file input supplied in memory.
    with patch.object(Path, "read_text", return_value=json.dumps(cfg)):
        return module.load_config("/unused/port-validation-fixture.json")


class SameHostListenerValidation(unittest.TestCase):
    def test_complete_candidate_is_only_three_reviewed_validation_changes(self):
        original = subprocess.check_output(["git", "-C", str(CLUSTER.parents[1]),
                                            "show", BASE + ":scripts/cluster/multinode.py"], text=True)
        changes = [
            ('    occupied = {router_node[k] for k in worker_ports}\n',
             '    occupied = {n[k] for n in nodes if n["host_ip"] == router_node["host_ip"]\n                for k in worker_ports}\n'),
            ('    local_ports = {relay_node[k] for k in worker_ports}\n',
             '    local_ports = {n[k] for n in nodes if n["host_ip"] == relay_node["host_ip"]\n                   for k in worker_ports}\n'),
            ('    if control["node_id"] == router["node_id"]:\n        local_ports.update',
             '    if relay_node["host_ip"] == router_node["host_ip"]:\n        local_ports.update'),
        ]
        for before, after in changes:
            self.assertEqual(original.count(before), 1)
            original = original.replace(before, after)
        self.assertEqual((CLUSTER / "multinode.py").read_text(), original)

    def test_valid_samehost_configuration_is_unchanged(self):
        cfg = fixture()
        self.assertEqual(parse(OLD, cfg), parse(NEW, cfg))

    def test_router_rejects_every_other_samehost_worker_listener(self):
        for router_field in ("port", "metrics_port"):
            for worker_field in ("port", "bootstrap_port", "reverse_bootstrap_port"):
                with self.subTest(router=router_field, worker=worker_field):
                    cfg = fixture()
                    cfg["router"][router_field] = cfg["nodes"][1][worker_field]
                    parse(OLD, cfg)  # Reproduces the original erroneous acceptance.
                    with self.assertRaisesRegex(ValueError, "router port collides"):
                        parse(NEW, cfg)

    def test_relay_rejects_every_other_samehost_worker_listener(self):
        for worker_field in ("port", "bootstrap_port", "reverse_bootstrap_port"):
            with self.subTest(worker=worker_field):
                cfg = fixture()
                cfg["group_control"]["port"] = cfg["nodes"][1][worker_field]
                parse(OLD, cfg)
                with self.assertRaisesRegex(ValueError, "group_control.port collides"):
                    parse(NEW, cfg)

    def test_relay_rejects_samehost_router_on_other_logical_group(self):
        for field in ("port", "metrics_port"):
            with self.subTest(router=field):
                cfg = fixture()
                cfg["group_control"]["node_id"] = cfg["nodes"][1]["node_id"]
                cfg["group_control"]["port"] = cfg["router"][field]
                parse(OLD, cfg)
                with self.assertRaisesRegex(ValueError, "group_control.port collides"):
                    parse(NEW, cfg)

    def test_distinct_physical_hosts_can_reuse_ports(self):
        cases = []
        cfg = fixture(samehost=False)
        for key in ("port", "bootstrap_port", "reverse_bootstrap_port"):
            cfg["nodes"][1][key] = cfg["nodes"][0][key]
        cases.append(cfg)
        for field in ("port", "metrics_port"):
            cfg = fixture(samehost=False)
            cfg["router"][field] = cfg["nodes"][1]["port"]
            cases.append(cfg)
            cfg = fixture(samehost=False)
            cfg["group_control"].update(node_id=cfg["nodes"][1]["node_id"],
                                         advertise_host="10.0.0.2", port=cfg["router"][field])
            cases.append(cfg)
        for index, cfg in enumerate(cases):
            with self.subTest(case=index):
                self.assertEqual(parse(OLD, cfg), parse(NEW, cfg))

    def test_all_groups_on_same_physical_host_are_included(self):
        cfg = fixture()
        extra = copy.deepcopy(cfg["nodes"][1])
        extra.update(node_id="decode-extra", engine_id="decode-extra", gpus=[4, 5],
                     port=30101, bootstrap_port=9019, reverse_bootstrap_port=61101)
        cfg["nodes"].append(extra)
        self.assertEqual(parse(OLD, cfg), parse(NEW, cfg))
        cfg["router"]["metrics_port"] = extra["bootstrap_port"]
        parse(OLD, cfg)
        with self.assertRaisesRegex(ValueError, "router port collides"):
            parse(NEW, cfg)

    def test_existing_rejections_remain_unchanged(self):
        for field in ("worker", "router", "relay", "router_self"):
            with self.subTest(field=field):
                cfg = fixture()
                if field == "worker":
                    cfg["nodes"][1]["port"] = cfg["nodes"][0]["port"]
                elif field == "router":
                    cfg["router"]["port"] = cfg["nodes"][0]["port"]
                elif field == "relay":
                    cfg["group_control"]["port"] = cfg["nodes"][0]["port"]
                else:
                    cfg["router"]["metrics_port"] = cfg["router"]["port"]
                errors = []
                for module in (OLD, NEW):
                    with self.assertRaises(ValueError) as error:
                        parse(module, cfg)
                    errors.append(str(error.exception))
                self.assertEqual(*errors)

    def test_third_group_relay_conflicts_with_swapped_logical_owners(self):
        for router_index, relay_index in ((0, 1), (1, 0)):
            for field in ("port", "bootstrap_port", "reverse_bootstrap_port"):
                with self.subTest(router=router_index, relay=relay_index, worker=field):
                    cfg = fixture()
                    extra = copy.deepcopy(cfg["nodes"][1])
                    extra.update(node_id="decode-extra", engine_id="decode-extra", gpus=[4, 5],
                                 port=30101, bootstrap_port=9019, reverse_bootstrap_port=61101)
                    cfg["nodes"].append(extra)
                    cfg["router"]["node_id"] = cfg["nodes"][router_index]["node_id"]
                    cfg["group_control"].update(node_id=cfg["nodes"][relay_index]["node_id"],
                                                 port=extra[field])
                    parse(OLD, cfg)
                    with self.assertRaisesRegex(ValueError, "group_control.port collides"):
                        parse(NEW, cfg)

    def test_nine_valid_model_family_tp_plans_are_unchanged(self):
        for family, sizes in (("qwen3", (1, 2, 4, 8)), ("qwen35_moe", (1, 2, 4, 8)), ("minimax_m2", (8,))):
            for tp in sizes:
                with self.subTest(family=family, tp=tp):
                    cfg = fixture(samehost=False)
                    cfg.update(model_family=family, tp_size=tp)
                    if family == "minimax_m2":
                        cfg["ep_size"] = 8
                    for node in cfg["nodes"]:
                        node["gpus"] = list(range(tp))
                    old = parse(OLD, cfg)
                    new = parse(NEW, cfg)
                    self.assertEqual(old, new)
                    self.assertEqual(OLD.plan(old), NEW.plan(new))


if __name__ == "__main__":
    unittest.main(verbosity=2)
