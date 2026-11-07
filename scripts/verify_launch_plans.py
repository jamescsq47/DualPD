"""Compare old/new CPU-only plans with explicit path remapping."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cluster"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remap(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [remap(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(remap(item, replacements) for item in value)
    if isinstance(value, dict):
        return {key: remap(item, replacements) for key, item in value.items()}
    return value


def main():
    old = load("original_multinode_plan", ROOT.parent / "slime/tools/dualpd/multinode.py")
    new = load("standalone_multinode_plan", ROOT / "scripts/cluster/multinode.py")
    fixture = json.loads((ROOT / "scripts/cluster/multinode.example.json").read_text())
    fixture["slime_root"] = str(ROOT.parent / "slime")
    fixture["sglang_root"] = str(ROOT.parent / "sglang-src/pd")
    rows = []
    for family, sizes in [("qwen3", [1, 2, 4, 8]), ("qwen35_moe", [1, 2, 4, 8]), ("minimax_m2", [8])]:
        for tp in sizes:
            config = copy.deepcopy(fixture)
            config["model_family"] = family
            config["tp_size"] = tp
            if family == "minimax_m2":
                config["ep_size"] = 8
            for node in config["nodes"]:
                node["gpus"] = list(range(tp))
            migrated = copy.deepcopy(config)
            migrated["slime_root"] = str(ROOT)
            migrated["sglang_root"] = str(ROOT.parent / "sglang_pd")
            replacements = [(config["slime_root"] + "/examples/pd", migrated["slime_root"]),
                            (config["slime_root"] + "/tools/dualpd", migrated["slime_root"] + "/scripts/cluster"),
                            (config["slime_root"], migrated["slime_root"]),
                            (config["sglang_root"], migrated["sglang_root"])]
            expected = remap(old.plan(config), replacements)
            actual = new.plan(migrated)
            # Root paths are part of the immutable config, so its digest must
            # change; compare the complete commands/environments themselves.
            expected.pop("config_sha256", None)
            actual.pop("config_sha256", None)
            identical = expected == actual
            rows.append(dict(model_family=family, tp=tp, identical_except_paths=identical))
            if not identical:
                print(json.dumps(rows, indent=2))
                raise SystemExit("launch plan drift")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
