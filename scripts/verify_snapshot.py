"""Read-only source equivalence and package independence checks.

Use the original sibling snapshots as optional migration references, never as
runtime dependencies. Exit nonzero on an unexpected engine/source difference.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ENGINE = WORKSPACE / "sglang_pd"


def main():
    source_map = json.loads((ENGINE / "dualpd-source-map.json").read_text())
    relocations = {item["original"]: item["path"] for item in source_map["files"]}
    files = subprocess.check_output(
        ["git", "-C", str(WORKSPACE / "sglang-src/pd"), "ls-files", "-z"]
    ).decode().split("\0")
    allowed = {"python/sglang/srt/mem_cache/hicache_storage.py",
               "python/sglang/srt/disaggregation/AGENTS.md"}
    reviewed_refactors = {item["original"] for item in source_map["files"]
                         if "current_sha256" in item}
    refactor_validation = None
    if reviewed_refactors:
        validation = subprocess.run(
            [sys.executable, str(ENGINE / source_map["refactor_validation"])],
            capture_output=True, text=True, check=True,
        )
        refactor_validation = json.loads(validation.stdout)
    equal = 0
    differences = []
    for name in filter(None, files):
        old = WORKSPACE / "sglang-src/pd" / name
        new = ENGINE / relocations.get(name, name)
        if not new.is_file():
            differences.append([name, "missing"])
        elif old.read_bytes() == new.read_bytes():
            equal += 1
        elif name not in allowed and name not in reviewed_refactors:
            differences.append([name, "unexpected change"])
    for item in source_map["files"]:
        expected_hash = item.get("current_sha256", item["sha256"])
        if hashlib.sha256((ENGINE / item["path"]).read_bytes()).hexdigest() != expected_hash:
            differences.append([item["path"], "relocated content changed"])
    router_map = json.loads((ENGINE / "router-source-map.json").read_text())
    for item in router_map["files"]:
        if hashlib.sha256((ENGINE / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            differences.append([item["path"], "router content changed"])
    # All these serving paths must retain their complete implementation; only
    # their import namespaces are remapped during extraction.
    core = ["inference.py", "agentic_kv_request.py", "late_binding_router.py",
            "pd_metrics.py", "model_http_transport.py", "data/browsecomp/agent.py",
            "data/browsecomp/env.py", "data/swe_bench_openenv/harness.py",
            "data/swe_bench/harness.py", "data/swe_bench/verifier.py",
            "data/swe_bench/apptainer_backend.py", "data/retool/harness.py"]
    def without_imports(path):
        tree = ast.parse(path.read_text())
        # Package location constants necessarily change when flattening.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"REPO_ROOT", "WORKSPACE_ROOT"}
                for t in node.targets
            )
        )]
        class RemoveImports(ast.NodeTransformer):
            def visit_Import(self, node):
                return None
            def visit_ImportFrom(self, node):
                return None
        return ast.dump(RemoveImports().visit(tree), include_attributes=False)
    for name in core:
        old = WORKSPACE / "slime/examples/pd" / name
        if name == "late_binding_router.py":
            new = ENGINE / "python/sglang/srt/disaggregation/dualpd/router" / name
        elif name in {"agentic_kv_request.py", "pd_metrics.py", "model_http_transport.py"}:
            new = ROOT / "runtime" / name
        else:
            new = ROOT / name
        old_ast = without_imports(old)
        new_ast = without_imports(new)
        # Flattening also updates one docstring and one installation hint.
        # Normalize only these two literal references, not executable logic.
        text_paths = {
            "data/retool/harness.py": "examples/pd/inference.py",
            "data/swe_bench/verifier.py": "examples/pd/requirements-swe-bench.txt",
        }
        if name in text_paths:
            old_ast = old_ast.replace(text_paths[name], text_paths[name].removeprefix("examples/pd/"))
        new_ast = new_ast.replace("dualpd.runtime.http_utils", "slime.utils.http_utils")
        if old_ast != new_ast:
            differences.append([name, "serving AST differs beyond imports"])
    result = {"engine_byte_identical_files": equal,
              "relocated_modules_and_tests": len(relocations),
              "router_implementations_hash_identical": len(router_map["files"]),
              "reviewed_refactor_equivalence": refactor_validation,
              "frontend_core_ast_identical_except_imports": len(core),
              "intentional_engine_changes": sorted(allowed), "differences": differences}
    print(json.dumps(result, indent=2))
    if differences:
        sys.exit(1)


if __name__ == "__main__":
    main()
