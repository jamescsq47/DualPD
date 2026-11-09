"""Whole-module immutable proof for maintained BrowseComp benchmark extraction.

The fixed recipe specializes only the accepted c256/Qwen3-8B path. It folds
successive keyword updates, preserving first insertion order and last values;
discarded expressions are pure Path/string/environment reads. This does not
prove arbitrary replacement os.environ objects or concurrently mutated inputs.
Runtime A/B and failure fixtures independently complement this static proof.
"""
import ast
import copy
from functools import lru_cache
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "152a620edbefef4b53a65fb9150c0d6b3fb10aa4"
MATRIX = "scripts/cluster/matrix_controller.py"
LOWC = "scripts/cluster/browsecomp_qwen3_8b_lowc_h100_matrix.py"
RUNNER = "scripts/cluster/engine_cleanup_quickcheck.py"
RECIPE = "scripts/benchmarks/browsecomp_qwen3_8b_h100.py"
MEASUREMENT = "scripts/benchmarks/measurement.py"
FUNCTIONS = ("gpu_processes", "configured_ports", "occupied_ports", "memory_events",
             "mean", "ensure_analysis", "summarize")


@lru_cache(maxsize=None)
def original(relative):
    return subprocess.check_output([
        "git", "-C", str(ROOT), "show", BASE + ":" + relative,
    ], text=True)


def function(source, name):
    return copy.deepcopy(next(node for node in ast.parse(source).body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def value(source, name):
    return copy.deepcopy(next(node.value for node in ast.parse(source).body
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(t, ast.Name) and t.id == name for t in node.targets)))


def dump(node):
    return ast.dump(node, include_attributes=False)


def module_dump(node):
    node = copy.deepcopy(node)
    # Only the leading module documentation is excluded. Function docstrings,
    # executable expressions, imports, annotations and guards remain checked.
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        node.body.pop(0)
    return dump(node)


def parsed(source):
    return ast.parse(source).body


class RecipeBindings(ast.NodeTransformer):
    def visit_Attribute(self, node):
        replacement = {
            "base.PD": "root", "base.PD_ENV": "ANALYSIS_ENV",
            "recipe.base.PD_ENV": "ANALYSIS_ENV", "env_bin.parent": "ANALYSIS_ENV",
        }.get(ast.unparse(node))
        return ast.parse(replacement, mode="eval").body if replacement else self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "concurrency":
            return ast.Constant(256)
        names = {"ROOT": "root", "PD": "root", "ENGINE": "engine", "PYTHON": "python"}
        if node.id in names:
            return ast.Name(names[node.id], node.ctx)
        if node.id == "cuda_lib":
            return ast.parse('ANALYSIS_ENV / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib"', mode="eval").body
        return node


class AnalysisBindings(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if dump(node) == dump(ast.parse('PD_ENV / "bin/python"', mode="eval").body):
            return ast.Name("python", ast.Load())
        return self.generic_visit(node)

    def visit_Name(self, node):
        return ast.Name("root", node.ctx) if node.id in {"PD", "SLIME"} else node


def expected_recipe(matrix, lowc, runner):
    bindings = {}
    for source, name in ((matrix, "common_environment"), (lowc, "common_environment"),
                         (lowc, "modified_environment"), (runner, "environment")):
        updates = [node.value for node in function(source, name).body
                   if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                   and dump(node.value.func) == dump(ast.parse("env.update", mode="eval").body)]
        assert len(updates) == 1 and not updates[0].args
        for keyword in updates[0].keywords:
            assert keyword.arg is not None
            if keyword.arg in bindings:
                # Previously evaluated but superseded values may not perform
                # I/O, arbitrary calls, mutation or user-defined computation.
                for call in (n for n in ast.walk(bindings[keyword.arg]) if isinstance(n, ast.Call)):
                    assert ast.unparse(call.func) in {"str", "os.environ.get"}
            bindings[keyword.arg] = copy.deepcopy(keyword.value)
    body = parsed("import os\nfrom pathlib import Path\n")
    body += [ast.Assign([ast.Name("ANALYSIS_ENV", ast.Store())], value(matrix, "PD_ENV")),
             ast.Assign([ast.Name("DATA", ast.Store())], ast.BinOp(value(matrix, "SCRATCH"), ast.Div(), ast.Constant("datasets"))),
             ast.Assign([ast.Name("MODEL", ast.Store())], value(lowc, "MODEL")),
             function(matrix, "clean_environment")]
    fn = parsed('''def environment(root: Path, engine: Path, python: Path, directory: Path):
    env = clean_environment()
    env.update()
    env.pop("HF_TOKEN", None)
    return env, ["bash", str(root / "scripts/new_method/run_4p4d_numa_case.sh")]
''')[0]
    fn.body[1].value.keywords = [ast.keyword(key, RecipeBindings().visit(expr))
                                 for key, expr in bindings.items()]
    body.append(fn)
    return ast.Module(body=body, type_ignores=[]), len(bindings)


def expected_measurement(matrix):
    body = parsed("import json\nfrom pathlib import Path\nimport socket\nimport statistics\nimport subprocess\n")
    for name in FUNCTIONS:
        fn = function(matrix, name)
        if name == "ensure_analysis":
            fn.args.kwonlyargs = [ast.arg("python", ast.Name("Path", ast.Load())),
                                   ast.arg("root", ast.Name("Path", ast.Load()))]
            fn.args.kw_defaults = [None, None]
            fn = AnalysisBindings().visit(fn)
        body.append(fn)
    return ast.Module(body=body, type_ignores=[])


class RunnerCalls(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "base"
                and dump(node.value) == dump(ast.parse("load_recipe().base", mode="eval").body)):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "base"):
            assert node.func.attr in {"memory_events", "gpu_processes", "occupied_ports",
                                      "configured_ports", "ensure_analysis", "summarize"}
            node.func.value.id = "measurement"
            if node.func.attr == "ensure_analysis":
                assert len(node.args) == 1 and not node.keywords
                node.keywords = [ast.keyword("python", ast.parse('recipe.ANALYSIS_ENV / "bin/python"', mode="eval").body),
                                 ast.keyword("root", ast.Name("ROOT", ast.Load()))]
        return self.generic_visit(node)


def expected_runner(source):
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "sys":
            continue
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "SCRIPT"):
            body.append(node)
            body.extend(parsed('''def load_benchmark_component(name):
    if name not in {"browsecomp_qwen3_8b_h100", "measurement"}:
        raise ValueError("unknown maintained benchmark component")
    path = ROOT / "scripts/benchmarks" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_dualpd_benchmark_" + name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load maintained benchmark component: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
if Path(__file__).resolve() != SCRIPT.resolve():
    raise RuntimeError("selected checkout does not match controller source")
recipe = load_benchmark_component("browsecomp_qwen3_8b_h100")
measurement = load_benchmark_component("measurement")
def validate_component_sources():
    for actual, relative in (
        (__file__, "scripts/cluster/engine_cleanup_quickcheck.py"),
        (recipe.__file__, "scripts/benchmarks/browsecomp_qwen3_8b_h100.py"),
        (measurement.__file__, "scripts/benchmarks/measurement.py"),
    ):
        if Path(actual).resolve() != (ROOT / relative).resolve():
            raise RuntimeError(f"benchmark component outside selected checkout: {relative}")
'''))
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name == "load_recipe":
                continue
            if node.name == "environment":
                node = parsed("def environment(directory):\n    return recipe.environment(ROOT, ENGINE, PYTHON, directory)\n")[0]
            elif node.name == "source_state":
                node.body.insert(0, parsed("validate_component_sources()\n")[0])
            elif node.name in {"resources", "run_on_node"}:
                node = RunnerCalls().visit(node)
        body.append(node)
    tree.body = body
    return tree


def verify(root=ROOT):
    matrix, lowc, runner = (original(path) for path in (MATRIX, LOWC, RUNNER))
    expected, bindings = expected_recipe(matrix, lowc, runner)
    checks = [(RECIPE, expected), (MEASUREMENT, expected_measurement(matrix)),
              (RUNNER, expected_runner(runner))]
    for relative, tree in checks:
        assert module_dump(ast.parse((root / relative).read_text())) == module_dump(tree), \
            (relative, "whole module differs beyond immutable extraction")
    for relative, source in ((MATRIX, matrix), (LOWC, lowc)):
        assert (root / relative).read_text() == source, (relative, "historical replay changed")
    return {"base_commit": BASE, "complete_runtime_modules_verified": 3,
            "unchanged_historical_modules_verified": 2,
            "final_keyword_bindings_from_four_layers": bindings,
            "measurement_functions_verified": len(FUNCTIONS),
            "controller_monitor_stop_cleanup_bodies_unchanged": True,
            "scope": "Fixed accepted regression only; source/configuration equivalence, not GPU readiness/performance, actualTCP/TP2/hybrid or complete stage7 archive migration."}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
