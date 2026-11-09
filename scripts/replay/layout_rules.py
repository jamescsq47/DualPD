"""Reviewed pure source-address/import rewrites for immutable replay proof.

Not imported by serving/launch code. ROOT is the recorded historical address
spelling, not a runtime project-discovery setting.
"""
import ast
from pathlib import Path
import re

ROOT = Path("/home/shangy/dualpd/DualPD")
CORE = {"process_supervisor", "qwen35_slurm", "multinode",
        "qwen35_native_pd_slurm", "qwen35_colocated_slurm",
        "dram_resource_sampler", "dram_complete_frame_analysis"}

def core_imports(source):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [(alias.asname or alias.name,
                      '_load_core(' + repr(alias.name) + ')')
                     for alias in node.names if alias.name in CORE]
            if names:
                assert len(names) == len(node.names), 'mixed core/noncore import needs explicit review'
        elif isinstance(node, ast.ImportFrom):
            if node.module in CORE and node.level == 0:
                names = [(alias.asname or alias.name,
                          '_load_core(' + repr(node.module) + ').' + alias.name)
                         for alias in node.names]
                assert all(alias.name != '*' for alias in node.names)
            elif node.module is None and node.level == 1:
                names = [(alias.asname or alias.name,
                          '_load_core(' + repr(alias.name) + ')')
                         for alias in node.names if alias.name in CORE]
                if names:
                    assert len(names) == len(node.names), 'mixed relative import needs explicit review'
        if names:
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            indent = ' ' * node.col_offset
            replacement = ('\n' + indent).join(name + ' = ' + value for name, value in names)
            edits.append((start, end, replacement))
    if not edits:
        return source
    assert '_load_core' not in source, 'new helper name collides'
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    body = tree.body
    anchor = 0
    for index, node in enumerate(body):
        if (index == 0 and isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            anchor = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == '__future__':
            anchor = node.end_lineno
        else:
            break
    bootstrap = (
        '\n# Resolve only maintained core/shared tools; historical siblings stay local.\n'
        'if __package__:\n'
        '    from ._core import load_core as _load_core\n'
        'else:\n'
        '    from _core import load_core as _load_core\n\n')
    # All replaced imports are below module documentation/future imports.
    return source[:offsets[anchor]] + bootstrap + source[offsets[anchor]:]

def relocate(relative, source, mapping):
    current = source
    if relative.endswith('.py'):
        current = current.replace('Path(__file__).resolve().parents[2]',
                                  'Path(__file__).resolve().parents[3]')
        current = current.replace('Path(__file__).parents[2]',
                                  'Path(__file__).parents[3]')
        if relative == 'scripts/new_method/run_qwen35_27b_tp2_swe500_sequence.py':
            current = current.replace('LAUNCHER.parents[2]', 'LAUNCHER.parents[3]')
    for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
        current = current.replace(str(ROOT / old), str(ROOT / new))
        # Match project-relative code addresses, never suffixes of external
        # /slime/... or immutable /scratch/... source snapshots.
        current = re.sub(r'(?<![\w/])' + re.escape(old) + r'(?![\w.])', new, current)
    if relative.startswith('scripts/new_method/') and relative.endswith('.sh'):
        current = current.replace('${SCRIPT_DIR}/../.."', '${SCRIPT_DIR}/../../.."')
        current = current.replace('${SCRIPT_DIR}/../../configs/', '${SCRIPT_DIR}/../../../configs/')
        for directory in ('internal',):
            current = current.replace('${SCRIPT_DIR}/' + directory + '/',
                                      '${PD_DIR}/scripts/new_method/' + directory + '/')
        for directory in ('tools', 'common', 'baseline'):
            current = current.replace('${SCRIPT_DIR}/../' + directory + '/',
                                      '${PD_DIR}/scripts/' + directory + '/')
        for name in ('run_4p4d_numa_case.sh', 'run_2p6d_numa_case.sh',
                     'run_1p3d_case.sh', 'run_1p5d_case.sh',
                     'run_2p6d_tp2_qwen3_32b.sh', 'run_4p4d_tp2_qwen3_32b.sh'):
            current = current.replace('${SCRIPT_DIR}/' + name,
                                      '${PD_DIR}/scripts/new_method/' + name)
        # Some small historical wrappers never needed a PD_DIR previously.
        if '${PD_DIR}' in current and not re.search(r'^PD_DIR=', current, re.M):
            rows = current.splitlines(keepends=True)
            positions = [i for i, line in enumerate(rows) if line.startswith('SCRIPT_DIR=')]
            assert len(positions) == 1
            rows.insert(positions[0] + 1,
                        'PD_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"\n')
            current = ''.join(rows)
    if relative.endswith('.py') and relative.startswith('scripts/cluster/'):
        current = core_imports(current)
    return current

def package_imports(source, siblings):
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    covered = [(node.lineno, node.end_lineno) for node in ast.walk(tree)
               if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
               and node.test.id == '__package__']
    edits = []
    def signature(nodes):
        result = []
        for node in nodes:
            if isinstance(node, ast.Import) and all(a.name in siblings for a in node.names):
                result.extend((a.name, a.asname) for a in node.names)
            elif (isinstance(node, ast.ImportFrom) and node.level in (0, 1)
                  and node.module is None and all(a.name in siblings for a in node.names)):
                result.extend((a.name, a.asname) for a in node.names)
            elif (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Name) and node.value.func.id == '_load_core'):
                result.append(('core', ast.dump(node, include_attributes=False)))
            else:
                return None
        return result
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or node.orelse or node.finalbody or len(node.handlers) != 1:
            continue
        handler = node.handlers[0]
        if not isinstance(handler.type, ast.Name) or handler.type.id != 'ImportError' or handler.name:
            continue
        body = signature(node.body)
        if body is None or not body or body != signature(handler.body):
            continue
        assert lines[node.lineno - 1].split('#', 1)[0].strip() == 'try:'
        assert lines[handler.lineno - 1].split('#', 1)[0].strip() == 'except ImportError:'
        edits += [(starts[node.lineno - 1], starts[node.lineno],
                   lines[node.lineno - 1].replace('try:', 'if __package__:', 1)),
                  (starts[handler.lineno - 1], starts[handler.lineno],
                   lines[handler.lineno - 1].replace('except ImportError:', 'else:', 1))]
        covered.append((node.lineno, node.end_lineno))
    for node in ast.walk(tree):
        if any(first <= getattr(node, 'lineno', -1) <= last for first, last in covered):
            continue
        original = None
        relative = None
        if isinstance(node, ast.Import) and any(a.name in siblings for a in node.names):
            assert all(a.name in siblings for a in node.names), 'mixed sibling import needs review'
            original = ast.get_source_segment(source, node)
            relative = 'from . import ' + ', '.join(
                a.name + (' as ' + a.asname if a.asname else '') for a in node.names)
        elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in siblings):
            original = ast.get_source_segment(source, node)
            relative = original.replace('from ' + node.module, 'from .' + node.module, 1)
        if original is not None:
            indent = ' ' * node.col_offset
            replacement = ('if __package__:\n' + indent + '    ' + relative + '\n'
                           + indent + 'else:\n' + indent + '    ' + original)
            edits.append((starts[node.lineno - 1] + node.col_offset,
                          starts[node.end_lineno - 1] + node.end_col_offset, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source
