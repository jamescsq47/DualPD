"""Complex replay configuration prefixes, never historical operational bodies.

Stop before the first mkdir. Docker, preflight Python, source snapshots, CUDA,
control-dir creation, process launch and launcher-specific cleanup suffixes are
not executed. Colocated installs the real runtime EXIT cleanup, which runs with
an asserted empty registry; no registered-service stop/kill path is exercised.
The
27B plugin directory is relocated ONLY in disposable copies; an empty fixture
file tests the existence guard, not plugin authenticity or native behavior.
Whole-file migration proof independently covers the unexecuted suffixes.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

from dualpd.scripts.verify_replay_layout import original

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    'run_qwen35_9b_tp1_browsecomp_4p4d.sh',
    'run_qwen35_9b_tp1_swe_openenv_4p4d_full.sh',
    'run_qwen35_9b_tp1_swe_openenv_4p4d_100.sh',
    'run_qwen35_9b_tp1_swe_colocated_100.sh',
    'run_qwen35_fused_swe500_2p6d.sh',
    'run_qwen35_fused_27b_tp2_swe500_4p4d.sh',
)
PLUGIN_LINE = 'export NIXL_PLUGIN_DIR=/tmp/pd-runtime/nixl-132-pr1987-plugin'
PLUGIN_GUARD = '''[[ -f "${NIXL_PLUGIN_DIR}/libplugin_UCX.so" ]] || {
  echo 'Missing audited NIXL PR1987 transport plugin; refusing unsafe fallback' >&2
  exit 2
}'''


def write(path, value, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    if executable:
        path.chmod(0o700)


def prefix(source, name, plugin):
    if name == NAMES[2]:
        # This short wrapper delegates to the already truncated full entry.
        assert source.count('exec bash ') == 1
        return source
    marker = '\nmkdir -p '
    assert marker in source, (name, 'missing reviewed side-effect boundary')
    selected = source.split(marker, 1)[0]
    if name == NAMES[-1]:
        assert selected.count(PLUGIN_LINE) == 1
        assert selected.count(PLUGIN_GUARD) == 1
        selected = selected.replace(PLUGIN_LINE, 'export NIXL_PLUGIN_DIR=' + str(plugin), 1)
    return selected + '''
# Fixture assertions only: the real colocated runtime EXIT cleanup sees no
# registered service; forbid even a plain kill attempt while collecting env.
if declare -F pd_cleanup_all >/dev/null; then
  [[ ${#PD_SERVICE_PGIDS[@]} -eq 0 ]] || exit 95
fi
kill() { echo 'forbidden kill in prefix fixture' >&2; exit 97; }
/usr/bin/env -0
'''


class ComplexReplayPrefixTests(unittest.TestCase):
    def test_six_configuration_prefixes_match_immutable_sources(self):
        with tempfile.TemporaryDirectory(prefix='dualpd-complex-prefix-') as temporary:
            fixture = Path(temporary)
            tree = fixture / 'workspace/DualPD'
            outside = fixture / 'outside'
            outside.mkdir()
            plugin = fixture / 'plugin'
            write(plugin / 'libplugin_UCX.so', 'fixture existence sentinel, not a shared library\n')
            runtime = 'scripts/common/runtime.sh'
            self.assertEqual((ROOT / runtime).read_text(), original(runtime))
            write(tree / runtime, original(runtime))
            # Stable clock only for deterministic run-name defaults. No mkdir,
            # Docker, GPU, process or plugin implementation is stubbed/executed.
            write(fixture / 'bin/date', '#!/bin/sh\nprintf "%s\\n" 20261109T000000Z\n', True)
            environment = {
                'PATH': str(fixture / 'bin') + ':/usr/bin:/bin',
                'HOME': '/fixture/home', 'LANG': 'C.UTF-8',
                'PD_ENV_BIN': str(fixture / 'bin'), 'CUDA_VISIBLE_DEVICES': '',
                'SGLANG_AGENTIC_FIXTURE_INHERITED': 'must-be-reset-by-fused-and-colocated',
            }
            configurations = [{}, {'MAX_INFLIGHT': '123', 'RUN_DIR': str(fixture / 'explicit-run'),
                                           'WORKLOAD_CONFIG': '/fixture/override-workload.yaml',
                                           'PD_FUSED_FAST_TOOL_THRESHOLD': '7',
                                           'PD_FUSED_CONGESTION_RECOMPUTE': 'false'}]
            cases = [(name, values) for name in NAMES for values in configurations]
            cases += [(NAMES[-2], values) for values in (
                {'PD_FUSED_4P4D': '1'}, {'PD_FUSED_2P2D': '1'},
                {'PD_DIAGNOSTIC_PREFILL_CHUNK_SIZE': '1024', 'PD_SERVE_ONLY': '1'},
            )]

            def install(directory, candidate):
                for name in NAMES:
                    relative = directory + '/' + name
                    source = ((ROOT / relative).read_text() if candidate else
                              original('scripts/new_method/' + name))
                    write(tree / relative, prefix(source, name, plugin))
                target = '${PREFIX_TREE}/' + directory + '/' + NAMES[1]
                write(fixture / 'bin/bash', '''#!/bin/bash
set -euo pipefail
case "$1" in
  "''' + target + '''") exec /bin/bash "$@";;
  *) echo "unapproved complex-prefix delegation" >&2; exit 91;;
esac
''', True)

            def run(directory, name, overrides):
                return subprocess.run(
                    ['/bin/bash', str(tree / directory / name)], cwd=outside,
                    env={**environment, 'PREFIX_TREE': str(tree), **overrides},
                    capture_output=True, timeout=10)

            def capture(directory, name, overrides):
                result = run(directory, name, overrides)
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                values = dict(item.decode().split('=', 1) for item in result.stdout.split(b'\0') if item)
                self.assertEqual(values['PWD'], str(tree) if 'colocated' in name or 'fused' in name else str(outside))
                return values

            baseline = {}
            install('scripts/new_method', False)
            for index, (name, values) in enumerate(cases):
                baseline[index] = capture('scripts/new_method', name, values)
            # The required plugin guard must fail closed in both source trees.
            sentinel = plugin / 'libplugin_UCX.so'
            sentinel.unlink()
            failed = run('scripts/new_method', NAMES[-1], {})
            self.assertEqual(failed.returncode, 2)
            self.assertIn(b'refusing unsafe fallback', failed.stderr)
            write(sentinel, 'existence sentinel\n')
            install('scripts/replay/new_method', True)
            for name in NAMES:
                (tree / 'scripts/new_method' / name).unlink()
            for index, (name, values) in enumerate(cases):
                with self.subTest(shell=name, overrides=values):
                    self.assertEqual(capture('scripts/replay/new_method', name, values), baseline[index])
            for values, message in (
                ({'PD_FUSED_2P2D': '1', 'PD_FUSED_4P4D': '1'}, b'Choose only one GPU topology'),
                ({'PD_DIAGNOSTIC_PREFILL_CHUNK_SIZE': '1024'}, b'requires PD_SERVE_ONLY=1'),
                ({'PD_FUSED_P_H2D_MAX_INFLIGHT': '0'}, b'must be a positive integer'),
            ):
                result = run('scripts/replay/new_method', NAMES[-2], values)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
            sentinel.unlink()
            failed = run('scripts/replay/new_method', NAMES[-1], {})
            self.assertEqual(failed.returncode, 2)
            self.assertIn(b'refusing unsafe fallback', failed.stderr)
            self.assertFalse((fixture / 'explicit-run').exists())
            self.assertFalse((tree / 'scripts/new_method/internal').exists())

