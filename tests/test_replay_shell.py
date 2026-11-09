"""Replay shells vs immutable sources: full environment/argv boundary A/B.

Real Bash and YAML profile loader, explicitly stubbed fingerprint validation.
The terminal pipeline is absent: an exact-path capture stub replaces it. This
tests five shells at the terminal pipeline and seven more at the common-case
boundary. The six remaining complex shells and native/model/readiness/
performance are outside this test's scope.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dualpd.scripts.verify_replay_layout import original

ROOT = Path(__file__).resolve().parents[1]
SHELLS = (
    'run_qwen3_8b_tp1_browsecomp_4p4d.sh',
    'run_h100_integration_a100.sh',
    'run_browsecomp_c512_p2d_direct_only.sh',
    'run_browsecomp_c576_p2d_direct_only.sh',
    'run_browsecomp_qwen3_8b_4p4d_ablation.sh',
)
SUPPORT = (
    'scripts/new_method/run_4p4d_numa_case.sh',
    'scripts/new_method/internal/agentic_case_environment.sh',
    'scripts/new_method/internal/apply_experiment_profile.sh',
    'scripts/tools/load_experiment_profile.py',
    'configs/profiles/browsecomp_qwen3_8b_tp1_4p4d.yaml',
)
ABLATIONS = (
    'pure-recompute', 'full-1s', 'd2p-slow-only-1s',
    'fast-direct-fail-recompute-1s', 'direct-only-recompute-1s',
    'd2p-direct-only', 'd2p-slow-only', 'random-routing',
    'p2d-direct-only', 'p2d-prebind-direct-only',
)
OVERRIDES = {
    'MAX_INFLIGHT': '123', 'FAST_TOOL_THRESHOLD_SECONDS': '3',
    'DIRECT_WAIT_SECONDS': '4', 'WARMUP_SECONDS': '301',
    'MEASURE_SECONDS': '1201', 'MODEL_PATH': '/fixture/explicit-model',
    'D2P_HOST_ARENA_GIB_PER_P': '96', 'P2D_HOST_ARENA_GIB_PER_P': '48',
    'SGLANG_AGENTIC_KV_FORCE_SLOW_PATH': 'true',
    'SGLANG_PD_ABLATION_P2D_PREBIND': 'true',
}


def write_fixture(path, text, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o700)


def install_stubs(fixture, directory):
    allowed = ['${CAPTURE_TREE}/' + directory + '/' + name for name in SHELLS]
    allowed += ['${CAPTURE_TREE}/scripts/new_method/run_4p4d_numa_case.sh']
    cases = '|'.join('"' + path + '"' for path in allowed)
    write_fixture(fixture / 'bin/bash', '''#!/bin/bash
set -euo pipefail
if [[ "$1" == "${CAPTURE_TREE}/scripts/new_method/internal/run_agentic_pipeline.sh" ]]; then
  exec "${CAPTURE_REAL_PYTHON}" "${CAPTURE_SCRIPT}" "$@"
fi
case "$1" in
''' + cases + ''') exec /bin/bash "$@";;
*) echo "unapproved Bash fixture target: $1" >&2; exit 91;;
esac
''', executable=True)
    write_fixture(fixture / 'bin/python', '''#!/bin/bash
set -euo pipefail
case "$1" in
  "${CAPTURE_TREE}/scripts/new_method/../tools/check_environments.py") exit 0;;
  "${CAPTURE_TREE}/scripts/new_method/../tools/load_experiment_profile.py") exec "${CAPTURE_REAL_PYTHON}" "$@";;
  *) echo "unapproved Python fixture target: $1" >&2; exit 92;;
esac
''', executable=True)


class ReplayShellTests(unittest.TestCase):
    def test_actual_five_shell_closure_matches_immutable_environment_and_argv(self):
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-shell-') as temporary:
            fixture = Path(temporary)
            tree = fixture / 'workspace/DualPD'
            for relative in [*('scripts/new_method/' + name for name in SHELLS), *SUPPORT]:
                if relative in SUPPORT:
                    # The integration fixture's maintained dependencies must
                    # still be byte-identical to the immutable support sources.
                    self.assertEqual((ROOT / relative).read_bytes(), original(relative).encode())
                write_fixture(tree / relative, original(relative))
            outside = fixture / 'outside'
            outside.mkdir()
            write_fixture(fixture / 'capture.py',
                          "import json,os,sys\nprint(json.dumps({'environment':dict(os.environ),'command':sys.argv[1:]}))\n")
            environment = {
                'PATH': str(fixture / 'bin') + ':/usr/bin:/bin', 'HOME': '/fixture/home',
                'LANG': 'C.UTF-8', 'PD_ENV_BIN': str(fixture / 'bin'),
                'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
                'CAPTURE_TREE': str(tree), 'CAPTURE_REAL_PYTHON': sys.executable,
                'CAPTURE_SCRIPT': str(fixture / 'capture.py'),
            }
            # Same tree and environment both phases: no filtering/normalizing
            # runtime flags, capacities, paths, or even shell-generated values.
            def capture(directory, name, args, overrides):
                result = subprocess.run(
                    ['/bin/bash', str(tree / directory / name), *args],
                    cwd=outside, env={**environment, **overrides},
                    capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(value['command'], [
                    str(tree / 'scripts/new_method/internal/run_agentic_pipeline.sh')])
                return value

            cases = [(name, ()) for name in SHELLS[:-1]]
            cases += [(SHELLS[-1], (variant,)) for variant in ABLATIONS]
            baseline = {}
            install_stubs(fixture, 'scripts/new_method')
            for name, args in cases:
                for label, overrides in (('default', {}), ('overrides', OVERRIDES)):
                    baseline[name, args, label] = capture('scripts/new_method', name, args, overrides)
            # Candidate bytes come from the actual checkout, not a proposed
            # transformation. Remove old entrypoints to avoid masking stale calls.
            for name in SHELLS:
                relative = 'scripts/replay/new_method/' + name
                write_fixture(tree / relative, (ROOT / relative).read_text())
                (tree / 'scripts/new_method' / name).unlink()
            install_stubs(fixture, 'scripts/replay/new_method')
            for name, args in cases:
                for label, overrides in (('default', {}), ('overrides', OVERRIDES)):
                    with self.subTest(shell=name, args=args, configuration=label):
                        actual = capture('scripts/replay/new_method', name, args, overrides)
                        self.assertEqual(actual, baseline[name, args, label])
            invalid = subprocess.run(
                ['/bin/bash', str(tree / 'scripts/replay/new_method' / SHELLS[-1]), 'invalid-ablation'],
                cwd=outside, env=environment, capture_output=True, text=True, timeout=20)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn('unknown ablation', invalid.stderr)
            # None of the code capable of starting a model was copied.
            self.assertFalse((tree / 'scripts/new_method/internal/run_agentic_pipeline.sh').exists())

    def test_seven_more_wrappers_preserve_case_boundary_contracts(self):
        # These wrappers reach the common-case boundary, not the pipeline.
        # Capturing before the case avoids engine/fingerprint/readiness actions.
        names = (
            'run_mixed_qwen3_8b_2p6d_ablation.sh',
            'run_qwen35_27b_tp2_browsecomp_4p4d.sh',
            'run_qwen35_27b_tp2_swe_openenv_4p4d_full.sh',
            'run_qwen3_32b_tp2_browsecomp_2p6d.sh',
            'run_qwen3_32b_tp2_browsecomp_recheck.sh',
            'run_qwen3_32b_tp2_swe_bench_verified_2p6d.sh',
            'run_qwen3_32b_tp2_terminal_bench_2p6d.sh',
        )
        cases = [(name, ()) for name in names[1:]] + [
            (names[0], (variant,)) for variant in
            ('full', 'd2p-direct-only', 'd2p-slow-only', 'random-routing', 'p2d-direct-only')]
        with tempfile.TemporaryDirectory(prefix='dualpd-replay-case-boundary-') as temporary:
            fixture = Path(temporary)
            tree = fixture / 'workspace/DualPD'
            outside = fixture / 'outside'
            outside.mkdir()
            write_fixture(fixture / 'capture.py',
                          "import json,os,sys\nprint(json.dumps({'environment':dict(os.environ),'command':sys.argv[1:]}))\n")
            environment = {
                'PATH': str(fixture / 'bin') + ':/usr/bin:/bin', 'HOME': '/fixture/home',
                'LANG': 'C.UTF-8', 'PD_ENV_BIN': str(fixture / 'bin'),
                'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
                'CAPTURE_TREE': str(tree), 'CAPTURE_REAL_PYTHON': sys.executable,
                'CAPTURE_SCRIPT': str(fixture / 'capture.py'),
            }

            def install(directory):
                allowed = '|'.join('"${CAPTURE_TREE}/' + directory + '/' + name + '"'
                                   for name in names)
                write_fixture(fixture / 'bin/bash', '''#!/bin/bash
set -euo pipefail
case "$1" in
  "${CAPTURE_TREE}/scripts/new_method/run_2p6d_numa_case.sh"|"${CAPTURE_TREE}/scripts/new_method/run_4p4d_numa_case.sh")
    exec "${CAPTURE_REAL_PYTHON}" "${CAPTURE_SCRIPT}" "$@";;
''' + allowed + ''') exec /bin/bash "$@";;
  *) echo "unapproved case-boundary target: $1" >&2; exit 91;;
esac
''', executable=True)

            def capture(directory, name, args, overrides):
                result = subprocess.run(
                    ['/bin/bash', str(tree / directory / name), *args],
                    cwd=outside, env={**environment, **overrides},
                    capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                value = json.loads(result.stdout.splitlines()[-1])
                target = ('run_4p4d_numa_case.sh' if name.startswith('run_qwen35_27b_')
                          else 'run_2p6d_numa_case.sh')
                self.assertEqual(value['command'], [str(tree / 'scripts/new_method' / target)])
                # Only these two declared project path values use SCRIPT_DIR
                # spellings containing '..'. Compare their resolved addresses;
                # all other env keys/values and argv remain byte-exact.
                for key in ('WORKLOAD_CONFIG', 'RUN_DIR'):
                    path = value['environment'].get(key)
                    if path:
                        self.assertTrue(Path(path).is_absolute(), (key, path))
                        value['environment'][key] = str(Path(path).resolve())
                return value

            baseline = {}
            for name in names:
                write_fixture(tree / 'scripts/new_method' / name, original('scripts/new_method/' + name))
            install('scripts/new_method')
            for name, args in cases:
                for label, overrides in (('default', {}), ('overrides', OVERRIDES)):
                    baseline[name, args, label] = capture('scripts/new_method', name, args, overrides)
            for name in names:
                relative = 'scripts/replay/new_method/' + name
                write_fixture(tree / relative, (ROOT / relative).read_text())
                (tree / 'scripts/new_method' / name).unlink()
            install('scripts/replay/new_method')
            for name, args in cases:
                for label, overrides in (('default', {}), ('overrides', OVERRIDES)):
                    with self.subTest(shell=name, args=args, configuration=label):
                        self.assertEqual(capture('scripts/replay/new_method', name, args, overrides),
                                         baseline[name, args, label])
            self.assertFalse((tree / 'scripts/new_method/run_2p6d_numa_case.sh').exists())
            self.assertFalse((tree / 'scripts/new_method/run_4p4d_numa_case.sh').exists())
