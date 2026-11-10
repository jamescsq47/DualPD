"""Relocated offline-tool contracts; never run SIF conversion or GPU services."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from dualpd.scripts.cluster import dram_complete_frame_analysis as analysis
from dualpd.scripts import verify_replay_layout as proof

ROOT = Path(__file__).resolve().parents[1]


def analysis_fixture(directory):
    directory.mkdir()
    boundary = dict(warmup_seconds=300, measurement_seconds=1200,
                    measurement_start_wall=1000, measurement_end_wall=2200)
    (directory / 'closed_loop_boundaries.json').write_text(json.dumps(boundary))
    env = dict(PREFILL_PORTS='27300', DECODE_PORTS='27301')
    (directory / 'launch_config.json').write_text(json.dumps(dict(env=env)))
    (directory / 'requests.jsonl').write_text('')
    (directory / 'logs').mkdir()
    rows = []
    for ts in range(1000, 2201, 60):
        for role, port, token, gpu in [('prefill', 27300, analysis.P_TOKEN, analysis.P_GPU),
                                       ('decode', 27301, analysis.D_TOKEN, analysis.D_GPU)]:
            metrics = {token: 100000 + ts * 10, gpu: ts}
            rows.append(dict(ts=ts, role=role, metrics=metrics,
                             endpoint_metrics=[dict(endpoint=f'http://127.0.0.1:{port}',
                                                    metrics=metrics)]))
    (directory / 'engine_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


@pytest.mark.parametrize('explicit', [False, True])
def test_analysis_selected_checkout_and_interpreter(tmp_path, monkeypatch, explicit):
    directory = tmp_path / 'saved run'
    analysis_fixture(directory)
    immutable = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    checkout = tmp_path / 'selected checkout'
    script = checkout / 'scripts/cluster/dram_complete_frame_analysis.py'
    monkeypatch.setattr(analysis, '__file__', str(script))
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs == dict(check=True, timeout=600)
        view = Path(argv[-1])
        (view / 'offload_analysis_summary.json').write_text('{"decode_tps": 10.0}')
        (view / 'steady_offload_analysis.png').write_bytes(b'fixture-not-real-plot')

    monkeypatch.setattr(analysis.subprocess, 'run', fake_run)
    overrides = {}
    expected_python = analysis.DEFAULT_ANALYSIS_PYTHON
    expected_root = checkout
    if explicit:
        # Explicit spelling (including ..) is retained in the subprocess argv.
        expected_root = tmp_path / 'other checkout/../other checkout'
        expected_python = str(tmp_path / 'selected python')
        overrides = dict(python=Path(expected_python), project_root=expected_root)
    view = analysis.prepare_analysis(directory, **overrides)
    assert calls == [([expected_python, str(expected_root / 'scripts/tools/analyze_pd_offload.py'),
                       '--run-dir', str(view)], dict(check=True, timeout=600))]
    for name, original in immutable.items():
        assert (directory / name).read_bytes() == original
    assert view == directory / analysis.VIEW
    audit = json.loads((directory / 'complete_frame_validation.json').read_text())
    assert audit['corrected_decode_tps'] == 10.0
    assert audit['counter_span_seconds'] == 1200
    assert (view / 'requests.jsonl').resolve() == directory / 'requests.jsonl'


def test_analysis_incomplete_window_still_rejected_before_analyzer(tmp_path, monkeypatch):
    directory = tmp_path / 'incomplete'
    analysis_fixture(directory)
    p = directory / 'closed_loop_boundaries.json'
    boundary = json.loads(p.read_text()); boundary['measurement_seconds'] = 1199
    p.write_text(json.dumps(boundary))
    monkeypatch.setattr(analysis.subprocess, 'run', lambda *a, **k: pytest.fail('analyzer launched'))
    with pytest.raises(RuntimeError, match='incomplete 300\\+1200'):
        analysis.prepare_analysis(directory, python=sys.executable, project_root=tmp_path)
    assert not (directory / analysis.VIEW).exists()


def shell_fixture(tmp_path):
    workspace = tmp_path / 'workspace with spaces'
    script_dir = workspace / 'DualPD/scripts/cluster'
    script_dir.mkdir(parents=True)
    selected = script_dir / 'run_swe_sif_shards_in_step.sh'
    shutil.copy2(ROOT / 'scripts/cluster' / selected.name, selected)
    # A real parsed, rejecting policy fixture: not a new permissive site policy.
    selected.with_name('skopeo-public-policy.json').write_text(
        '{"default": [{"type": "reject"}]}')
    bindir = tmp_path / 'stub bin'
    bindir.mkdir()
    stub = bindir / 'python'
    stub.write_text('#!' + sys.executable + '\n' + '''
import json, os, pathlib, sys
args = sys.argv[1:]
shard = args[args.index('--shard-index') + 1]
capture = pathlib.Path(os.environ['STUB_CAPTURE'])
capture.mkdir(exist_ok=True)
(capture / (shard + '.json')).write_text(json.dumps(args))
print('captured shard ' + shard)
print('stderr shard ' + shard, file=sys.stderr)
sys.exit(7 if shard == os.environ.get('FAIL_SHARD') else 0)
''')
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if k not in {'BASH_ENV', 'ENV', 'PYTHONSTARTUP'}
           and not k.startswith(('SWE_SIF_', 'SLURM_', 'FAIL_SHARD', 'BASH_FUNC_'))}
    env.update(PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               SLURM_JOB_ID='12345', SLURM_PROCID='1', SLURM_NTASKS='3',
               SWE_SIF_WORKERS_PER_NODE='2', SWE_SIF_LOG_TAG='fixture',
               SWE_SIF_TEMP_BASE=str(tmp_path / 'temporary roots'),
               SWE_SIF_DIR=str(tmp_path / 'output sifs'))
    return workspace, selected, stub, env


def run_capture(script, env, capture):
    result = subprocess.run(['/bin/bash', str(script)],
                            env=dict(env, STUB_CAPTURE=str(capture)),
                            capture_output=True, text=True, timeout=30)
    argv = {p.stem: json.loads(p.read_text()) for p in capture.glob('*.json')}
    return result, argv


@pytest.mark.parametrize('fail_shard', ['', '1', '4'])
def test_sif_original_default_shards_argv_logs_and_failure_aggregation(tmp_path, fail_shard):
    workspace, selected, stub, env = shell_fixture(tmp_path)
    env['FAIL_SHARD'] = fail_shard
    # The immutable historical shell is inert outside this disposable fixture.
    # Relocate only its checkout spelling, then intercept its Python command.
    old = proof.original('scripts/cluster/run_swe_sif_shards_in_step.sh')
    # Old unquoted code addresses had no spaces; quote those selected addresses
    # in the fixture only so a path-spelling test can safely capture their argv.
    old = old.replace('/home/shangy/dualpd/DualPD/scripts/cluster/convert_swe_oci_to_sif.py',
                      '"' + str(selected.with_name('convert_swe_oci_to_sif.py')) + '"')
    old = old.replace('/home/shangy/dualpd/DualPD/scripts/cluster/skopeo-public-policy.json',
                      '"' + str(selected.with_name('skopeo-public-policy.json')) + '"')
    old = old.replace('/home/shangy/dualpd/swe-sif-step-', str(workspace / 'swe-sif-step-'))
    original = tmp_path / 'original.sh'; original.write_text(old)
    before, old_argv = run_capture(original, env, tmp_path / 'old capture')
    log_before = {p.name: p.read_bytes() for p in workspace.glob('swe-sif-step-*')}
    after, new_argv = run_capture(selected, env, tmp_path / 'new capture')
    assert before.returncode == after.returncode == (1 if fail_shard else 0)
    assert old_argv == new_argv
    assert set(new_argv) == {'1', '4'}
    for args in new_argv.values():
        assert args[args.index('--shard-count') + 1] == '6'
        assert args[args.index('--mksquashfs-processors') + 1] == '4'
        assert args[args.index('--skopeo') + 1] == '/usr/bin/skopeo'
    assert log_before == {p.name: p.read_bytes() for p in workspace.glob('swe-sif-step-*')}
    assert len(log_before) == 4


def test_sif_explicit_site_paths_and_python_are_single_arguments(tmp_path):
    workspace, selected, stub, env = shell_fixture(tmp_path)
    log_dir = tmp_path / 'selected logs'
    env.update(SWE_SIF_LOG_DIR=str(log_dir), SWE_SIF_PYTHON=str(stub),
               SWE_SIF_DATASET=str(tmp_path / 'data with $literal; chars.jsonl'),
               SWE_SIF_OCI_LAYOUT=str(tmp_path / 'oci with spaces'))
    policy = tmp_path / 'selected signature policy.json'
    policy.write_text('{"default": [{"type": "reject"}]}')
    policy_bytes = policy.read_bytes()
    env['SWE_SIF_POLICY'] = str(policy)
    result, argv = run_capture(selected, env, tmp_path / 'capture')
    assert result.returncode == 0, result.stderr
    for args in argv.values():
        assert args[0] == str(selected.with_name('convert_swe_oci_to_sif.py'))
        assert args[args.index('--dataset') + 1] == env['SWE_SIF_DATASET']
        assert args[args.index('--oci-layout') + 1] == env['SWE_SIF_OCI_LAYOUT']
        assert args[args.index('--policy') + 1] == str(policy)
    assert json.loads(policy.read_text()) == {'default': [{'type': 'reject'}]}
    assert policy.read_bytes() == policy_bytes
    assert len(list(log_dir.glob('swe-sif-step-*'))) == 4
    assert not list(workspace.glob('swe-sif-step-*'))


@pytest.mark.parametrize('override', [dict(SWE_SIF_WORKERS_PER_NODE='0'),
                                       dict(SWE_SIF_TEMP_BASE='relative'),
                                       dict(SLURM_PROCID='')])
def test_sif_invalid_inputs_never_invoke_conversion(tmp_path, override):
    _, selected, _, env = shell_fixture(tmp_path)
    env.update(override)
    result, argv = run_capture(selected, env, tmp_path / 'capture')
    assert result.returncode != 0
    assert not argv
    assert not Path(env['SWE_SIF_DIR']).exists()


@pytest.mark.parametrize('explicit', [False, True])
def test_sif_missing_policy_fails_before_directories_or_workers(tmp_path, explicit):
    _, selected, _, env = shell_fixture(tmp_path)
    if explicit:
        env['SWE_SIF_POLICY'] = str(tmp_path / 'missing policy.json')
    else:
        selected.with_name('skopeo-public-policy.json').unlink()
    result, argv = run_capture(selected, env, tmp_path / 'capture')
    assert result.returncode == 2
    assert 'SWE_SIF_POLICY' in result.stderr
    assert not argv
    assert not Path(env['SWE_SIF_DIR']).exists()
    assert not Path(env['SWE_SIF_TEMP_BASE']).exists()
