"""Check public launcher defaults without starting any GPU services."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/new_method/run_qwen3_8b.sh'


def capture(tmp_path, mode, overrides=None):
    stub = tmp_path / 'bash'
    stub.write_text('#!/bin/sh\nprintf "CASE_SCRIPT=%s\\n" "$1"\nexec /usr/bin/env\n')
    stub.chmod(0o755)
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'MODEL_PATH': '/model',
           'PD_DATA_ROOT': '/data'}
    env.update(overrides or {})
    result = subprocess.run(['/bin/bash', str(SCRIPT), mode], env=env,
                            text=True, capture_output=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


@pytest.mark.parametrize('mode', ['browsecomp', 'swe'])
def test_common_defaults(tmp_path, mode):
    env = capture(tmp_path, mode)
    assert env['CASE_SCRIPT'] == str(SCRIPT.parent / 'run_4p4d_numa_case.sh')
    assert env['MODEL_PATH'] == '/model'
    assert env['DUALPD_ENGINE_ROOT'] == str(ROOT / '3rdparty/sglang')
    assert env['SGLANG_OVERLAY_ROOT'] == str(ROOT / '3rdparty/sglang/python')
    assert env['D2P_HOST_ARENA_GIB_PER_P'] == '128'
    assert env['P2D_HOST_ARENA_GIB_PER_P'] == '64'
    assert env['FAST_TOOL_THRESHOLD_SECONDS'] == env['DIRECT_WAIT_SECONDS'] == '1'
    assert env['WARMUP_SECONDS'] == '300'
    assert env['MEASURE_SECONDS'] == '1200'
    assert env['PREFILL_GPUS'] == '0 2 4 6'
    assert env['P_H2D_MAX_INFLIGHT'] == '2'
    assert env['SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH'] == '32'
    assert env['SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW'] == '8'


def test_browsecomp(tmp_path):
    env = capture(tmp_path, 'browsecomp')
    assert env['QA_DATA'] == '/data/browsecomp/bc_train.jsonl'
    assert env['WORKLOAD_CONFIG'] == ''
    assert env['REQUESTS'] == '680'
    assert env['MAX_INFLIGHT'] == '512'
    assert env['CLOSED_LOOP'] == '1'
    assert env['PD_SKIP_SEARCH'] == '0'
    assert env['DECODE_MEM_FRACTION_STATICS'].endswith('0.60')


def test_swe(tmp_path):
    env = capture(tmp_path, 'swe')
    assert Path(env['WORKLOAD_CONFIG']).is_file()
    assert env['REQUESTS'] == '500'
    assert env['MAX_INFLIGHT'] == '64'
    assert env['CLOSED_LOOP'] == '0'
    assert env['PD_SKIP_SEARCH'] == '1'
    assert env['MODEL_TOOL_CALL_PARSER'] == 'qwen25'


def test_overrides_and_model_root(tmp_path):
    env = capture(tmp_path, 'browsecomp', {
        'MODEL_PATH': '', 'PD_MODEL_ROOT': '/models', 'MAX_INFLIGHT': '512',
        'RUN_DIR': '/results', 'FAST_TOOL_THRESHOLD_SECONDS': '0.5',
        'WORKLOAD_CONFIG': '', 'DUALPD_ENGINE_ROOT': '/engine',
    })
    assert env['MODEL_PATH'] == '/models/Qwen3-8B'
    assert env['MAX_INFLIGHT'] == '512'
    assert env['RUN_DIR'] == '/results'
    assert env['FAST_TOOL_THRESHOLD_SECONDS'] == '0.5'
    assert env['WORKLOAD_CONFIG'] == ''
    assert env['SGLANG_OVERLAY_ROOT'] == '/engine/python'


@pytest.mark.parametrize('args,code', [(['--help'], 0), (['bad'], 2),
                                     (['browsecomp', 'extra'], 2), ([], 1)])
def test_preflight(args, code):
    result = subprocess.run(['/bin/bash', str(SCRIPT), *args],
                            env={'PATH': '/usr/bin:/bin'}, capture_output=True)
    assert result.returncode == code


def test_syntax():
    subprocess.run(['/bin/bash', '-n', str(SCRIPT)], check=True)
