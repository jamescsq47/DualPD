"""Public baseline launch contracts, without starting GPU or Mooncake services."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/baseline/run_qwen3_8b_browsecomp.sh'


@pytest.mark.parametrize('mode,case', [('no_reverse', 'no_reverse'),
                                      ('hicache', 'native_mooncake')])
@pytest.mark.parametrize('concurrency', ['512', '128'])
def test_default_alignment_and_isolation(tmp_path, mode, case, concurrency):
    stub = tmp_path / 'bash'
    stub.write_text('#!/bin/sh\nprintf "CASE_SCRIPT=%s\\n" "$1"\nexec /usr/bin/env\n')
    stub.chmod(0o755)
    env = {'PATH': f'{tmp_path}:/usr/bin:/bin', 'MODEL_PATH': '/model',
           'PD_DATA_ROOT': '/data', 'BASELINE_ENV_BIN': '/baseline/bin',
           'PD_ENV_BIN': '/modified/bin', 'PYTHONPATH': '/modified/python',
           'SGLANG_OVERLAY_ROOT': '/modified/python',
           'DUALPD_ENGINE_ROOT': '/modified',
           'SGLANG_AGENTIC_KV_LIFECYCLE': 'true',
           'SGLANG_AGENTIC_MULTINODE_ENABLED': 'true',
           'SGLANG_PD_ABLATION_P2D_PREBIND': 'true'}
    if concurrency != '512':
        env['MAX_INFLIGHT'] = concurrency
    result = subprocess.run(['/bin/bash', str(SCRIPT), mode], env=env,
                            text=True, capture_output=True, check=True)
    got = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    assert got['CASE_SCRIPT'] == str(SCRIPT.parent / 'run_pd_case.sh')
    assert got['CASE_MODE'] == case
    assert got['PD_ENV_BIN'] == '/baseline/bin'
    assert got['PYTHONPATH'] == str(ROOT)
    assert 'SGLANG_OVERLAY_ROOT' not in got
    assert 'DUALPD_ENGINE_ROOT' not in got
    assert not any(k.startswith(('SGLANG_AGENTIC_', 'SGLANG_PD_')) for k in got)
    assert got['MAX_INFLIGHT'] == concurrency
    assert got['REQUESTS'] == '680'
    assert got['CLOSED_LOOP'] == 'true'
    assert got['WARMUP_SECONDS'] == '300'
    assert got['MEASURE_SECONDS'] == '1200'
    assert got['QA_DATA'] == '/data/browsecomp/bc_train.jsonl'
    assert got['PREFILL_GPUS'] == '0 2 4 6'
    assert got['DECODE_GPUS'] == '1 3 5 7'
    assert got['P_HICACHE_SIZE'] == '128'
    assert got['D_HICACHE_SIZE'] == '56'


@pytest.mark.parametrize('args,code', [(['--help'], 0), (['invalid'], 2),
                                     (['hicache', 'extra'], 2), ([], 1)])
def test_preflight(args, code):
    result = subprocess.run(['/bin/bash', str(SCRIPT), *args],
                            env={'PATH': '/usr/bin:/bin'}, capture_output=True)
    assert result.returncode == code


def test_syntax():
    subprocess.run(['/bin/bash', '-n', str(SCRIPT)], check=True)
