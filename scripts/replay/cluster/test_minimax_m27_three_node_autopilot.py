import json
from pathlib import Path

import pytest
if __package__:
    from . import minimax_m27_three_node_autopilot as m
else:
    import minimax_m27_three_node_autopilot as m


def test_order():
    assert m.ORDER == [(256,'agentic'),(256,'hicache'),(384,'agentic'),(384,'hicache'),
        (448,'agentic'),(448,'hicache'),(256,'native'),(384,'native'),(448,'native')]


@pytest.mark.parametrize('exp', m.EXPERIMENTS)
def test_manifest(exp, tmp_path):
    cfg = m.build_config(exp)
    assert cfg['run_id'] == exp['run_id']
    text = json.dumps(cfg)
    assert json.loads(m.TEMPLATES[exp['method']].read_text())['run_id'] not in text
    if exp['method'] == 'agentic':
        path = tmp_path/'cfg.json'; path.write_text(text)
        cfg = m.multinode.load_config(path)
        plan = m.multinode.plan(cfg)
        assert len(plan['workers']) == 3
        assert [n['role'] for n in cfg['nodes']] == ['prefill','decode','decode']
        assert all(n['d2p_host_gib_per_rank']*8 == 512 for n in cfg['nodes'] if n['role']=='decode')
        assert sum(n['d2p_host_gib_per_rank']*8 for n in cfg['nodes'] if n['role']=='decode') == 1024
        assert cfg['p2d_host_gib_per_rank'] * 8 == 256
        assert cfg['compact_handoff_payload'] is True
        assert cfg['profile_host_prepare'] is True
        assert cfg['serialize_p2d_direct_with_forward'] is True
        assert cfg['group_control']['enabled'] is True
        command = cfg['workload_command']
        assert cfg['slurm_node_jobs'] == m.JOBS
    else:
        comps = cfg['components']
        assert {v['node'] for v in comps.values()} == set(m.JOBS)
        for v in comps.values(): assert v['job_id'] == m.JOBS[v['node']]
        for k in ['prefill-0','decode-0','decode-1']:
            argv = comps[k]['command']
            assert argv[argv.index('--tp-size')+1] == '8'
            assert argv[argv.index('--ep-size')+1] == '8'
            assert not any('AGENTIC' in key for key in comps[k]['environment'])
        assert cfg['settings']['ep'] == 8
        assert comps['router']['command'].count('--decode') == 2
        command = comps['workload']['command']
        if exp['method'] == 'hicache':
            for k in ('prefill-0','decode-0','decode-1'):
                argv = comps[k]['command']
                storage = json.loads(argv[argv.index('--hicache-storage-backend-extra-config')+1])
                assert storage['master_server_address'].startswith(m.IPS['node2701']+':')
            assert '--disaggregation-decode-enable-offload-kvcache' in comps['decode-1']['command']
            assert cfg['settings']['mooncake_store_size_gb'] == 512
        else:
            assert '--disable-radix-cache' in comps['prefill-0']['command']
            assert '--disaggregation-decode-enable-offload-kvcache' not in comps['decode-1']['command']
    assert command[command.index('--max-inflight')+1] == str(exp['concurrency'])
    assert len(command[command.index('--decode-endpoints')+1].split(',')) == 2
    assert '--preserve-source-order' in command


def test_result_ignores_failed_and_requires_metrics(tmp_path, monkeypatch):
    exp = m.EXPERIMENTS[0]
    monkeypatch.setattr(m.core, 'run_dir', lambda _: tmp_path)
    monkeypatch.setattr(m.core, 'load_progress', lambda _: [dict(status='Status.FAILED',completed_at=i) for i in range(500)])
    assert m.result_for(exp) is None


def test_templates_unchanged():
    for method, path in m.TEMPLATES.items():
        before = path.read_bytes()
        m.build_config(next(e for e in m.EXPERIMENTS if e['method']==method))
        assert path.read_bytes() == before


def test_t450_completed_only_two_decode_counters(tmp_path, monkeypatch):
    exp = m.EXPERIMENTS[0]
    monkeypatch.setattr(m.core, 'run_dir', lambda _: tmp_path)
    rows = [dict(status='Status.COMPLETED',completed_at=i, reward=0) for i in range(1,451)]
    rows += [dict(status='Status.FAILED',completed_at=.5,environment_error='test')]
    monkeypatch.setattr(m.core, 'load_progress', lambda _: rows)
    work=tmp_path/'workload'; work.mkdir()
    (work/'measurement_start.json').write_text(json.dumps({'measurement_start_wall':0}))
    metrics=[]
    for t in (0,451):
        for role, count in [('prefill',1),('decode',2)]:
            cat='extend' if role=='prefill' else 'decode'
            metrics.append(dict(ts=t,role=role,engine_count=count,endpoint_metrics=[dict(endpoint=f'{role}{i}',metrics={
                f'sglang_gpu_execution_seconds_total|category=forward_{cat}':t*4,
                'sglang_realtime_tokens_total|mode=decode':t*100,
                'sglang_realtime_tokens_total|mode=prefill_compute':t*10,
                'sglang_realtime_tokens_total|mode=prefill_cache':t*90,
                'sglang_full_token_usage':.4,'sglang_num_running_reqs':20}) for i in range(count)]))
    (work/'engine_metrics.jsonl').write_text('\n'.join(map(json.dumps,metrics)))
    result=m.result_for(exp)
    assert result['t450_seconds']==450
    assert result['failed']==1
    assert result['t450_decode_tps']==200
    details=m.forward_details(exp,result)
    assert details['prefill']['forward']==.5
    assert details['decode']['forward']==.5
    assert details['decode']['running']==20
    metrics[-1]['endpoint_metrics'].pop()
    (work/'engine_metrics.jsonl').write_text('\n'.join(map(json.dumps,metrics)))
    assert m.result_for(exp) is None
