#!/usr/bin/env python3
"""Immutable 1P:2D MiniMax manifests and a serial T450 experiment controller.

Site orchestration only: reuse the tested engine and supervised component
entrypoints; never change snapshot ownership, fences, or baseline algorithms.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from types import SimpleNamespace

import qwen38_matrix_autopilot as core
import qwen35_native_pd_slurm as native
import qwen35_slurm as site
import multinode

ROOT = core.ROOT
RUNS = core.RUNS
JOBS = {"node1800": "26049949", "node2701": "26049949", "node2800": "26078099"}
IPS = {"node1800": "172.16.18.0", "node2701": "172.16.27.1", "node2800": "172.16.28.0"}
CONTROL = RUNS / "minimax-m27-three-node-1p2d-c256c384c448-20261018-r1"
REPORT = ROOT / "examples/pd/runs-host/SWEBENCH_MINIMAX_M27_TP8_THREE_NODE_1P2D.md"
TEMPLATES = {
    "agentic": RUNS / "minimax-m27-tp8ep8-agentic-c448-20261017-compactmeta-r6/multinode-v3.json",
    "hicache": RUNS / "minimax-m27-tp8ep8-two-node-hicache-mooncake-1p1d-swe500-c448-20261017-r1/native-pd-manifest.json",
    "native": RUNS / "minimax-m27-tp8ep8-two-node-native-oneway-1p1d-swe500-c448-20261017-r1/native-pd-manifest.json",
}
ORDER = [(c, m) for c in (256, 384, 448) for m in ("agentic", "hicache")]
ORDER += [(c, "native") for c in (256, 384, 448)]
EXPERIMENTS = [dict(concurrency=c, method=m,
    run_id=f"m27-tp8ep8-1p2d-{m}-swe500-c{c}-20261018-r1") for c, m in ORDER]
LABELS = {"agentic": "Agentic-PD r6", "hicache": "HiCache/Mooncake", "native": "No-reverse PD"}


def replace_strings(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_strings(x, old, new) for x in value]
    if isinstance(value, dict):
        return {k: replace_strings(v, old, new) for k, v in value.items()}
    return value


def set_arg(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend([flag, str(value)])


def build_config(experiment):
    method = experiment['method']
    cfg = json.loads(TEMPLATES[method].read_text())
    cfg = replace_strings(cfg, cfg['run_id'], experiment['run_id'])
    if method == 'agentic':
        cfg['slurm_nodes']['extra_decode'] = 'node2800'
        cfg['slurm_node_jobs'] = dict(JOBS)
        cfg['group_control']['token'] = secrets.token_urlsafe(32)
        d1 = copy.deepcopy(cfg['nodes'][1])
        d1.update(node_id='d1', engine_id='decode-1', host_ip=IPS['node2800'])
        cfg['nodes'].append(d1)
        # User explicitly requires 512 GiB on EACH Decode node (1 TiB total).
        cfg['d2p_host_gib_per_rank'] = 64
        for node in cfg['nodes']:
            node['d2p_host_gib_per_rank'] = 64
        command = cfg['workload_command']
        set_arg(command, '--max-inflight', experiment['concurrency'])
        set_arg(command, '--prefill-endpoints', f"http://{IPS['node1800']}:23900")
        set_arg(command, '--decode-endpoints', ','.join(f'http://{IPS[n]}:23900' for n in ('node2701', 'node2800')))
        return cfg
    cfg['nodes']['extra_decode'] = 'node2800'
    cfg['addresses']['extra_decode'] = IPS['node2800']
    cfg['settings'].update(ep=8, decode_groups=2, concurrency=experiment['concurrency'])
    components = cfg['components']
    d1 = copy.deepcopy(components['decode-0'])
    d1 = replace_strings(d1, IPS['node2701'], IPS['node2800'])
    d1 = replace_strings(d1, 'decode-0', 'decode-1')
    d1.update(node='node2800', job_id=JOBS['node2800'])
    if method == 'hicache':
        # Only local_hostname changes. Both Ds use the SAME existing 512 GiB
        # store on node2701, not a nonexistent second store on node2800.
        flag = '--hicache-storage-backend-extra-config'
        storage = json.loads(components['decode-0']['command'][components['decode-0']['command'].index(flag) + 1])
        storage['local_hostname'] = IPS['node2800']
        set_arg(d1['command'], flag, json.dumps(storage, separators=(',', ':')))
    components['decode-1'] = d1
    components['router']['command'].extend(['--decode', f"http://{IPS['node2800']}:{d1['port']}"])
    workload = components['workload']['command']
    set_arg(workload, '--max-inflight', experiment['concurrency'])
    set_arg(workload, '--decode-endpoints', ','.join(components[k]['health'].removesuffix('/health') for k in ('decode-0', 'decode-1')))
    for component in components.values():
        component['job_id'] = JOBS[component['node']]
    return cfg


def remote(node, argv, *, timeout=60, check=True):
    return core.command(['srun', '--jobid', JOBS[node], '--overlap', '-N1', '-n1', '-c1', '-w', node, *argv], timeout=timeout, check=check)


def allocation_check():
    remaining = []
    for job in set(JOBS.values()):
        fields, nodes = site.allocation(job, min_nodes=1)
        assert all(n in nodes for n, j in JOBS.items() if j == job)
        end = subprocess.check_output(['date', '-u', '-d', fields['EndTime'], '+%s'], text=True)
        remaining.append(int(end) - int(time.time()))
    return 'RUNNING', min(remaining)


def gpu_process_count():
    count = 0
    for node in JOBS:
        out = remote(node, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).stdout
        count += len([x for x in out.splitlines() if x.strip().isdigit()])
    return count


def agentic_action(experiment, action, *, node_id=None, component=None, detach=False):
    path = core.run_dir(experiment) / 'multinode-v3.json'
    args = SimpleNamespace(job_id=JOBS['node1800'], prefill_node='node1800', python=str(core.PD_PYTHON),
        cpus_per_node=224, detach=detach, launch_transport='srun')
    node = {'p0': 'node1800', 'd0': 'node2701', 'd1': 'node2800'}.get(node_id, 'node1800')
    result = site.invoke_multinode(args, path, action, node_id=node_id, slurm_node=node, component=component)
    if result:
        raise RuntimeError(f'{action}/{component}/{node_id} exited {result}')


def prepare_and_start(experiment):
    allocation_check()
    if gpu_process_count():
        raise RuntimeError('refusing launch: one of the three nodes has GPU processes')
    directory = core.run_dir(experiment)
    directory.mkdir(exist_ok=False)
    cfg = build_config(experiment)
    path = directory / ('multinode-v3.json' if experiment['method'] == 'agentic' else 'native-pd-manifest.json')
    core.atomic_json(path, cfg)
    core.atomic_json(directory / 'provenance.json', {
        'template': str(TEMPLATES[experiment['method']]),
        'template_sha256': hashlib.sha256(TEMPLATES[experiment['method']].read_bytes()).hexdigest(),
        'node_jobs': JOBS, 'scope': 'SWE500 source-order T450 engineering comparison',
        'changes': ['1P:1D -> 1P:2D', 'requested concurrency', 'unique run paths/token',
                    'Agentic D2P Host 512 GiB per D node, 1024 GiB total; P2D 256 GiB'],
    })
    if experiment['method'] == 'agentic':
        cfg = multinode.load_config(path)
        # Validate before launching any GPU; persist full effective plan.
        core.atomic_json(directory / 'effective-plan.json', multinode.plan(cfg))
        agentic_action(experiment, 'start-control', detach=True)
        agentic_action(experiment, 'control-ready')
        agentic_action(experiment, 'start-worker', node_id='p0', detach=True)
        site.wait_model_endpoint(cfg['nodes'][0], timeout=1800)
        for node in ('d0', 'd1'):
            agentic_action(experiment, 'start-worker', node_id=node, detach=True)
        agentic_action(experiment, 'wait-ready')
        agentic_action(experiment, 'start-router', detach=True)
        agentic_action(experiment, 'run-workload', detach=True)
    else:
        # Generic component launcher already supports arbitrary node/job maps.
        args = SimpleNamespace(cpus_per_worker=224, launch_transport='srun')
        order = [k for k in ('mooncake-master', 'mooncake-client', 'prefill-0', 'decode-0', 'decode-1', 'router', 'workload') if k in cfg['components']]
        for name in order:
            if name == 'router' and experiment['method'] == 'hicache':
                # Preserve the preceding baseline's post-load advisory cache
                # release, but address each node through its actual job.
                for node, job in JOBS.items():
                    local = dict(cfg, job_id=job, components={k:v for k,v in cfg['components'].items() if v['node']==node})
                    native.drop_model_weight_page_cache(local)
            native.launch_component(args, path, name)
            if name != 'workload':
                native.wait_http(cfg['components'][name]['health'], 1800 if name.startswith(('prefill-', 'decode-')) else 600)


def stop_experiment(experiment):
    directory = core.run_dir(experiment)
    freeze_result(experiment)
    if experiment['method'] == 'agentic':
        for component, node_id in [('workload', None), ('router', None), ('worker', 'd1'), ('worker', 'd0'), ('worker', 'p0'), ('control', None)]:
            # A finished workload's supervisor may already have exited. The
            # subsequent exact-run process/GPU checks remain authoritative.
            try:
                agentic_action(experiment, 'stop', component=component, node_id=node_id)
            except RuntimeError as error:
                core.log(str(error))
    else:
        path = directory / 'native-pd-manifest.json'
        cfg = native.load_manifest(path)
        args = SimpleNamespace(launch_transport='srun')
        for name in ('workload', 'router', 'decode-1', 'decode-0', 'prefill-0', 'mooncake-client', 'mooncake-master'):
            if name in cfg['components']:
                native.control_component(args, path, name, 'stop')


def freeze_result(experiment):
    path = core.run_dir(experiment) / 't450-result.json'
    if path.exists():
        return
    result = result_for(experiment)
    if result is None:
        raise RuntimeError('cannot finalize without a valid T450 result')
    result['details'] = forward_details(experiment, result)
    core.atomic_json(path, result)
    core.atomic_json(path.parent/'t450-progress.json', core.load_progress(path.parent))
    archive_logs(experiment)


def archive_logs(experiment):
    """Preserve in-flight ownership/policy evidence at the T450 stop boundary."""
    directory = core.run_dir(experiment)
    dest = directory/'t450-logs'
    dest.mkdir(exist_ok=True)
    if experiment['method'] == 'agentic':
        cfg = json.loads((directory/'multinode-v3.json').read_text())
        local = Path(cfg['local_root'])/experiment['run_id']
        sources = [('node1800', local/name/'service.log', name) for name in ('prefill-0','router-0','agentic-group-control','workload')]
        sources += [(node, local/name/'service.log', name) for node,name in [('node2701','decode-0'),('node2800','decode-1')]]
    else:
        cfg = json.loads((directory/'native-pd-manifest.json').read_text())
        sources = [(c['node'], Path(c['local_dir'])/'service.log', name) for name,c in cfg['components'].items()]
    for node, source, name in sources:
        remote(node, ['cp', str(source), str(dest/(name+'.log'))])


# Runs only after all GPU processes are gone; exact immutable run scope.
# Never signals another run or the allocation's batch process.
CLEAN_PROBE = r'''
import os,signal,sys,time,json,subprocess
run=sys.argv[1]; clean=sys.argv[2]=='clean'; needle=run.encode()
exclude=set(); pid=os.getpid()
while pid and pid not in exclude:
 exclude.add(pid)
 try: pid=int(open('/proc/'+str(pid)+'/stat').read().split()[3])
 except (OSError,ValueError): break
hits=[]
for name in os.listdir('/proc'):
 if not name.isdigit() or int(name) in exclude: continue
 try:
  if os.stat('/proc/'+name).st_uid != os.getuid(): continue
  cmd=open('/proc/'+name+'/cmdline','rb').read()
  if needle in cmd: hits.append(int(name))
 except OSError: pass
if clean:
 for pid in hits:
  try: os.kill(pid,signal.SIGTERM)
  except ProcessLookupError: pass
 # No broad kill or forced unmount. Next probe must verify every survivor.
 time.sleep(3)
 prefix='/var/tmp/shangy/pd-swe-apptainer/'+run+'/'
 for line in open('/proc/self/mountinfo'):
  parts=line.split(); target=parts[4]
  if target.startswith(prefix) and target.endswith('/rootfs'):
   subprocess.run(['fusermount3','-u',target],capture_output=True)
print(json.dumps({'pids':hits}))
'''


def wait_clean(run_id):
    deadline = time.monotonic() + 300
    clean_samples = 0
    while time.monotonic() < deadline:
        gpu = gpu_process_count()
        counts = []
        for node in JOBS:
            out = remote(node, ['python3', '-c', CLEAN_PROBE, run_id, 'probe']).stdout
            counts.append(len(json.loads(out)['pids']))
        if not gpu and not sum(counts):
            # Also unmount this run's abandoned verifier rootfs after its
            # processes have stopped. Preserve all data/artifacts.
            for node in JOBS:
                remote(node, ['python3', '-c', CLEAN_PROBE, run_id, 'clean'])
            clean_samples += 1
            if clean_samples >= 2:
                return
        else:
            clean_samples = 0
            if not gpu:
                for node in JOBS:
                    remote(node, ['python3', '-c', CLEAN_PROBE, run_id, 'clean'])
        time.sleep(10)
    raise RuntimeError(f'exact-run cleanup incomplete for {run_id}; refusing next launch')


def result_for(experiment):
    directory = core.run_dir(experiment)
    frozen = directory / 't450-result.json'
    if frozen.exists():
        return json.loads(frozen.read_text())
    rows = core.load_progress(directory)
    completed = sorted((r for r in rows if r.get('status') == 'Status.COMPLETED'), key=lambda r: r['completed_at'])
    if len(completed) < 450:
        return None
    marker = directory / 'workload/measurement_start.json'
    if not marker.exists():
        return None
    start = json.loads(marker.read_text())['measurement_start_wall']
    end = completed[449]['completed_at']
    before = [r for r in rows if r['completed_at'] <= end]
    failed = sum(r.get('status') != 'Status.COMPLETED' or bool(r.get('environment_error')) for r in before)
    # Wait for the sampler's next flush on all three endpoints. A partial
    # scrape must not silently yield one-D throughput or truncate T450.
    latest = {}
    metrics_path = directory / 'workload/engine_metrics.jsonl'
    if not metrics_path.exists():
        return None
    for line in metrics_path.read_text().splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue
        for ep in row.get('endpoint_metrics', []):
            if ep.get('metrics'):
                latest[(row['role'], ep['endpoint'])] = row['ts']
    for role, expected in [('prefill', 1), ('decode', 2)]:
        stamps = [t for (r, _), t in latest.items() if r == role]
        if len(stamps) != expected or min(stamps) < end:
            return None
    tokens, tps, source = core.t450_decode_metrics(directory, start, end)
    if tokens is None:
        return None  # Never stop before durable metrics cover both Ds.
    return dict(concurrency=experiment['concurrency'], method=experiment['method'], run_id=experiment['run_id'],
        completed=len(completed), failed=failed, environment_errors=failed,
        solved=sum(float(r.get('reward') or 0) > 0 for r in rows),
        t450_seconds=end-start, t450_decode_tokens=tokens, t450_decode_tps=tps,
        t450_decode_source=source, start=start, end=end,
        t50=completed[49]['completed_at'], t300=completed[299]['completed_at'],
        result_scope='T450-completed-only; not 300s+1200s acceptance')


def forward_details(experiment, result):
    sys.path.insert(0, str(ROOT / 'examples/pd/scripts/tools'))
    from summarize_native_swe_colocated import interval_value
    from collections import defaultdict
    series = defaultdict(list)
    path = core.run_dir(experiment) / 'workload/engine_metrics.jsonl'
    for line in path.read_text().splitlines():
        try: row = json.loads(line)
        except json.JSONDecodeError: continue
        for ep in row.get('endpoint_metrics', []):
            series[(row['role'], ep['endpoint'])].append((row['ts'], ep.get('metrics', {})))
    a, b = result['start'], result['end']
    totals = {}
    for role in ('prefill', 'decode'):
        endpoints = [s for (r, _), s in series.items() if r == role]
        expected = 1 if role == 'prefill' else 2
        if len(endpoints) != expected:
            raise ValueError(f'missing {role} endpoint coverage')
        cat = 'extend' if role == 'prefill' else 'decode'
        acc = dict(forward=0., new=0., cached=0., kv=0., running=0.)
        for s in endpoints:
            s.sort()
            if s[-1][0] < b:
                raise ValueError('metrics have not yet covered T450 for every endpoint')
            fk = next((k for k in [f'sglang_gpu_execution_seconds_total|category=forward_{cat}',
                f'sglang_forward_execution_seconds_total|category={cat}'] if any(k in m for _, m in s)), None)
            if fk is None: raise ValueError('missing device Forward counter')
            acc['forward'] += interval_value(s, fk, a, b, True) / (8 * expected)
            for dest, key, counter in [('new', 'sglang_realtime_tokens_total|mode=prefill_compute', True),
                ('cached', 'sglang_realtime_tokens_total|mode=prefill_cache', True),
                ('kv', 'sglang_full_token_usage', False), ('running', 'sglang_num_running_reqs', False)]:
                if any(key in m for _, m in s):
                    acc[dest] += interval_value(s, key, a, b, counter) * ((b-a) if counter else 1/expected)
        totals[role] = acc
    return totals


def update_report(state, current_progress):
    lines = ['# MiniMax-M2.7 · 三节点 TP8/EP8 · 1P:2D', '',
        f"状态：{state['status']}；{state.get('current_run_id', '')}，{current_progress}/500。", '',
        'P+Router+verifier=node1800；D0=node2701；D1=node2800。24×H100；不跨节点模型并行。',
        '沿用 MiniMax r6 引擎/配置，c256/384/448；SWE500 source-order，temperature=.6/top_p=.95/top_k=20，',
        'context=131072，chunk/batch=8192，单轮8192 token/最多64轮，mem_fraction_static=.8。',
        'Agentic 工具/Direct各1s，拥堵重算关闭；D→P每D节点512 GiB（合计1024 GiB），P→D总256 GiB。',
        'HiCache每rank64 GiB，Mooncake共享store总512 GiB，wait_complete；原生baseline代码不修改。',
        'Agentic沿用r6 compact_handoff_payload=true、profile_host_prepare=true和P→D/Forward串行开关。',
        '指标窗口均为0–T450；T450按第450道COMPLETED计算，FAILED不充数。达到指标后保存快照、停止并切组。',
        '停止前在途Host/传输由t450-logs保留；强制停止不是自然排空或历史queued=durable=release守恒证明。',
        '这是固定500题T450工程对比，不宣称满足300s预热+1200s稳态验收。', '',
        '| 顺序 | 并发 | 方法 | T450(s) | Decode tok/s | 完成/失败(T450前) |',
        '|---:|---:|---|---:|---:|---:|']
    results = []
    for i, exp in enumerate(EXPERIMENTS):
        result = result_for(exp)
        if result:
            results.append((exp, result))
            lines.append(f"| {i+1} | {exp['concurrency']} | {LABELS[exp['method']]} | {result['t450_seconds']:.2f} | {result['t450_decode_tps']:.1f} | 450/{result['failed']} |")
        else:
            lines.append(f"| {i+1} | {exp['concurrency']} | {LABELS[exp['method']]} | — | — | 待完成 |")
    lines += ['', '| 并发 | 方法 | P Forward/GPU | D Forward/GPU | P累计Forward(s/GPU) | 实际Prefill(M) | 复用率 | P KV | D KV均值 | D running/组 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for exp, result in results:
        detail = result.get('details') or forward_details(exp, result)
        p, d = detail['prefill'], detail['decode']
        reuse = p['cached'] / (p['new'] + p['cached']) if p['new'] + p['cached'] else 0
        lines.append(f"| {exp['concurrency']} | {LABELS[exp['method']]} | {p['forward']:.2%} | {d['forward']:.2%} | {p['forward']*result['t450_seconds']:.1f} | {p['new']/1e6:.3f} | {reuse:.2%} | {p['kv']:.2%} | {d['kv']:.2%} | {d['running']:.2f} |")
    if state.get('error'): lines += ['', f"暂停原因：{state['error']}"]
    temporary = REPORT.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    os.replace(temporary, REPORT)


_poll = core.poll


def poll(state):
    index = state['index']
    if index < len(EXPERIMENTS):
        exp = EXPERIMENTS[index]
        result = result_for(exp)
        if result and not (core.run_dir(exp) / 't450-result.json').exists():
            freeze_result(exp)
    finished = _poll(state)
    if not finished and time.time() - state.get('last_started_epoch', time.time()) > 300:
        if gpu_process_count() != 24:
            raise RuntimeError('expected exactly 24 GPU workers across the three nodes')
        if time.time() - state.get('last_progress_epoch', time.time()) > 1800:
            raise RuntimeError('no episode progress for 30 minutes')
    return finished


def configure():
    core.CONTROL_DIR, core.STATE_PATH, core.LOCK_PATH = CONTROL, CONTROL/'state.json', CONTROL/'controller.lock'
    core.REPORT, core.EXPERIMENTS, core.SEED_RUNS = REPORT, EXPERIMENTS, []
    core.MIN_START_SECONDS = 2 * 3600
    core.JOB_ID = JOBS['node1800']
    core.prepare_and_start, core.stop_experiment = prepare_and_start, stop_experiment
    core.wait_clean, core.gpu_process_count, core.job_state = wait_clean, gpu_process_count, allocation_check
    core.update_report, core.poll = update_report, poll
    core.completed_result = lambda *args: None
    core.t450_result = lambda c, m, d: result_for(next(e for e in EXPERIMENTS if e['run_id'] == d.name))


if __name__ == '__main__':
    configure()
    CONTROL.mkdir(parents=True, exist_ok=True)
    if not core.STATE_PATH.exists():
        core.atomic_json(core.STATE_PATH, dict(schema_version=1, index=0,
            status='starting', error='', last_poll_epoch=time.time()))
    raise SystemExit(core.main())
