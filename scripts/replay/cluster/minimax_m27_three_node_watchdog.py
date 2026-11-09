#!/usr/bin/env python3
"""Silent read-only monitor; wake the owning thread only on persistent faults."""
import argparse
import fcntl
import json
import os
import subprocess
import time

if __package__:
    from . import minimax_m27_three_node_autopilot as matrix
else:
    import minimax_m27_three_node_autopilot as matrix


def sample():
    path = matrix.CONTROL / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {'status': 'starting'}
    status = state['status']
    if status == 'complete': return state, 'complete'
    if status == 'paused': return state, 'controller paused: ' + state.get('error', '')
    pids = subprocess.run(['pgrep', '-f', 'minimax_m27_three_node_autopilot.py'], capture_output=True, text=True, timeout=15)
    if not pids.stdout.strip(): return state, 'controller process missing'
    _, seconds = matrix.allocation_check()
    if seconds < 7200: return state, f'allocation expires in {seconds}s; do not acquire other nodes'
    snapshots = {}
    for node in matrix.JOBS:
        output = matrix.remote(node, ['python3', '-c',
            "import json; m=dict((x.split(':')[0],int(x.split()[1])) for x in open('/proc/meminfo') if ':' in x); print(json.dumps({'available_kib':m['MemAvailable'],'total_kib':m['MemTotal']}))"], timeout=45).stdout
        snapshots[node] = json.loads(output)
        if snapshots[node]['available_kib'] < 128 * 1024**2:
            state['node_memory'] = snapshots
            return state, f'low memory on {node}: {snapshots[node]["available_kib"]} KiB available'
    state['node_memory'] = snapshots
    # Startup/cleanup transitions can legitimately have fewer than 24 ranks.
    # Controller performs strict GPU checks after a successful startup.
    age = time.time() - float(state.get('last_poll_epoch', time.time()))
    if age > 3600: return state, f'controller heartbeat stale for {age:.0f}s'
    return state, ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--interval-seconds', type=int, default=600)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    matrix.CONTROL.mkdir(exist_ok=True, parents=True)
    lock = (matrix.CONTROL/'watchdog.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous, repeats, last_sent = '', 0, 0.
    while True:
        try:
            state, issue = sample()
        except Exception as error:
            state, issue = {}, f'probe error: {type(error).__name__}: {error}'
        # Numeric details (remaining seconds/available bytes) change between
        # samples; persistent warnings must not reset merely for that reason.
        repeats = repeats + 1 if issue and previous else 1
        previous = issue
        event = {'time': time.time(), 'state': state, 'issue': issue, 'repeats': repeats}
        alert = issue and (repeats >= 2 or state.get('status') in ('paused', 'complete'))
        if alert and time.time() - last_sent >= 1800:
            message = (f'[MiniMax 1P:2D watchdog] {issue}. '
                f'Inspect {matrix.CONTROL}/state.json and controller.log; only node1800/node2701/node2800 '
                'are authorized. Diagnose, repair if safe and continue the nine-run matrix; '
                'do not change engine semantics or baseline code silently. Normal progress needs no user message.')
            if not args.dry_run:
                result = subprocess.run(['codex', 'queue', '--thread', args.thread_id, '--message', message],
                    capture_output=True, text=True, timeout=60)
                event.update(alert_sent=result.returncode == 0, alert_output=(result.stdout+result.stderr)[-2000:])
                if result.returncode == 0: last_sent = time.time()
        matrix.core.atomic_json(matrix.CONTROL/'watchdog-state.json', event)
        with (matrix.CONTROL/'watchdog.log').open('a') as out:
            out.write(json.dumps(event, ensure_ascii=False)+'\n')
        if args.once or (issue == 'complete' and last_sent): return 0
        time.sleep(max(60, args.interval_seconds))


if __name__ == '__main__':
    raise SystemExit(main())
