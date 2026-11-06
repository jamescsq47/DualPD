#!/usr/bin/env python3
"""Reuse the quiet matrix watchdog, scoped to the four DRAM cases."""
from pathlib import Path
import json
import shlex
import subprocess
import browsecomp_qwen3_8b_pd_scale_h100_watchdog as watch

watch.CONTROL = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-c512-20261022")
watch.STATE = watch.CONTROL / "state.json"
watch.HEARTBEAT = watch.CONTROL / "heartbeat.json"
watch.WATCH_STATE = watch.CONTROL / "watchdog-state.json"
watch.WATCH_LOG = watch.CONTROL / "watchdog.jsonl"
watch.EXPECTED = 4
watch.expected_gpu_processes = lambda key: 8
original_queue = watch.queue
original_probe = watch.node_probe


def observation_probe(node, directory):
    # Read on the writer's node: the login node can observe stale NFS file
    # lengths while the collector keeps its line-buffered stream open.
    program = r'''import json,sys,time
from pathlib import Path
directory=Path(sys.argv[1])
boundary_path=directory/'closed_loop_boundaries.json'
boundary=json.loads(boundary_path.read_text()) if boundary_path.exists() else {}
if boundary.get('status')=='steady_churn' or not (directory/'engine_metrics.jsonl').exists():
 print(json.dumps({'observation_phase':'startup_or_postprocess'})); sys.exit(0)
with (directory/'host_resources_2s.jsonl').open('rb') as stream:
 stream.seek(0,2); length=stream.tell(); stream.seek(max(0,length-1024*1024))
 lines=stream.read().splitlines()
# A concurrent append may leave a partial final line; use the last complete row.
last=None
for line in reversed(lines):
 try: last=json.loads(line); break
 except ValueError: pass
if last is None: raise RuntimeError('No complete DRAM samples')
print(json.dumps({'dram_sample_age_seconds':time.time()-last['ts'],
 'dram_pool_count':last['pool_count'],'dram_sample_valid':last.get('valid_sample'),
 'dram_pool_errors':last.get('pool_errors'),
 'pss_sample_valid':last.get('process_sample_valid'),
 'pss_sample_age_seconds':last.get('process_sample_age_seconds'),
 'pss_sample_error':last.get('process_sample_error')}))
'''
    command = ["/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd/bin/python3.12",
               "-c", program, str(directory)]
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                             node, shlex.join(command)], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=30, check=True)
    return json.loads(result.stdout)


def node_probe(job_id, node):
    result = original_probe(job_id, node)
    state = watch.read_json(watch.STATE)
    item = watch.current_item(state)
    if not item or item[1].get("status") != "running":
        return result
    directory = Path(item[1]["run_dir"])
    observation = observation_probe(node, directory)
    result.update(observation)
    if observation.get("observation_phase"):
        return result
    age = observation["dram_sample_age_seconds"]
    if age > 15 or not observation.get("dram_sample_valid"):
        raise RuntimeError(f"DRAM observation unhealthy: age={age:.1f}s pools={observation['dram_pool_count']} "
                           f"errors={observation.get('dram_pool_errors')}")
    if not observation.get("pss_sample_valid"):
        raise RuntimeError(f"PSS observation unhealthy: age={observation.get('pss_sample_age_seconds')} "
                           f"error={observation.get('pss_sample_error')}")
    return result


def queue(thread, message):
    return original_queue(thread, message.replace("P:D缩放", "DRAM消融").replace("12/12", "4/4"))


watch.queue = queue
watch.node_probe = node_probe
if __name__ == "__main__":
    raise SystemExit(watch.main())
