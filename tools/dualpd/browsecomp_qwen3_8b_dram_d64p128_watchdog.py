#!/usr/bin/env python3
"""Reuse the independent DRAM watchdog for one new-method follow-up."""
import os
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import browsecomp_qwen3_8b_dram_h100_watchdog as source

watch = source.watch
watch.CONTROL = Path("/orcd/scratch/orcd/003/shangy/dualpd/runs/h100-browsecomp-qwen3-8b-dram-d64p128-c512-20261028")
watch.STATE = watch.CONTROL / "state.json"
watch.HEARTBEAT = watch.CONTROL / "heartbeat.json"
watch.WATCH_STATE = watch.CONTROL / "watchdog-state.json"
watch.WATCH_LOG = watch.CONTROL / "watchdog.jsonl"
watch.EXPECTED = 1


def ssh_resource_probe(job_id, node):
    # A monitor must not wait for a second Slurm step's resources while the
    # batch step owns the allocation. SSH is already used by the DRAM probe.
    status, allocated = watch.allocation(job_id)
    if status != "RUNNING" or allocated != node:
        raise RuntimeError(f"probe target no longer allocated: {job_id} {status} {allocated}")
    item = watch.current_item(watch.read_json(watch.STATE))
    directory = str(item[1].get("run_dir", "")) if item else ""
    program = r'''import json,os,subprocess,sys,time
from pathlib import Path
gpu=[x for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines() if x.strip()]
ports=set()
for line in subprocess.check_output(['ss','-H','-ltn'],text=True,timeout=10).splitlines():
 try: ports.add(int(line.split()[3].rsplit(':',1)[1]))
 except (IndexError,ValueError): pass
helpers=[]; workloads=closed_loop=0
for entry in os.listdir('/proc'):
 if not entry.isdigit(): continue
 try:
  comm=open('/proc/'+entry+'/comm').read().strip()
  if comm in {'apptainer','squashfuse_ll','fuse-overlayfs'}: helpers.append(comm)
  cmd=open('/proc/'+entry+'/cmdline','rb').read().split(b'\0')
  if any(x == b'dualpd.inference.py' or x.endswith(b'/inference.py') for x in cmd):
   workloads+=1
   if b'--closed-loop' in cmd: closed_loop+=1
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
mem={}
for line in open('/proc/meminfo'):
 key,value,*_=line.replace(':','').split()
 if key in {'MemAvailable','Committed_AS','CommitLimit'}: mem[key]=int(value)
result={'gpu_processes':len(gpu),'ports':sorted(ports),'helpers':len(helpers),'workloads':workloads,'closed_loop_workloads':closed_loop,'mem_available_kib':mem.get('MemAvailable',0),'commit_ratio':mem.get('Committed_AS',0)/max(mem.get('CommitLimit',1),1)}
directory=Path(sys.argv[1]) if sys.argv[1] else None
if directory and directory.is_dir():
 boundary=directory/'closed_loop_boundaries.json'
 result['measurement_complete']=boundary.exists() and json.loads(boundary.read_text()).get('status')=='steady_churn'
 errors=[]
 for path in (directory/'logs').glob('*.log'):
  with path.open('rb') as stream:
   stream.seek(0,2); stream.seek(max(0,stream.tell()-262144)); tail=stream.read().decode(errors='replace')
  if 'quarantining source KV' in tail or 'NIXL_ERR_REMOTE_DISCONNECT' in tail:
   errors.append(path.name+': NIXL disconnect/quarantined KV')
 result['transport_errors']=errors
 frames=[]
 path=directory/'engine_metrics.jsonl'
 if path.exists():
  with path.open('rb') as stream:
   stream.seek(0,2); stream.seek(max(0,stream.tell()-4194304)); lines=stream.read().splitlines()
  for line in lines:
   try: row=json.loads(line)
   except ValueError: continue
   endpoints=row.get('endpoint_metrics',[])
   key='sglang_realtime_tokens_total|mode=decode'
   if row.get('role')=='decode' and len(endpoints)==4 and all(key in e.get('metrics',{}) for e in endpoints):
    frames.append((row['ts'],sum(e['metrics'][key] for e in endpoints)))
  if frames:
   last=frames[-1]; window=[r for r in frames if r[0]>=last[0]-120]
   result['decode_counter_age_seconds']=time.time()-last[0]
   result['decode_tail_span_seconds']=last[0]-window[0][0]
   result['decode_tail_tokens']=last[1]-window[0][1]
print(json.dumps(result))'''
    command = ["python3", "-c", program, directory]
    try:
        response = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                   node, shlex.join(command)], text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=40, check=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"SSH resource probe timed out after 40 seconds on {node}") from None
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"SSH resource probe failed on {node}: {exc.stderr[-500:]}") from None
    result = json.loads(response.stdout)
    if result.get("transport_errors"):
        raise RuntimeError("transport failure: " + "; ".join(result["transport_errors"]))
    if not result.get("measurement_complete") and result.get("decode_tail_span_seconds", 0) >= 90 and result.get("decode_tail_tokens", 1) <= 0:
        raise RuntimeError("no Decode counter growth in the last >=90 seconds of complete frames")
    if not result.get("measurement_complete") and result.get("decode_counter_age_seconds", 0) > 120:
        raise RuntimeError("no complete Decode counter frame in the last 120 seconds")
    return result


source.original_probe = ssh_resource_probe


def queue(thread, message):
    message = message.replace("P:D缩放消融已完成12/12", "DRAM新方法D64/P128已完成1/1")
    message = message.replace("P:D缩放矩阵", "DRAM新方法D64/P128实验")
    return source.original_queue(thread, message)


watch.queue = queue


def main():
    job_id = sys.argv[sys.argv.index("--job-id") + 1]
    watch.CONTROL.mkdir(parents=True, exist_ok=True)
    watch.append_event(dict(event="startup_wait", pid=os.getpid(), job_id=job_id))
    deadline = None
    while True:
        status, _ = watch.allocation(job_id)
        if status == "PENDING":
            time.sleep(30)
            continue
        if status != "RUNNING" or watch.HEARTBEAT.exists():
            break
        deadline = deadline or time.monotonic() + 180
        if time.monotonic() >= deadline:
            break
        time.sleep(5)
    return watch.main()


if __name__ == "__main__":
    raise SystemExit(main())
