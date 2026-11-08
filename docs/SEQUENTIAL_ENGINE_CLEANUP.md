# Sequential engine cleanup and BrowseComp validation

The active user goal is to organize the remaining code in stages, checking
BrowseComp throughput after each stage before continuing. CPU/AST checks alone
do not satisfy the performance gate. Existing actual feature/ablation switches
and local/TCP paths must remain supported.

## Allocation

The user requested one dedicated H100 node for seven days, reused for all
subsequent tests. `scripts/cluster/engine_cleanup_7day.sbatch` requests one node,
eight H100s, all 224 logical CPUs, 1800 GiB RAM and exclusive access in
`pi_songhan`. All CPUs are requested so GPU-local NUMA bindings can use
both sockets. A heartbeat
keeper records the assigned node and job ID; it does not start GPU workloads
or clean unknown processes. Do not use other allocations without a new reason
and do not cancel another user's jobs.

## Required stages

1. Establish a real BrowseComp quick-check baseline on the dedicated node with
   the current extracted repositories. Freeze the input order, sampling, model,
   concurrency, topology, Host arenas and measurement protocol for comparisons.
2. Separate Host transport/arena/registration responsibilities, one coherent
   extraction at a time, maintaining fences and historical import identities.
3. Separate Router admission, capacity/load policy and HTTP orchestration.
4. Extract Scheduler extension responsibilities, leaving explicit native
   integration hooks rather than large inline agentic implementations.
5. Remove control-plane dependencies on DMA/worker modules where only protocol
   constants or ledger types are needed.
6. Consolidate configuration parsing/validation while retaining actual switches,
   defaults and the distinct local/TCP policy scopes.
7. Separate maintained launchers from historical replay scripts; replace
   machine-specific paths only where doing so does not change replay identity.
8. Re-run final functionality and representative BrowseComp validation, inspect
   all stage results, source maps, package entries and cleanup gates.

Each stage needs relevant lifecycle/fault tests, source-equivalence checks,
independent audit GO, and a real before/after BrowseComp throughput comparison.
A regression requires investigation or a repeated comparison before advancing.
Quick checks are not new 300+1200 formal benchmark results. Record their exact
windows and uncertainty separately. Do not declare completion after merely
finishing the first safe extraction or while GPU validation is still missing.

## Reproducible runner

`scripts/cluster/engine_cleanup_quickcheck.py` reuses the historical successful
BrowseComp Qwen3-8B TP1 4P:4D configuration at c256: temperature 0, seed 2026,
fixed 680-source schedule with closed-loop recycling, context 40960, response
limit 36864, D→P Host 512 GiB total and P→D Host 256 GiB total, tool/admission
thresholds 1 s, congestion thresholds 32/8. The first comparison deliberately
retains the complete 300 s warmup + 1200 s measurement, avoiding a shorter
window masquerading as the historical steady-state result.

Prepare each unique attempt with `--prepare --plan /scratch/attempt/plan.json
--job-id 28285059 --stage baseline`. The plan records the exact source revision,
all frontend source files (including untracked modules), effective environment,
input hashes and isolated paths. Refactor stages additionally require
`--reference /scratch/before/result.json`. Obtain independent audit GO for the
exact plan SHA256 in a sibling `audit.json`, then use `--wait-for-job --plan
/scratch/attempt/plan.json`. The controller polls the existing job every 30 s,
starts one overlapping step only after RUNNING, and never submits replacements.
Its progress is durable in `controller.json` and `runtime.json` rather than
repeated chat output.

On-node checks require the assigned node, eight H100s, CPU affinity covering
every GPU-local NUMA domain, one fixed node-local per-job lock, and three clean
resource probes before and after the run. On error only the exact owned
launcher process group receives TERM, reaching the existing pipeline's ordered
shutdown traps. Unknown processes are never signalled. A dirty cleanup gate
stops the sequence. Decode metrics stale for 90 s or no growth for 300 s during
an active workload also stop the current run; post-window draining is excluded.

The result retains the existing zero-measured-failure, positive tail120 Decode,
OOM and full-window gates. A >10% Decode drop is recorded as requiring repeat
or investigation, not accepted. This threshold detects gross regressions; a
single pass is not a claim of statistical equivalence. The final stage still
requires representative functionality/ownership validation as well as throughput.
