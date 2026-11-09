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
Keep the original accepted baseline as the performance reference across stages,
not only each immediate predecessor, so small sequential drops cannot hide
cumulative regression. Record neighboring-stage results as additional context.
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
exact plan SHA256 in a sibling `audit.json`. The current verified launch path is
detached SSH to the allocated node with `--on-node --plan /scratch/attempt/plan.json`;
verify its actual adopted Slurm cgroup, memory limit, PPID1 and live progress
after SSH disconnects. A login-side background process tied to a Codex session
is not a durable controller. The optional `--wait-for-job` path still polls the
existing job every30 s and starts an overlapping step only after RUNNING; it is
not the transport used by the accepted baseline/DMA runs. Neither path submits
replacement allocations.
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

## Current Host work sequence

The DMA ABI and registration/prewarm layers passed actual throughput comparisons
against the original baseline (−0.16% and−0.40% Decode respectively). The physical
buffer/Arena layer passed651 CPU records,543 native fault cases and2 real tiny
H100 registered/bounce roundtrips. Its independently audited full BrowseComp
comparison `host-buffers-r1` passed at11254.37 Decode token/s (−0.09% versus
original baseline),100% parent reuse,0 failures and clean GPU/port/helper gates
on job28285059/node2700. None of these
events completes the remaining Host cleanup or stages3–8 above.

After registration performance acceptance, the next coherent extraction is the
physical snapshot/buffer/Arena layer: indexed DMA descriptors, CPU copy workers,
launch fence, staging/bounce buffers, lazy/materialized snapshot views and
extent suballocation. These share mapping/fence lifetimes and should be tested
as a complete layer, not through a benchmark for every helper. Keep the current
snapshot classes and historical imports identical; move private-hook test patches
only to their defining module. In particular, hybrid tests must still patch
Manager hooks in Host staging while snapshot-factory hooks move with the snapshot
implementation. Native TP, partial-launch/fence, descriptor, hybrid-state and
Host release/retry tests must pass before a separate actual performance check.

After physical-buffer performance acceptance, the next proposed coherent Host
boundary is the D writer/optional Relay role: four exact nodes
`_LEDGER_ENTRY_UNSET`, `_cleanup_nixl_room`, `AgenticDRelayWorker` and
`AgenticDHostStagingClient`. The new writer module should import ledger, buffers,
DMA, lifecycle/base/utils directly, never import staging back. It owns the one
sentinel object used by both the `progress` default and identity check; staging
reexports that same object and both classes. Preserve the historical logger.
Existing class/instance hooks remain valid by alias identity. In particular,
`test_agentic_hybrid_dma.py`'s `_device_indices_to_host` patch is a P-manager hook
and must remain at staging; do not mechanically retarget all private hooks.
Scope was independently reviewed read-only during the preceding frozen model
run; this is not permission to edit before its performance gate passes.

Writer extraction needs explicit sentinel/default-argument identity and cold
import/legacy pickle tests, omitted-vs-explicit-None-vs-provided-entry/local-only
progress checks, partial-launch drain/quarantine, CPU commit/unregister retry,
TP peer abort and failed drain ACK, plus Relay room/close ordering. Existing
production behavior is the reference, not a reason to weaken new assertions.

Independent scoping also identified a pre-existing optional-Relay risk:
`_fail_active()` publishes fallback in `finally` even if `_finish_active()` fails
to synchronize. Do not silently alter it in an AST-identical extraction or claim
it fixed. Retain it as a separately scoped fault-test/ownership analysis item;
any semantic repair requires its own lifecycle mapping and independent audit.
