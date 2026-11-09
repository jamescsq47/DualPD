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

Local Router stage in progress: capacity (33 methods), admission (38) and HTTP
orchestration (27) now share the original concrete instance, with its constructor
and select_pair retained. All100 original method ASTs and14 relocated top-level
nodes reconstruct the complete original module. Runtime global-binding and
wrong-import mutation checks supplement—not replace—that proof. Focused original
Router114 tests plus2 subtests and26 new structural/mutation cases have passed.
Independent code audit GO and an independent26/26 rerun are complete. Full CPU
gate passes208 frontend+481 engine+2 subtests (691 records); node2700 native/
Router regression passes716+2 subtests and2 actual CUDA roundtrips. A new actual
BrowseComp comparison subsequently passes300.861+1200.001 seconds at11206.66
Decode token/s (−0.52% versus ORIGINAL baseline),100% reuse,6668 completions,
0 failures/scoped memory events and clean GPU/port/helper gates. Ownership sets
match12418 P2D Host and21503 D2P Direct snapshots;all10 measured D2P Host durables
restore/release,131 only-drain durables are documented separately. This is a
gross-regression pass, not statistical equivalence. TCP Router is unchanged and still part of stage3;
stages4–8 are not completed by this local extraction.

TCP Router separation is now implemented but not performance accepted: the281-
line concrete root retains8 methods and all decorated callbacks/main/service
guard; capacity12 and HTTP9 methods share the same instance. Whole29-method/
module reconstruction, exact imports and untouched original15 tests pass. The
focused37 cases and independent37/37 give code GO; full CPU713 records, native740
records, actualCUDA2 and9 launch plans pass. The next frozen local BrowseComp
case is a nonregression check only, not TCP model-path throughput validation.
Representative actual TCP coverage and all later stages remain required.

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

## After the D-writer performance gate

Independent read-only scoping recommends advancing to stage3 Router rather than
benchmarking more Host helper moves. P-manager remains the single P-side
lifecycle coordinator for admission, workset leases, restoration, cancellation
and release; underlying DMA/registration/buffers/ledger/D-writer are distinct.
Its size alone is not a reason to split method ranges mechanically. Numerous
explicit `AgenticPHostStagingManager.method(self, ...)` calls support namespace
fixtures and class-level monkeypatches; naive mixins/global rebinding would
change those hooks. Final review may still require an explicit complete H2D
operation-state component with compatibility adapters, not copied helper bodies.

Stage3 should separate Router capacity/reservations, admission and HTTP
orchestration, preserving local versus TCP policies and all queue/timeout/cancel
branches. Keep the pinned legacy Router byte-identical. Any new source proof
must reconstruct the complete original class/method bodies in original order;
do not simply exempt modified Router files from hash/serving-AST verification.

Stage5 must move the coherent metadata/storage definitions AND change pure
control-plane imports to ledger/protocol leaves (including P2D constants). Moving
metadata alone is not proof of CUDA independence: the writer-stage H100 cold
import trace demonstrated the inherited native dependency initialization chain.
No source changes for these later stages occur while writer-r1 is frozen.

Writer-r1 subsequently passed301.030+1200.020 seconds at11214.24 Decode token/s
(−0.45% versus original baseline),100% parent reuse and0 failures/OOM, with final
GPU/port/helper gates empty and controllers gone. This releases the next code
stage's freeze; actual Router organization and its own performance acceptance
remain required. Stages3–8 have not been completed by the Host work.

Router pre-scope found a specific archival-entry constraint: standalone
`launch_late_binding_router.py` loads the maintained Router file under
`dualpd_engine_late_binding_router` so native imports remain on the selected old
overlay. Relative helper imports need a file-as-package spec with
`submodule_search_locations=[routerdir]`, inserted in sys.modules before exec;
do not append an active checkout to sys.path or replace the native overlay.
Test both canonical-package and private-file entry paths. Preserve exact logger
names for both paths, not helper `__name__`. Keep concrete Router `__init__`
and MiniLoadBalancer last in its MRO; extracted mixins must not add constructors
or super calls. Move `_PrefillRedirect` intact because its own zero-argument
super belongs to that exception class. Existing instance/class hooks must still
resolve through the canonical concrete Router.

The Router legacy-load test directly rebinds root `_sync_json_get` in
`test_legacy_load_network_wait_runs_outside_router_loop` (not a monkeypatch API).
If `_fetch_decode_load_legacy` moves, retarget only that defining-module hook and
retain all assertions, especially network work running outside the ASGI thread.
An old-root function reexport alone does not forward global rebinding.
