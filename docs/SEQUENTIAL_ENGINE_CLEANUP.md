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

## Current accepted state

Stages1–4 have accepted full300+1200 local BrowseComp comparisons. The latest
Scheduler-components-r2 gives11238.17 Decode token/s, −0.24% versus ORIGINAL
baseline,100% parent reuse, no execution failure/OOM and clean final gates.

Stage5 is implemented atclean engined265f7946aa3d1369ecdadf9d2d39a77c84e300b:
CPU813, real-kernel native1018 and actual H100 DMA2 pass with independent code
and exact-plan GO. Its control-dependencies-r1 model comparison is accepted:
Decode11223.49 token/s (−0.37% versus ORIGINAL), Prefill75081.77,100% parent
reuse,6665 protocol successes, zero failures/OOM and clean final resources.

Stage6 is now formally applied after that acceptance: trusted-loader failure cannot
partially select/apply a profile, and16 exactly common case argv settings share
one array builder. Whole-file proof,14 destructive mutations and29 actual Bash
argument comparisons pass independent review. Same-IP cross-group listener
validation also closes three missing checks without changing valid plans.
Existing defaults, overrides,
case-layer profile scope and local/TCP policy differences are preserved. Formal
old/new tests now use immutable references, automatic roots and honest argv-only
scope. Full CPU928 and native1018 records, actual H100 DMA2 and independent
CODE GO pass. Stage6 case-configuration-r1 also passes independent exact-plan
and final model acceptance:300.973000+1200.001009 seconds, Decode11212.44
token/s (−0.47% versus ORIGINAL),100% parent reuse,6658 protocol successes,
0 execution failures/OOM and clean final GPU/ports/helpers/owned-process gates.
Complete request and ownership records are saved with the new pool case.

Stage7a is now independently performance accepted: current quickcheck
uses explicit-root scripts/benchmarks recipe/measurement modules instead of
importing historical controllers. Complete immutable three-module proof,
18 source mutations and54 focused tests+33subtests pass. Three accepted real
logs reproduce all25 summary values old/new exactly with fixed-zero cgroup
fixtures. Direct isolated-Python entry and foreign-checkout rejection pass.
Full CPU990 records, real-kernel native1018, actual H100 registered/bounce2,
88 shell syntax checks and independent final CODE GO now pass. The native
testcase inventory exactly matches the accepted stage6 run. An initial native
preflight was mistakenly invoked on the login host and stopped at missing
libcuda before tests; no environment or engine was changed, and the corrected
SSH H100 invocation passes. Committed frontend6bc1523 and the new exact plan
receive independent GO. Benchmark-components-r1 completes300.721429+1200.000559
seconds on retained node2700: Decode11193.29 token/s (−0.64% versus ORIGINAL),
Prefill75079.41,100% parent reuse,6650 protocol successes and0 failures/OOM.
Independent full request/ownership/source audit and fresh final resource checks
pass. Controller1090276/launcher1090300 are gone, GPU/ports/helpers empty;
allocation keeper remains. Saved acceptance-audit.json releases this serving
freeze for stage7b source work, not its next unaudited GPU launch.

During that serving-source freeze, an external immutable6bc1523 fixture-only
draft prepares the first historical replay closure: five BrowseComp shells
under scripts/replay/new_method, plus the paired V1 smoke/helper-test files
under scripts/replay/cluster. The five-shell candidate changes only root depth
and the base recipe's call back to the maintained4P4D case. All28 complete
environment/argv comparisons pass (four entries and ten ablations, each with
defaults and explicit overrides); invalid ablation still exits2. The real Bash
and YAML loader run, but fingerprint checks are stubbed and the terminal serving
pipeline is absent/captured. Initial strict Python-stub path assumptions were
corrected to the actual literal new_method/../tools paths; no runtime source
or setting was changed. Four immutable V1 exact-output-ID helper tests also
pass without invoking its run() or native/RPC/model code. These are external
preparation evidence only, not applied migration, V2 TCP coverage or throughput
acceptance. The remaining cross-family controllers still need full dependency
review; this small closure is not a substitute for stage7 as a whole.
Independent preparation rerun reproduces the entire saved28-comparison result
and verifies unchanged cluster/new_method checkout hashes around execution.
Preparation GO is limited to the temporary five-shell closure; actual relocation,
external callers/old-path compatibility, native readiness, resource cleanup and
model acceptance remain unproven.

A supplemental external immutable6bc1523 AST inventory now records201 files,
176 import/literal-path candidate edges and109 dynamic loader/path call sites
in stage7-import-dependencies.json. Unlike the earlier filename-only search,
it includes bare-import edges, e.g. DRAM watchdog→scale watchdog and MiniMax
controllers→Qwen controller/public Slurm entries. These are review candidates,
not a resolved dependency graph: shell callers, computed paths, sys.path/global
rebinding and external pinned snapshots still require manual interpretation.
No entry is classified obsolete or authorized for deletion by these counts.

Further static review finds operational replay hazards, not safe CPU test
entries: qwen35_35b_a3b_single_node_2p2d_controller_reload_guard.sh contains a
historical fixed controller PID and TERM action. Never execute it to test a
relocation. Several PPD wrappers still select the old workspace's Slime tools
and examples/pd directories; those pre-existing experiment/source selections
cannot be rewritten as if they were ordinary references to the current
standalone project. Retain their historical identity, label reauthorization and
address review requirements, and test only intercepted trusted fixtures.

Stages7–8 remain open: maintained versus replay entry classification/archival, installed
package/source/reference audit, actual
TCP/TP2 and hybrid model-path coverage, and final representative performance and
cleanup acceptance. None is replaced by local TP1 results or launch dry plans.

Stage7b's independent complete destination proposal now covers201/201 original
launcher files with verified immutable hashes and no destination collisions:
142 historical replay files (122 cluster+20 new_method) move as one coherent
closure;59 maintained/core/shared/test/compatibility files stay at canonical
addresses. The earlier seven-file fixture is only a tested subset, not the
planned end state. External stage7b-complete-mapping-audit.json and
stage7b-semantic-migration-audit.md cover cross-model mutable module identity,
DRAM retained-tool copies, fused/checkpoint helpers, outside callers and PPD
source-selection hazards. The142 address moves are now applied in the worktree;
complete migration/runtime replay GO is not yet claimed.

Stage7b work-in-progress has142 old paths absent and142 new paths present,
91 Python AST parses and50 Shell/Slurm syntax checks passing. Shared/core files
stay canonical. Imports now separate core alias identity from historical sibling
package/direct selection;39 files receive the latter repair, including removal
of import-only broad ImportError fallbacks that could mask real core failures.
Six guarded real imports cover MiniMax, ratio12 and DRAM callers in direct and
package modes, with no engine import, environment mutation or operational action.
The maintained benchmark's whole immutable proof remains intact: one historical
module is byte-identical and the other has exactly two address edits. Two added
negative mutations reject regression of those addresses.

An immutable6bc temporary fixture independently reproduces the existing sequence
test failure: its assertion expected Mamba0.9 while its launcher already set0.5.
Only that stale test expectation is corrected; runtime parameters are unchanged.
The sequence/V1 helper/fused-topology focus passes29 tests. Complete201-entry
source inventory now reconstructs all142 moved files and byte-checks56 retained
code files, both outside callers, owner executable bits and finite pinned edits.
Extra non-cache files, symlinks and unsupported file types are rejected. The
owner-mode and extensionless/native-file bypasses found by independent audit
are fixed and independently rechecked, with negative cases.

The new core loader test suite passes14 tests plus33 subtests, including12
guarded real-caller direct/package and normal/failure combinations, original
exception identity, missing/invalid/dependency failure rollback, parent-only
reuse, pickle/logger identity and20 concurrent calls through two helper copies.
Concurrency does not cover unrelated ordinary imports. Five actual archived
shells pass28 full environment/argv A/B comparisons against immutable6bc1523,
with defaults and explicit overrides, all10 BrowseComp ablations and invalid
variant rejection. Real Bash/YAML execute in isolated temporary fixtures; the
terminal pipeline is absent, fingerprint validation is explicitly stubbed and
support scripts are immutable originals. Independent audit scoped GO covers
these two tests, not overall source/runtime or GPU acceptance.

The shell tests additionally assert current maintained SUPPORT bytes equal the
immutable copies. A second test now covers7 more wrappers (mixed5 ablations,
Qwen3.5-27B BrowseComp/SWE and Qwen3-32B BrowseComp/recheck/SWE/Terminal),22 pairs
at the common-case boundary. It never copies or runs the common case/pipeline.
Only WORKLOAD_CONFIG and RUN_DIR spellings are resolved for comparison; all
other environment values and final argv remain exact. Both shell tests pass
2 tests+50 subtests; independent review and an independent2-test/50-comparison
rerun accept this expanded scope. This is not blanket historical replay GO.

The remaining6 complex entrypoints now have15 byte-exact configuration-prefix
A/B comparisons, stopping before first mkdir. The first100 wrapper delegates
to the truncated full entry. Topology conflict, invalid lane count and missing
plugin fail-closed checks pass. Runtime support is byte-identical; Colocated
installs and executes real empty-registry EXIT cleanup, explicitly asserted
empty with a forbidden-kill sentinel. No registered-service stop path or
launcher-specific cleanup suffix is exercised. The27B NIXL path is relocated
only in temporary copies with an empty existence sentinel; real guard text is
checked, but native plugin validity is not established. Docker/preflight/CUDA,
source snapshots, control-dir creation and model launch remain unexecuted.
Independent scoped audit and independent1-test/15-comparison rerun accept this
test and corrected cleanup description. All18 archived Shell entrypoints now
have explicitly different contract depths (5 pipeline,7 case-boundary,6 config
prefix), not18 complete operational replays.

Fourteen further finite address-edit sets now repair8 sbatch controller helpers
and6 PPD TOOLS directories;5 PPD controllers also get the maintained summary
helper address. These exact edits leave Slime harness, PR source/SIF, recorded
engine/model/data/run identities and runtime flags unchanged. Independent review
accepted these14 exact changes and independently reran the complete source
proof; fresh combined source/negative/contract tests and full CPU gates pass.
Frontend CPU regression passes290 tests+263 subtests in153.19s; the engine group
passes568 tests+37 subtests in127.11s (7 non-fatal CPU/deprecation warnings).
The full CPU command exits0. This frontend collection predates the added
second shell test, which has a separate successful focused/independent run.
The focused source/core/shell group before the second shell-test
expansion passed22 tests+144 subtests; the source snapshot verifier passes with
no unexpected differences. Missing old
pytest/dependency paths remain historical replay limitations, not silently
validated entrypoints. Complete historical native/operational suffix execution
is not claimed. Independent overall archival CODE GO now accepts142-file
archival and unchanged maintained-entry runtime, with a separate independent
23-test/166-subtest core/layout/shell run. This is not GPU launch authorization.
The latest combined CPU gate completes with292 frontend tests+300 subtests
(157.37s) and568 engine tests+37 subtests (129.08s), overall exit0 and7 non-fatal
warnings. The unchanged engine/native fixtures retain their accepted native1018
and actual H100 CUDA2 evidence; independent exact-plan preaudit confirms that
reuse is sufficient for this pure archival step, not new model-family coverage.
The new replay-archive-r1 plan is prepared with SHA256
2786d1d9af3333e400c7f6f92859d6fde0272ca8f03d5bd882ff00cd706ac291;
source/99env/input/ORIGINAL reference and clean owned-node preflight match.
Final post-commit exact-plan launch GO and the new full BrowseComp comparison
remain open. ActualTCP/TP2/hybrid and final
package/source/reference integration remain later required stages.
No stage7b GPU run has started.

Independent import-boundary design review recommends a seven-name core/shared
fileloader rather than globally adding core to sys.path: process_supervisor,
qwen35_slurm, multinode, qwen35_native_pd_slurm, qwen35_colocated_slurm,
dram_resource_sampler and dram_complete_frame_analysis. Reuse an existing
correct-origin bare/public object without renaming its metadata; reject already
split identities or wrong origins. New loads must respect the small real
dependency graph, bind real parent-package attributes, and roll back only newly
created aliases on failure while propagating the original import error.
Historical sibling imports stay siblings; only core references move to this
boundary. Direct/package/mixed-name support is a new tested compatibility
guarantee, not an assertion that the old scripts always supported every form.
The loader is now applied, with scoped identity/fault test audit GO but not yet
overall stage7b CODE GO. The44 root-depth uses also include two
separately reviewed special cases: ratio-test source lookup and sequence
LAUNCHER.parents[2], both requiring depth3 after archival while preserving input
and launcher identities.

An external replay-loader-draft now implements that seven-name boundary without
touching the checkout: seven fixture test groups and14 fresh-process imports of
the real canonical tools pass. The strengthened r2 bare-first cases perform an
ordinary bare import before calling the loader (the test harness alone
temporarily recreates the original core search path, then restores it); they
are not merely loader-first self-comparisons. Public-first, source/conflict
rejection, dependency function identity and unchanged loader sys.path are also
checked. Independent preparation review accepts the limited design, not actual
replay callers or model readiness. Concurrent unrelated ordinary imports are
explicitly unsupported; complete helper-copy/concurrency, serialization and
caller/source-snapshot acceptance remain required before formal adoption.

External TCP-observer preparation also has independent r2 PREP-only GO and10
stdlib test groups. Initial9-test review exposed an unvalidated event label
that could mask the original exception; r2 validates a nonempty exact str and
precomputes labels before installing its generic synchronous wrapper. This
limited mechanism is not installed in an engine process and does not establish
physical ownership, concurrent causality, collector persistence, flush/shutdown,
real hook/attempt/rank mappings or TCP model readiness. Original results and r2
results are retained separately under tcp-observer-draft in the pool run root.

## Historical stage notes (superseded where a later acceptance is recorded)

### Stage7 preparation and immutable references

Independent read-only review confirms that the current quickcheck imports the
historical low-concurrency matrix, which dynamically imports matrix_controller.
It also calls that base module for GPU/port cleanup, cgroup memory events,
analysis and summary, not just environment construction. The analysis Python
selection is part of the preserved contract. Do not archive these files before
removing this maintained-path dependency.

First extract the current fixed BrowseComp recipe and measurement/resource
helpers into scripts/benchmarks, keeping historical modules untouched. Prove
complete environment/argv and resource/summary outputs against immutable
frontend152a620 before a new full300+1200 comparison. The current benchmark
explicitly uses1/1 seconds and512+256GiB total Host arenas; the older named
BrowseComp shell recipe uses2/2 and1200-second warmup and is a historical replay,
not the current default. Preserve all actual ablation switches and Python/PATH
ordering. No source change is authorized during the active stage6 comparison.

Then classify historical matrix/autopilot/watchdog and pinned fused/checkpoint
families by complete dependency closure, retaining original path/hash/external
snapshot identity in manifests. Shared maintained CLI, multinode, Slurm and
process-supervisor paths remain stable initially. V1 multinode_smoke uses file
control and cannot count as V2 TCP acceptance. Any eventual relocation needs
explicit __file__/sibling-import/subprocess/self-component and monkeypatch
identity tests, not generic forwarding wrappers. This review is preparation,
not implemented or performance-accepted stage7 work.

Read-only preparation now freezes stage7-measurement-reference.json beside the
pool run directories. Immutable frontend152a620 matrix source SHA
74b915c5be96f3c6fe2dd12a6135eb1a35cd460b5f660b1c3c4128eca2b0c47d
reproduces all25 summary fields for both accepted Scheduler R2 and control-plane
runs exactly. The fixture records hashes of every input and expected outputs;
its cgroup before/after counters are explicitly fixed zero, not live resource
measurements. This is a future extraction reference, not stage7 implementation.

Independent preparation audit verifies the full201-file launcher inventory and
all hashes, with23 maintained entries, two V1 replay files, one historical2s/
w1200 recipe and175 entries still requiring review. Both25-field measurement
fixtures were independently reproduced from immutable ASTs. The filename graph
is deliberately lexical and misses bare-import edges; an empty reference list
is not proof of no dependencies or obsolescence. Semantic closure and formal
source/fault/GPU migration gates remain required after stage6 acceptance.

The external stage7-measurement-characterization.py now executes16 stdlib
unittest groups against immutable152a620 matrix/quickcheck sources. Independent
rerun passes: summary status/window/TP/tail/failure/OOM behavior, analyzer
short-circuit/input validation/exact Python-argv-cwd/failure propagation, GPU
probe command/errors, port union/bind/close, cgroup path/absence, consecutive
three-clean probes and unknown-helper no-signal failure. All hardware/network/
subprocess resource operations are mocked. The initial test expected pid rather
than the actual retained pid,process_name query; only that fixture was corrected.
No production source changed. These are old-behavior references, not migrated
code A/B, TERM/drain ordering or real resource-gate acceptance.

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

The TCP-stage unchanged-local regression subsequently passes300.855+1200.000
seconds at11203.87 Decode token/s (−0.54% versus ORIGINAL baseline),100% reuse,
6660 completions,0 failures/scoped memory events and empty final resource gates.
Whole-run sets match12054 P2D Host/21362 D2P Direct snapshots;10 measured D2P
Host durables restore/release,138 only-drain durables are documented separately.
This does not close actual TCP model-path coverage or any Scheduler/later stage.

Scheduler workset extraction is implemented as an intermediate stage4 gate:
complete lease+broker move together (39 broker methods), native Scheduler
reexports before eager annotations, and the whole original native module
reconstructs. Full CPU730 records, expanded native919 records, independent
focused18/18 and actualCUDA2 pass, with9 launch plans unchanged. The code audit
is GO; exact-plan audit and the frozen BrowseComp comparison remain required.
Direct/TP/ready/admission separation and all later stages remain open.

Scheduler workset subsequently received exact-plan GO and full performance
acceptance:300.601+1200.000 seconds,11207.29 Decode token/s (−0.51% versus
ORIGINAL baseline),100% reuse,6654 completions and clean final gates; see the
accepted result table and ownership-summary.json. Direct/TP/ready separation
is now applied as54 complete methods with four thin native ownership hooks.
Full CPU766 records and native969 records pass; the initial native fixture's
old barrier patch failure is preserved alongside a clean r2 rerun. Exact source
reconstruction and all nine launch plans pass. GPU DMA and the new full-window
comparison are pending, and stages5–8 plus genuine TCP model coverage remain open.

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
