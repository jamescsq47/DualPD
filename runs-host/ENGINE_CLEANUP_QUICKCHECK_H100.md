# Sequential engine cleanup: BrowseComp quick regression

This is a new staged validation record, not a replacement for the historical
300 s + 1200 s BrowseComp benchmark tables. The baseline below is measured from
the extracted projects; each subsequent cleanup still needs its own comparison.

## Dedicated allocation request

| Item | Requested value |
|---|---|
| Job | `28285059` (`dualpd-cleanup-7d`) |
| Partition | `pi_songhan` |
| Account | `mit_general` |
| Nodes | 1, exclusive |
| GPU | 8 × H100 |
| CPU | 224 logical CPUs (both sockets; NUMA checked before launch) |
| Host memory | 1800 GiB |
| Wall-time limit | 7 days, counted from actual allocation start |
| Submission | 2026-11-08 02:17:29 cluster time |
| Initial state | Pending; no node assigned at submission |
| Actual allocation | `node2700`, started 2026-11-08 10:04:17 EST (15:04:17 UTC) |
| Allocation expiry | 2026-11-15 10:04:17 EST (15:04:17 UTC) |
| Allocation state/log root | `/orcd/scratch/orcd/003/shangy/dualpd/runs/engine-cleanup-validation-20261108` |

The actual job/node/expiry must be rechecked through Slurm and the live
`allocation.json` heartbeat. Do not infer that a node is available merely from
this Markdown record. Do not submit another allocation for each experiment.

## Validation sequence

1. Audit the quick-check launcher and verify current ownership/fault gates.
2. Establish a real baseline with the current standalone repositories on this
   allocation. Freeze the BrowseComp protocol before changing engine modules.
3. Refactor one part, run relevant tests and independent audit, then rerun the
   same quick check. Investigate regressions before moving to the next part.
4. Record each comparison and the final GPU/port/process cleanup gate here.

The requested full node supersedes the initial idea of using four spare GPUs
with a smaller topology. The frozen first regression is Qwen3-8B TP1 4P:4D,
c256, Host D→P512/P→D256 GiB total, thresholds1s/1s, 300s business warmup plus
1200s measurement. This retains the original full-window gates. No performance
comparison is valid until the actual run passes them.

Read-only Slurm recheck at 2026-11-08 02:26:52 **EST (UTC−05:00)**: still
PENDING. The scheduler predicts node2700 at 10:01:05 EST / 15:01:05 UTC;
this is not an assigned node or guaranteed start. The pending request was
updated in place from 64 to 224 CPUs to cover both NUMA domains; no second
allocation was submitted.

## Results

| Stage | Engine revision | Decode token/s | Prefill token/s | Parent reuse | Functional/cleanup gate | Status |
|---|---|---:|---:|---:|---|---|
| Baseline | `4cde672cc4540fa41c87273a6b3862166535a22c` | 11265.01 | 75367.90 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `baseline-r3` |
| Host DMA ABI extraction | `90e99ace17f198ed9c33f8a63a6a59d697d1f446` | 11246.55 | 74816.50 | 99.992% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `host-dma-r1`; Decode−0.16% |
| Host registration/prewarm extraction | `0605c2241beaa079399ec31fa50c9b48e8a788e2` | 11220.31 | 75281.01 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `host-registration-r1`; Decode−0.40% |
| Physical Host buffers/Arena extraction | `a41aab25415bcb2abe943845bfd00ac06e9274bb` | 11254.37 | 75462.39 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `host-buffers-r1`; Decode−0.09% |
| D Host writer/Relay extraction | `f720985e4efe6f37cafcdd39a7cd9f7d5d2532e7` | 11214.24 | 74889.97 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `host-writer-r1`; Decode−0.45% |
| Local Router capacity/admission/HTTP extraction | `72518d3db7ece7e79fedd759d2bcf0f53d48521e` | 11206.66 | 75201.48 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `router-local-r1`; Decode−0.52% |
| TCP Router extraction: unchanged local-path regression | `3d1e52419255892bc5bde0c8c25c4ca74adae38e` | 11203.87 | 75023.20 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted local regression `router-tcp-r1`; Decode−0.54%; not TCP model coverage |
| Scheduler complete-workset lease/broker extraction | `2dbb67ccc5c4c2a61a00440efe58831b08d1c08c` | 11207.29 | 74549.10 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `scheduler-workset-r1`; Decode−0.51%; remaining Scheduler roles still open |
| Scheduler Direct/TP/ready/ownership component extraction | `a550ccaecad5d2b27ebb15a2788c1e1dc0daff8f` | 11238.17 | 74979.20 | 100.00% | 0 failures/OOM; GPU/ports/helpers all0 | Accepted `scheduler-components-r2`; Decode−0.24%; actual TCP/hybrid coverage remains open |

Physical buffer stage retains twelve function/class and sixteen libc initializer
ASTs exactly. Historical aliases are preserved; manager/lifecycle bodies and143
environment identifiers remain unchanged. Independent code audit GO covers the
physical ownership/fault invariants. The first tiny CUDA run preceded confirmation
of the final strengthened fixture hash; this ordering slip was identified and the
same exact two tests were rerun after GO (2 passed in2.65s). Native fault fixtures
are not full TP>1 model tests. The subsequently accepted full-window comparison
is recorded in the results table and final-stage notes below.

The CPU results and AST equivalence are
preconditions, not substitutes for the GPU throughput check requested here.

## Initial controller readiness (historical, superseded)

- Active attempt: `baseline-r2` under the allocation state/log root above.
  `baseline-r1` was superseded during pre-launch audit and never ran a workload.
- Independent audit: **GO**, auditor `/root/audit_quickcheck`; scope is this
  baseline only, after RUNNING and node preflight, not overall performance
  acceptance.
- Exact plan SHA256:
  `2a50eb194a9c958eb61ffe2d0393fdff0e95d5eddc1a92b2c0daa093b9048ef2`.
- Runner SHA256:
  `0dbd6aa8f0670077d9134127751c01b8cd0dcadbc04a079ba6ce40d89cf34592`.
- Expanded CPU gate: frontend 206 passed; engine 392 passed plus 2 subtests
  (600 JUnit records). The dedicated runner suite was also independently
  retested: **23/23 passed**, including a real nested-shell cleanup test.
- Source check: 4411 unchanged engine files, 73 relocated entries, 9 reviewed
  AST transformations, three unchanged router implementations; differences `[]`.
- Automatic waiting controller confirmed live on `ulogin001`, PID `273568`,
  command `engine_cleanup_quickcheck.py --wait-for-job .../baseline-r2/plan.json`;
  `controller.json` records the polled **PENDING** allocation. This is a real
  waiting process, not merely a lock or planned command. No GPU run has started.
- The controller starts exactly one baseline on this allocation when RUNNING;
  source/audit/NUMA checks are repeated before launch. It does not submit a
  second allocation, change the engine, or advance to another cleanup stage.

The keeper does not replace per-run health checks. Inspect the live process,
Slurm job and result artifacts when resuming; recorded PIDs are not permanent
proof of liveness. GPU/port/FUSE cleanup remains **unmeasured** until the first
actual node run completes.

## Actual baseline restart, 2026-11-09

The login-side waiting controller stopped at the interrupted agent session,
before allocation. The allocation itself later started successfully, but no
model run was launched automatically. The first detached SSH attempt for r2
then failed before launching any service: compute-host `Path.resolve()` mapped
`/home/shangy` to `/orcd/home/002/shangy`, differing from the prepared plan.
Neither attempt is a GPU throughput result.

The runner now preserves the explicitly selected checkout spelling and applies
it consistently to the historical recipe's roots; source/input/config hash
checks remain strict. No engine, model, sampling, topology, capacity or timing
parameter changed. Frontend 208 + engine392 + two subtests passed (602 records);
the 25 runner tests also passed independently. Source equivalence differences
remain `[]`.

Current run: **baseline-r3**, exact plan
`46c3cfa724ed243b97efdcc8b39933ff326c49a7a34a22b5ba0f70cb6b97a6f6`, runner
`1969660e26442f2acc3caed66b668af2db0147804af05aeb67a066f6c7cd78e7`.
Independent `/root/audit_quickcheck` returned GO for this plan and SSH transport.
It started at approximately 2026-11-09 01:15 EST / 06:15 UTC. Actual node
controller PID940624 was confirmed alive with PPID1 after SSH returned; its
runtime heartbeat and launcher are active. Pre-run gate: zero GPU processes,
zero configured ports, zero Apptainer/FUSE helpers, three consecutive probes.
NUMA preflight confirms CPUs0–223 and access to both GPU-local NUMA domains.

SSH was adopted into
`/system.slice/slurmstepd.scope/job_28285059/step_extern/user/task_0`.
The job and extern/user memory limits are 1932735283200 bytes (1800 GiB), and
initial OOM counters were zero in the task and all enclosing job scopes. Raw
scope evidence is `baseline-r3/cgroup-before.json`; do not equate unavailable
or unrelated OOM counters with node-global no-OOM proof.

Progress lives under `baseline-r3/runtime.json`, `launcher.log`,
`ssh-node-controller.log` and `logs/`. The completed `result.json` records
300.983 s warmup, 1200.001 s measurement, 6704 completions, zero failures,
Decode11265.01 token/s and Prefill75367.90 token/s. Completion-weighted
page-aligned parent reuse is100%; average per-physical-GPU Forward time is
70.10% Prefill and98.51% Decode. Decode also made1332929 tokens of progress in
the final120 s. Scoped memory-event deltas are all0.

At approximately01:57 EST /06:57 UTC, `runtime.json` became complete and
`post-cleanup.json` confirmed GPU processes0, configured occupied ports0 and
Apptainer/FUSE helpers0 across three consecutive probes. A subsequent independent
SSH probe again found all0, no live controller PID940624, and job-scope OOM
counters0. The allocation keeper remains available for the next stage.

## Host DMA extraction gate

Frontend208 tests and engine403 tests plus2 subtests passed (613 JUnit records).
Independent code audit GO covers the unchanged CUDA ABI/submission semantics,
six historical helper/type aliases and canonical private caches. The verifier
checks all10 relocated DMA nodes and the remaining Host worker AST unchanged.
The exact GPU plan still requires independent audit before execution.

Baseline whole-run logs (including warmup and controlled drain, not measured
window rates) contain11241 unique P→D Host snapshots at each queued, durable,
Prefill-release, H2D-complete and Host-release stage. D→P contains120 D2H
starts/completions, all120 with a mapped D source-release event;8 have an H2D
completion and Host release. The remaining112 durable snapshots have no H2D
completion in these logs. Their D2H completion timestamps are all after the
measurement boundary:0 during warmup,8 during measurement,112 during controlled
drain. All8 measured-window durable snapshots have H2D completion/release.
The112 drain snapshots must not be presented as measured-window leaks or as
restored; their final reclamation is established at process teardown, not by
individual Host-release log events. Final resource cleanup is independently
verified above.

Further set-based checking finds21436 unique D→P Direct send completions and
the exact same21436 P receive completions, all with mapped D source-release
events. The120 D→P Host durable snapshots exactly match the120 explicit
`d_release_after_p_host` snapshot IDs. Raw log-derived accounting is retained
in `baseline-r3/ownership-summary.json`. These event-set checks do not by
themselves establish release latency or every instantaneous ownership state.

The DMA comparison `host-dma-r1` is now running via detached SSH. Exact-plan
audit GO covers plan SHA256
`27d98c712ee81576fe32222a1b7b303f4073a97137b01a5156a1a1d59433579b`.
Actual controller951316 has PPID1 and belongs to job28285059's extern/task0
cgroup; enclosing job memory limit1800 GiB and all job OOM counters0 at startup.
See `host-dma-r1/cgroup-before.json`. No throughput acceptance is implied by
startup or the code audit.

DMA result completed at approximately2026-11-09 02:49 EST /07:49 UTC:
300.681 s warmup,1200.001 s measurement,6682 completions,0 failures,
Decode11246.55 token/s (ratio0.998361 to baseline), Prefill74816.50 token/s.
P/D per-GPU Forward fractions69.51%/98.44%; tail120 s Decode1341993 tokens.
Scoped OOM deltas0; post-cleanup GPU/ports/helpers all empty and an independent
SSH probe again confirmed GPU0, both controller/launcher exited and job OOM0.
The 99.992% parent-prefix reuse has one harness terminal-parser repair: the
extra17280 Prefill tokens exactly equal the repair's finalized-parent cost;
this is not unexplained parent KV loss. Raw details are in
`offload_analysis_summary.json` under `completion_weighted_reuse` and
`terminal_repairs`. Passing one comparison excludes a gross regression; it is
not a statistical-equivalence claim across all model/TP configurations.

The DMA run's `ownership-summary.json` retains whole-run event-set checks:
11508 P→D Host snapshots match at all five queued/durable/P-release/H2D/Host-release
stages;21432 D→P Direct send/receive snapshots match and all have D source
release;141 D→P Host durable snapshots exactly match explicit D release IDs.
All8 measured-window Host durables completed restoration/release;133 remaining
durables occurred during controlled drain and have no individual restoration
completion in logs. Final whole-process cleanup is verified separately.

Registration comparison started at approximately2026-11-09 03:03 EST /08:03 UTC,
reusing job28285059/node2700. Exact plan SHA256
`cb48f0aa926761d780f3b6b04f4e4b44aa7b4de652d8ff282a35b9a2006c6cf3`
has independent audit GO. Before launch, the actual SSH cgroup was the allocated
job's extern/task0, with1800 GiB limit and job OOM counters0. Pre-cleanup confirms
GPU/ports/helpers0. Controller962919 owns launcher962943 and writes progress to
`host-registration-r1/runtime.json`. The source/configuration is frozen; this is
not performance acceptance of registration or completion of later cleanup work.

Registration result completed at approximately2026-11-09 03:43 EST /08:43 UTC:
300.607 s warmup,1200.004 s measurement,6667 completions,0 failures,
Decode11220.31 token/s (ratio0.996032 to the original baseline), Prefill75281.01.
Parent-prefix reuse100%; P/D per-GPU Forward70.19%/98.38%; tail120 s Decode
1347257 tokens. Scoped memory-event deltas are all0. Post-cleanup GPU/ports/helpers
are all empty; a fresh SSH probe also confirms no live controller/launcher,
GPU0 and job-scope OOM counters0. This passes the stage's gross-regression gate,
not a statistical equivalence claim for all models or completion of later stages.

Registration ownership accounting is retained in `ownership-summary.json`:
11595 P→D Host IDs match across queue/durable/P-release/H2D/Host-release;
21607 D→P Direct send/receive IDs match and all map to D source release;
141 D→P Host durables match explicit D releases. All7 measured-window Host
durables restore/release;134 additional durables occur only during controlled
drain without individual restore completion. Final process cleanup is verified
separately; event sets do not prove instantaneous ownership or release latency.

Physical buffer comparison has now launched on the same job/node. Exact plan
SHA256 `8020523c3454b70a3875d5cc46e4d1b5aa147a2aec2569ef37fa8db247621edc`
received independent GO. Detached SSH controller974553 (PPID1), launcher974577,
is in the allocated extern/task0 cgroup; job memory limit1800 GiB and scoped
OOM counters0. Its live runtime reports running and the first model GPU process
is present. This is startup evidence only, not performance acceptance.
Source/configuration are frozen until results and final cleanup are complete.

Follow-up live checks confirm all8 P/D model services ready, all8 registration
prewarm contexts complete, and search/router healthy. The680-source BrowseComp
schedule has loaded; business warmup is active. Controller974553 and its owned
launcher974577 remain live, with a fresh Decode counter491271 at1794215623.19.
Job memory events remain0. No throughput acceptance is inferred from this
startup/warmup sample; the full measured window and final gates are outstanding.

Physical buffer final result:301.006 s warmup and1200.000 s measurement,
6688 completions,0 failures, Decode11254.37 (0.999055 of ORIGINAL baseline),
Prefill75462.39, parent reuse100%. P/D per-GPU Forward70.22%/98.43%;
tail120 Decode1318396 tokens. Job-scoped memory-event deltas0, three post-cleanup
probes passed; independent SSH confirms both controller/launcher gone, GPU0 and
job OOM0. Whole-run event sets match11338 P→D Host snapshots at all five stages
and21628 D→P Direct send/receive/source-release IDs. All137 D→P Host durables
match D source release;9 measured-window snapshots restore/release,128 additional
durables occur only during controlled drain without individual restoration.
See `ownership-summary.json`; final process cleanup is separate evidence.
This passes the gross-regression gate, not statistical equivalence or final
acceptance of all model/TP configurations and later cleanup stages.

D writer/Relay preparation: four exact AST nodes moved, canonical sentinel and
legacy aliases retained; main Host module4155 lines. Independent code GO and
final revised cold-import fixture GO precede the final native557/CUDA2 passes.
CPU665 records passed after an unchanged overlay subprocess SciPy-import crash
was independently rechecked2/2 and the full gate repeated. The original native
fixture failure was an overly strong no-CUDA-init claim: accepted-a41 cold Host
import already initializes CUDA via native sgl_kernel. Two fresh subprocesses
now verify parity, capturing writer state before staging plus cycle/alias guards.
No production fix or old assertion weakening was used. Detailed history lives
in `tests/reports/host-writer-import-evidence.json`. Relay's pre-existing
finally-on-fence-error risk is documented unchanged; the frozen benchmark leaves
that optional path disabled and cannot certify it. Full performance acceptance
and all remaining cleanup stages are still outstanding.

D writer comparison launched on job28285059/node2700 with detached SSH
controller987384 and owned launcher987408. Exact plan SHA256
`a008b50d436963028c4273a5af09e7213d87ec90855c064c0bd5a7b0b69b14e3`
received independent GO; effective config and inputs match BOTH original
baseline-r3 and host-buffers-r1 (only RUN_DIR differs). Live allocation/cgroup
checks show1800 GiB job limit and scoped memory events0. Three preflight resource
probes passed and runtime reports running. Engine/configuration are frozen until
the full300+1200 and final cleanup/result gates complete. This is startup, not
performance acceptance of writer or completion of remaining stages.

Writer live follow-up: all8 model services and registration-prewarm contexts
became ready; search/router started and680 BrowseComp sources loaded at about
10:12 UTC. Runtime/actual PID probes confirm controller987384 and launcher987408
live, with fresh Decode709314 at1794219181.97 during business warmup. This does
not replace the final full-window/ownership/cleanup acceptance.

Writer final result at about10:40 UTC:301.030 s warmup and1200.020 s measurement,
6671 completions,0 failures, Decode11214.24 (0.995493 of ORIGINAL baseline),
Prefill74889.97, parent reuse100%. P/D per-GPU Forward69.68%/98.42%; tail120
Decode1334845 tokens. Scoped memory-event deltas0 and post-cleanup GPU/ports/
helpers all empty. Fresh SSH confirms controller/launcher gone, GPU0 and job
OOM0. Whole-run sets match12008 P→D Host snapshots at all five stages and21496
D→P Direct send/receive IDs, all mapped to D source release. All156 D→P Host
durables match D release;15 measured-window durables restore/release,141 additional
durables occur only in controlled drain with no individual restore completion.
See `ownership-summary.json`; process reclamation is verified separately.
The original-baseline gross-regression gate passes; this is not statistical
equivalence, optional Relay fault correctness or full TP/model acceptance.

Local Router preparation: the concrete entry is430 lines, retaining its exact
constructor and select_pair. Three method-only components hold33 capacity,
38 admission and27 HTTP methods on that same instance. Complete100-method/
14-top-node AST reconstruction passes, including runtime-global bindings;
12 destructive mutations are rejected. Original114+2 subtests and26 new focused
tests pass; independent audit also reruns26/26 and returns scoped code GO.
Full CPU gate passes208 frontend+481 engine+2 subtests (691 JUnit records).
Node2700 native/Router regression passes716+2 subtests;2 actual H100 registered/
bounce roundtrips also pass. Source proof retains143 environment identifiers,
4407 native byte-identical files,2 unchanged TCP/pinned Routers and11 direct+
1 reconstructed serving AST comparison. Nine launch plans preserve commands/
environments except documented paths. Actual local-Router throughput is pending;
TCP/Scheduler/control-plane/configuration/launcher/final stages remain open.

Local Router comparison launched at about11:09 UTC on job28285059/node2700,
engine72518d3db7ece7e79fedd759d2bcf0f53d48521e, frontendefe0103adaf4728f7899d97053c58781e6561493.
Plan SHA2560003916349599874260b4a4041b3b868db3aac39e93dbc4dd4f6466b70476927
received independent exact-plan GO. Source/config/input/reference checks match
both ORIGINAL baseline-r3 and host-writer-r1 (only run directory/config source
identity differs as documented). Detached SSH controller999909 has PPID1 in the
owned extern cgroup; launcher999933 is live and runtime reports running. Job RAM
limit is1800 GiB; scoped events0 and all three preflight resource probes clean.
Models are starting serially. Engine and serving sources remain frozen until
the complete300+1200 result and final resource gates; this is not performance
acceptance or completion of remaining stages.

Local Router final result at about11:49 UTC:300.861 s warmup and1200.001 s
measurement,6668 completions,0 failures, Decode11206.66 (0.994820 of ORIGINAL
baseline;−0.52%), Prefill75201.48, parent reuse100%. P/D per-GPU Forward70.00%/
98.36%; tail120 Decode1328240 tokens. Scoped memory-event deltas0 and three
post-cleanup probes pass with GPU/ports/helpers empty; independent SSH confirms
controller999909 and launcher999933 gone, GPU0 and job OOM0. Whole-run event sets
match12418 P→D Host snapshots across all five stages and21503 D→P Direct send/
receive IDs, all mapped to D source release. All141 D→P Host durables match D
release;10 measured-window durables restore/release,131 additional durables
occur only in controlled drain with no individual restoration. See the saved
ownership-summary.json; final process reclamation is separate evidence. The
read-only verification script reproduces the preceding writer summary exactly.
This passes the gross-regression gate, not statistical equivalence or full
TP/model-path acceptance. Next runtime stage is the separate TCP Router;
Scheduler/control-plane/configuration/launcher/final stages remain required.

TCP Router preparation: concrete root281 lines,8 retained methods plus unchanged
callbacks/main/entry; capacity12 and HTTP9 methods in two method-only components.
Whole original29-method/module AST reconstructs; exact imports/global bindings
and original15 test sources retained. Focused37 and independent37/37 pass with
code GO. Full CPU713 records, node2700 native740 records and2 actual DMA roundtrips
pass, zero failures/errors/skips;9 launch plans match. The initial new slots-
pickle fixture error is recorded in router-tcp-test-evidence.json and corrected
without runtime changes. The pending same-config BrowseComp case is explicitly
an unchanged-local-path nonregression check, not actual TCP model-path coverage.
All later Scheduler/control-plane/configuration/launcher/final work remains open.

TCP-stage unchanged-local-path comparison launched at about12:05 UTC on
job28285059/node2700, engine3d1e52419255892bc5bde0c8c25c4ca74adae38e and
frontendd1ca50aa4db741ea4d688537a80783ddb50f7319. Exact plan SHA256
8fcd5350e02e3acda80aac021ee1052fafc051fe06c3d6e6a4c15300ac24a4d9 has independent
GO. Detached controller1012110 and launcher1012134 are live; controller PPID1 in
the owned extern cgroup,1800 GiB limit and scoped memory events0. Three preflight
resource probes are clean. Config/input/command match ORIGINAL baseline-r3 and
router-local-r1 except run directory and documented source identity. Sources
are frozen for complete300+1200 and final cleanup/result gates. This is startup,
not local throughput acceptance, TCP model-path coverage or full-goal completion.

TCP-stage local live follow-up: eight model services and all8 registration-
prewarm contexts passed; search/router ready and680 BrowseComp sources loaded
at about12:17 UTC. At12:18 UTC the actual owned controller1012110/launcher1012134
remain live and Decode grows to607703 tokens (sample1794226689.3743975). Scoped
memory events remain0, source checkout clean/frozen and P2D Host binding appears
in the unchanged local Router logs. This is business warmup, not full-window
acceptance. Read-only Scheduler planning additionally identified the precise
broker globals, eager-annotation constraint and a readiness-sync private test
hook; that expanded future fault inventory is in docs/NEXT_ENGINE_BOUNDARIES.md.

TCP-stage local regression final at about12:45 UTC:300.855 s warmup and1200.000 s
measurement,6660 completions,0 failures, Decode11203.87 (0.994572 of ORIGINAL
baseline;−0.54%), Prefill75023.20,100% parent reuse. P/D per-GPU Forward69.80%/
98.33%. Scoped memory-event deltas0; final GPU/ports/helpers empty and independent
SSH confirms controller1012110/launcher1012134 gone, GPU0 and job OOM0. Whole-run
sets match12054 P→D Host snapshots across five stages and21362 D→P Direct send/
receive/source releases. All148 D→P Host durables match D source release;
10 measured-window durables restore/release,138 only-drain durables have no
individual restoration and are accounted separately from final process cleanup.
See ownership-summary.json. This accepts unchanged-local-path nonregression,
not statistical equivalence, actual TCP model coverage or remaining goal stages.

Tail metric clarification: original validator reports1294413 tokens between the
first/last samples inside the final120-second interval, spanning116.169 sampled
seconds. The harness boundary rolling-cache counter is1319512. These are distinct
sampling definitions, not contradictory full-window throughput. No algorithm or
accepted result is changed; see tail-sampling-audit.json. Retain the canonical
sampled-progress metric for comparable liveness gates, not an exact120-second
token-total claim.

Scheduler-components-r1 launched at about14:03 UTC on the same dedicated
job28285059/node2700. Enginea550ccaecad5d2b27ebb15a2788c1e1dc0daff8f and
frontend e221909499d93c2ce513ce6c9752de44e6cbbd4d are recorded in the exact
plan/code audit. PlanSHA23c7c17c179a1ba2c95305e8c531d0dbe2becc00e5457e65eb611bce4d0ae474
has independent GO after CPU766, real-kernel native969, actualCUDA2,
independent36 and unchanged nine launch plans. First native fixture failure is
preserved separately from the passing rerun; no runtime barrier was changed.
Controller1039054 and launcher1039078 are live in the owned extern cgroup;
three preflight GPU/port/helper probes are empty and scoped memory events0.
Serving sources are frozen for full300+1200 versus ORIGINALbaseline-r3.
This is model startup, not performance/ownership/final-cleanup acceptance or
actual TCP model-path coverage. Later cleanup stages remain required.

Live follow-up at14:16 UTC: all eight P/D model services and all eight Host
registration contexts are ready (maximum119.608 s prewarm, before measurement).
Search/Router are healthy,680 fixed sources are loaded, and business warmup is
active. Canonical Decode counters rise84103→132880→173288→207417 across samples
1794233754.847 through1794233761.398. Controller/launcher remain in the owned
extern cgroup, scoped memory events0, and serving source is still frozen. These
are liveness observations, not a completed steady-state or throughput result.

Scheduler-components-r1 is NOT accepted. Its300.640+1200.000-second window
completed with6675 measured completions and0 failures, but final requests.jsonl
write raised OSError122 Disk quota exceeded. The scratch user directory is
about1TiB; global filesystem free space is not the user's remaining quota.
The resulting568740967-byte logical request file is incomplete (JSON parsing
fails after612 valid rows), so it cannot establish parent-reuse/full result
acceptance. Launcher exits1; runtime recordsfailed. Final resource probes pass:
GPU/ports/helpers all0, both owned controller/launcher gone and scoped OOM0.
One transient NVML probe timeout occurred during normal registered-memory drain
and resolved; it did not trigger a restart or allocation cancellation.

Preserve the failed attempt and all historical data. Prepare a fresh
scheduler-components-r2 on the same job/node and unchanged engine/frontend,
moving only result/run-directory-derived paths to the existing user-owned
/orcd/pool/004/shangy/dualpd/runs/engine-cleanup-validation-20261108 hierarchy
(421GiB available at diagnosis). Models/data/input order/commands/protocol and
ORIGINALbaseline-r3 reference remain fixed. The retry needs independent exact-
plan GO and a complete new300+1200 window; missing records are not reconstructed
or fabricated from aggregate counts. Serving code remains frozen meanwhile.

Retry scheduler-components-r2 received independent exact-plan GO
(SHAa01593496a71098063d2c19b2e46e40f05e37581da27f5a1d5495b058798152f)
and launched at about14:49 UTC on the same allocation/node. Controller1049992
and launcher1050016 are live; all three preflight resource probes are empty.
Enginea550ccaecad5d2b27ebb15a2788c1e1dc0daff8f, frontend serving manifest,
runner, input hashes and ORIGINALreference are unchanged; only RUN_DIR moves.
The new attempt is currently startup, not accepted performance or completion.

Final scheduler-components-r2 acceptance: independent audit GO after actual
300.587666-second warmup and1200.011824-second measurement. Result SHA256
6c1f86161797ed4c700c0f928112cd76c232be616e3e8dc4c56e19d588cb0208.
All8281 request records parse, have unique indices, and6683 finish in the
measurement window, matching its counters:2370 completed+4313 truncated with
no error/failed/aborted records. Protocol success is not task accuracy or proof
that every task naturally completed. Decode11238.167499 token/s is0.997617155
of ORIGINALbaseline-r3; this is a gross-regression check, not statistical
equivalence. Scoped memory-event deltas are0. Final owned SSH verification finds
controller1049992/launcher1050016 gone, GPU/helper processes and configured
listener ports empty. Allocation28285059 and its keeper remain retained.

Whole-run ownership event sets: P2D Host11089 queued=durable=P release=H2D
complete=Host release; D2P Direct21487 sent=received=mapped D source releases.
D2P Host129 durable=explicit D release;9 measured-window snapshots restore and
release, while120 become durable only during drain and lack individual restore
events. Final process cleanup is separate from individual snapshot restoration.
These event sets do not establish instantaneous release latency or reconstruct
P2D Direct releases. R1 remains a failed quota-limited attempt, not accepted.
This unlocks subsequent control-plane cleanup, not a new GPU launch or full-goal
completion; actual TCP/TP2/hybrid and configuration/launcher work remain open.
