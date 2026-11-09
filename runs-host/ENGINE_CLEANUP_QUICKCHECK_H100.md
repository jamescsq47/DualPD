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
| Physical Host buffers/Arena extraction | `a41aab25415bcb2abe943845bfd00ac06e9274bb` | — | — | — | CPU651 records; H100 fault543 and actual roundtrip2 passed | Throughput pending `host-buffers-r1` |

Physical buffer stage retains twelve function/class and sixteen libc initializer
ASTs exactly. Historical aliases are preserved; manager/lifecycle bodies and143
environment identifiers remain unchanged. Independent code audit GO covers the
physical ownership/fault invariants. The first tiny CUDA run preceded confirmation
of the final strengthened fixture hash; this ordering slip was identified and the
same exact two tests were rerun after GO (2 passed in2.65s). Native fault fixtures
are not full TP>1 model tests. Full performance acceptance remains pending.

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
