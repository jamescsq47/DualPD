# Sequential engine cleanup: BrowseComp quick regression

This is a new staged validation record, not a replacement for the historical
300 s + 1200 s BrowseComp benchmark tables. No new GPU throughput result has
been measured yet.

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
| Baseline | `4cde672cc4540fa41c87273a6b3862166535a22c` | — | — | — | Pre-run clean; loading model | Running `baseline-r3` on node2700 |

The existing CPU results (577 passing records) and AST equivalence are
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
`ssh-node-controller.log` and `logs/`. The result and post-cleanup gates are
still pending. Do not advance to a new engine stage before actual acceptance.
