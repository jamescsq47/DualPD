# Scheduler workset boundary

This is an intermediate part of stage4, not Scheduler cleanup completion.
The lease and broker now live together in the canonical
sglang.srt.disaggregation.agentic_workset module. The complete original two
classes retain their39 broker methods and four lease properties, all allocation,
fence and retirement branches. The native Scheduler imports the same objects at
the old definition position, before its early-Direct record evaluates annotations.
Two explicit __module__ assignments preserve old serialization paths without a
Scheduler back-import. Native Scheduler imports otherwise remain unchanged.

## Lifecycle scope before editing

The component owns complete P HBM worksets during Direct/Host restoration,
binding and suffix handoff. An intent owns no physical pages. Broker.service,
invoked at a scheduler-safe boundary, alone allocates/frees workset pages,
respecting native chunk reserve, page alignment,
hybrid state rollback and immutable TP allocation epochs. A transport attempt
must quiesce before a release_pending lease can retire; stale lease/owner/attempt
callbacks cannot reclaim a successor. Parent binding and suffix transfer retain
the existing Radix/native Req ownership boundary. TP cancellation, partial I/O,
bind failure and shutdown keep their original group retirement requirements.

There is no new timeout, queue, allocator, state table, priority or policy.
Capacity rejection keeps the intent and rolls back partial allocations exactly
as before. Cancellation and exit use the same owner-specific fenced cleanup.
No Direct, Slow, Forward or control-plane path acquires a new wait dependency.
The native Scheduler remains responsible for physical service and group hooks.

## Eight invariants

1. Unique complete ownership: same lease identity/state transitions and native allocator.
2. P2D Direct release: untouched native integration/transport.
3. P2D Host durable release: untouched native integration/transport.
4. D2P Host durable release: untouched writer/ledger/transport.
5. Independent progress: same lock and queue on one shared broker; no new worker.
6. TP atomicity: same immutable epochs, tombstones and all-rank retirement/handoff.
7. Parent correctness: same page conversion/readiness synchronization, parent+suffix
   reservation, Mamba slots, Radix donation and explicit recompute accounting.
8. Gate: reconstruction, mutation and native lifecycle/fault tests, independent
   code/plan GO, then full frozen BrowseComp and ownership/cleanup acceptance.

The verifier reconstructs the entire original native module, including imports,
mixed Scheduler hooks, decorators/defaults and initialization—not just moved
classes. The existing H2D readiness hook retargets only broker-specific lookup;
all assertions and unrelated Scheduler hooks remain intact. A same-path native
source-map entry has explicit original/current hashes and reconstruction.

Native alias/pickle tests need real installed native imports on the owned H100
node, even though their tensors are CPU. They are separate from portable CPU
structural checks, with no dependency stubs or silent skips.

The pre-model full CPU gate passes208 frontend+520 engine+2 subtests (730
records). Expanded owned-node native tests pass917+2 subtests (919 records),
including the newly inventoried Scheduler consumers. The independent auditor
passes17 structure+1 native alias fixture separately and grants code GO for the
exact workset/structure/native/verifier hashes. No model experiment has yet
accepted this stage; two real H100 registered/bounce CUDA roundtrips also pass
after code GO, using unchanged fixture aa0876bc9a3b865aeb1ecd0853e9f1548ffa67e21ac109df87e572e8f1eb9387.
Exact-plan GO and actual performance remain the next gates.
Initial new-fixture failures and corrections remain in
tests/reports/workset-test-evidence.json, with the original failed XML retained.

Direct progress, TP coordination and ready/admission remain subsequent
responsibilities, followed by the other full-goal stages.

The frozen scheduler-workset-r1 comparison is now independently accepted:
300.600558 s warmup+1200.000298 s measurement, Decode11207.29 token/s (−0.51%
against ORIGINAL baseline), Prefill74549.10,6654 completions,100% parent reuse,
0 failures/scoped memory events and final GPU/ports/helpers empty. Event sets
match11413 P2D Host snapshots across five phases and21370 Direct snapshots
through mapped D release. All143 D2P Host durables have D source release;
11 measured-window durables restore/release,132 only-drain durables are
documented separately, not individually restored or counted as measured loss.
Independent result/source/ownership/fresh-SSH cleanup audit is GO. This unlocks
the next runtime edit, not the next GPU run or full Scheduler completion.
