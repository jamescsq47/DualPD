# Remaining Scheduler component design (read-only preflight)

Reference: engine2dbb67ccc5c4c2a61a00440efe58831b08d1c08c.
The current workset model case is frozen; this document is NOT code GO and
nothing below authorizes runtime edits before its full performance acceptance.
Independent reviewer: /root/audit_quickcheck. Main rederived all54 method names
from the current AST, not from an approximate name-prefix grouping.

The final target is50 complete method bodies in three method-only components,
sharing the same native Scheduler instance/queues/locks, with four native
ownership integration hooks. A46-method move retaining four huge methods for
compatibility would be only an intermediate gate, NOT final cleanup. Do not
substitute that easier-to-test state for the actual target.

## Direct session and physical progress

20 methods; 2495 body lines in this reference.

- `_agentic_start_early_direct_receive`
- `_agentic_drop_early_direct_receive`
- `_agentic_handle_unstarted_direct_abort`
- `_agentic_collect_direct_abort_events`
- `_agentic_return_early_direct_to_slow`
- `_agentic_early_direct_progress_worker`
- `_agentic_poll_early_direct_receives`
- `_agentic_collect_direct_arrivals`
- `_agentic_early_direct_slots_used`
- `_agentic_admit_queued_direct_receives`
- `_agentic_poll_early_direct_receives_once`
- `_agentic_bind_early_direct_receive`
- `_agentic_try_finalize_early_direct_bind`
- `_agentic_finalize_early_direct_bind`
- `_agentic_admit_early_direct_bind`
- `_agentic_rollback_prepared_direct_bind`
- `_agentic_start_direct_load`
- `_agentic_return_request_direct_to_decode`
- `_agentic_poll_direct_load`
- `_agentic_clear_direct_receiver`

## TP group coordination

15 methods; 1647 body lines in this reference.

- `_agentic_tp_abort_mailbox_write`
- `_agentic_mark_tp_direct_failed`
- `_agentic_progress_tp_direct_grants`
- `_agentic_abort_tp_direct_grant`
- `_agentic_commit_tp_direct_groups`
- `_agentic_fail_tp_direct_bind`
- `_agentic_v2_prepare_activation_control`
- `_agentic_v2_consume_activation_control`
- `_agentic_tp_reduce_direct_status`
- `_agentic_tp_reduce_host_status`
- `_agentic_tp_reduce_workset_retire_status`
- `_agentic_tp_host_next_action`
- `_agentic_tp_prepare_admission_control`
- `_agentic_tp_consume_admission_control`
- `_agentic_tp_start_direct_shard`

## Ready workset and native admission

15 methods; 868 body lines in this reference.

- `_agentic_refill_host_events`
- `_agentic_should_defer`
- `_agentic_queue_class`
- `_agentic_io_active`
- `_agentic_io_kind`
- `_agentic_progress_granted_slow`
- `_agentic_bind_completed_waiters`
- `_drain_agentic_kv_waiting_queue_tp1`
- `_drain_agentic_kv_waiting_queue`
- `_prioritize_agentic_prefill_ready`
- `_agentic_mark_p_host`
- `_agentic_abandon_load`
- `_agentic_consume_if_already_on_gpu`
- `_agentic_publish_p_accepted`
- `_agentic_publish_p_scheduled`

## Native ownership hooks

4 methods; 274 body lines in this reference.

- `_agentic_snapshot_store`
- `_agentic_service_p_workset_leases`
- `_agentic_abort_cleanup`
- `_record_agentic_forward_read_fence`

## Explicit compatibility, not a root import cycle

Seven Scheduler attribute expressions span four methods: two abort-mailbox
calls, one Direct slot-count call, two Direct group-abort calls and the two
control-key constant reads. Moving them must preserve native CLASS dispatch,
not change to self/type(self), which changes monkeypatch behavior and breaks
Scheduler.method(SimpleNamespace(...)) fixtures.

Proposed non-caching resolver reads only the already initialized canonical
sys.modules['sglang.srt.managers.scheduler'].Scheduler. It performs no import,
starts no task and fails closed without a native class. The exact seven
expressions become _native_scheduler_class().attribute with all arguments
unchanged. Method/constant patches and replacement of the entire root.Scheduler
must remain visible on every lookup. No generic globals proxy or cached class.
Canonical production launch paths are imported run_scheduler_process; separately
specify/test any alternate file/spec import support, never silently load a
second native module. A root-owned non-caching callback is an alternative only
with an audited repeated-root ownership design, not a captured static class.

The early-Direct record belongs with Direct (or one coherent type dependency
layer), preserving eager annotation objects, all defaults, old root aliases and
historical pickle identity. Helpers may import Req/WorksetLease/lifecycle types
from their defining owners, NOT import the record or Scheduler back from root.
Use the historical Scheduler logger. No separate component instances or new
state owners; components dispatch to peers through the same self.

The native module keeps initialization, watchers/worker start-stop, allocator/
Radix/Req ownership, event loops and mixed native call sites. Do not miss
_merge_disagg_prefill_ready or external prefill.py allocator-service/merge calls.
V2 activation is a real maintained path, not a removable legacy switch.

## Required proof and tests

- Reconstruct the entire original native module, complete54 methods and mixed
  hooks, original MRO/decorators/defaults/annotations/initialization. Inverse
  normalization may restore ONLY the exact seven reviewed Scheduler expressions.
- Scan all test/module global patches again. In particular preserve RankSet
  reversal and all assertions in test_agentic_tp_host_commit_order.py, retargeting
  only module.set to consume_control's true defining module. Shared class-object
  metadata.from_req patches retain root identity. No broad test rewrite.
- Fresh cold imports must not load Scheduler through a helper. Resolver must see
  dynamic method/constant/root-class replacement, retain NS invocation semantics,
  reject absent/wrong owner, no fallback import/cache. Mutation checks must reject
  self/type(self)/wrong root/cached class/lost constant/extra root cycle.
- Preserve completion edges, readiness-before-Req recovery, resident Mamba
  admission, complete suffix reserve and Radix protection. Granted Direct leases
  count even before receiver creation; completed physical DMA releases lane use.
- Preserve unstarted abort, partial-DMA fences, stale-owner/attempt isolation,
  all-rank prepare/commit/rollback/retire, hash-order independence, delayed HTTP
  arrivals and V2 no-V1-broker/watchers/single-memory-authority rules.
- Run fullCPU and the expanded917+2 native baseline with new faults; independent
  code GO, real CUDA as required, clean commit and exact-plan GO; then frozen
 300+1200 BrowseComp against ORIGINAL baseline, ownership and final clean gates.

No final Scheduler, actual TCP model-path, control-plane/configuration/launcher
or overall goal acceptance is claimed by this preflight document.

## Configuration inventory scope must follow moved code

Current extension-only literal inventory is143 identifiers; native Scheduler
has67, with51 not present in extension, yielding194 unique identifiers across
both. Moving Scheduler methods into the extension will change the old
directory-only inventory even without changing one switch. Before that move,
extend the reference AND current verifier scopes to extension+entire native
Scheduler, retain the sole proven no-op selector removal, and report194 as the
broader preserved scope rather than hardcoding143 or waiving new identifiers.
Native Scheduler still owns many constructor-time settings; do not move those
parsers or change dynamic reads just to satisfy an inventory check.
