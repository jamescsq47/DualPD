# Remaining final validation: evidence required, not completion

The sequential cleanup goal remains open until all stages in
SEQUENTIAL_ENGINE_CLEANUP.md have implementation, relevant source/fault gates,
independent audit and appropriate real BrowseComp comparison evidence. This note
prepares the representative model-path checks; it authorizes no launch by itself.

## Current accepted versus missing evidence

Accepted local TP1 BrowseComp runs cover fixed Qwen3-8B4P:4D/c256 and the complete
300+1200 protocol through Host, local/TCP Router extraction and Scheduler components.
The control-plane dependency run is accepted at11223.49 Decode token/s,−0.37%
versus ORIGINALbaseline,100% parent reuse and clean final gates. Stage6
configuration code has CPU928/native1018/CUDA2 and independent CODE GO; its new
model comparison is also accepted at11212.44 Decode token/s,−0.47% versus
ORIGINALbaseline,100% parent reuse and clean final gates. Calling a local
Router comparison a TCP model test would be incorrect. Nine model/TP dry plans
and real-kernel TP/Mamba fault fixtures are source/functionality evidence, not
actual distributed model forwards or accuracy verification.

## Actual TCP/TP model probe

Use the maintained scripts/cluster/multinode.py and its real group_control V2
relay, GlobalEndpointMiniLoadBalancer and HTTP application_final callback. Do
not reuse historical multinode_smoke.py unchanged: that script reads
cfg.control_root, AgenticEarlyClaimStore and Host ledger files, which V2 does not
use. No fake route markers, shared control filesystem or local-only substitute
can demonstrate TCP model coverage.

The allocated node can host distinct logical P/D groups at one routable IPv4
address, with unique node_id/engine_id, disjoint physical GPUs and distinct
worker/bootstrap/reverse/bootstrap/router/control listener ports. A TP2 P group
and TP2 D group require four GPUs, leaving room for a later retrieval workload.
Same-host logical TCP validation establishes the protocol/model path, not
multi-host network performance or failure-domain tolerance. Choose actual GPU
NUMA placement from the owned node, not the example's machine-specific IPs.

Prepare a fresh isolated config after source cleanup, record model/config/input
hashes and exact commands, keep the group token out of chat/documentation, and
obtain independent lifecycle/fault/code and exact-plan GO. Confirm allocation
ownership, CPU/NUMA visibility and empty GPU/ports/helpers before starting.

The probe must cover:

1. Real TP2 model forwards and two agentic generations using exact token IDs.
   Require actual returned output IDs or logprob token IDs; refuse text
   retokenization as proof of the exact KV parent. Construct the second prompt
   from the complete first input/output IDs plus a deterministic tool suffix.
   Record finish_reason and prove TOOL through AgenticRequestMetadata.classify_output
   on the returned IDs. Worker handoff retains UNKNOWN provisionally too, whereas
   length/abort are final; merely observing a retained parent is not TOOL proof.
2. Both tool-fast Direct and tool-slow Host returns through actual V2 metadata,
   worker detection and group commands. Existing V1 tool ACK markers must not
   be fabricated. Measure delays, report which route actually occurred, and fail
   rather than label a different successful route as coverage of the intended one.
   The actual Direct deadline starts at coordinator D2PPolicyActor.offer_parent,
   not the client's return/sleep timestamp; child arrival and capacity race with
   that deadline. Record the actual path instead of inferring it from sleep.
3. Actual incremental parent reuse, allowing only page-aligned tail recompute;
   distinguish explicit full recompute, capacity rejection and genuine KV loss.
   Compare representative deterministic outputs against a deliberate full
   recompute reference using the same model/TP/sampling, with any numerical
   variation reported rather than hidden behind forced-output constraints.
4. Both TP shards' complete ownership, allocation/admission/fence/retirement
   behavior. Source releases follow physical completion, not merely relay ACKs.
   Report snapshot event sets and any unconsumed terminal/drain states separately.
5. HTTP application_final callback cleanup for last generation, with real relay
   acknowledgement and eventual worker state reclamation. No shared-ledger ACK
   may replace this check.
   HTTP200/relay ACK only proves forwarding, not completed release. Include a
   genuine application-final decision for a normally stopped UNKNOWN/provisional
   generation so this is not merely a no-op callback after length/terminal
   already freed everything. Establish both ranks' reclaim before shutdown.
6. Cancellation/timeout/partial-commit failure evidence at the proper layer;
   existing injected unit/fault tests may establish dangerous failures without
   deliberately corrupting a live allocation. Do not infer those failures from
   success-only model runs.
7. Graceful dependency-ordered shutdown and independent final GPU, listening
   port, Apptainer/squashfuse_ll/fuse-overlayfs and owned-process zero gates.

## Observability and same-host launcher gaps to resolve

Current V2 progress_diagnostics exposes aggregate policy/entry counts and prints
at10-second intervals; a successful short two-turn probe can vanish between
samples. No current public per-generation historical route/ownership query has
been established. Consequently delay labels, cached-token counts, HTTP200 and
periodic aggregate logs cannot prove items2/4/5. Before the model probe, derive
a separately audited per-snapshot observation scheme at existing transfer ACK,
coordinator callback and physical-release completion points. Prefer a scoped
test observer without changing decisions, queues or ownership; do not silently
add a new public runtime service. Record immutable generation/attempt/operation,
endpoint/rank and physically relevant transitions. Observer/fault tests and
audit are required before relying on those records. Nothing is added during
the frozen local comparison.

The current launcher combines all same-host worker ports for overlap checks,
but Router ports are checked only against the configured logical Router node's
worker, and relay ports only against the logical relay node's worker/Router.
Two different logical node IDs on one physical host can therefore hide a
cross-group listener collision. The eventual launcher cleanup must check a
complete per-physical-host port union (all workers/bootstrap/reverse, Router,
metrics, relay), with rejection tests. Until then an exact probe plan needs
that independent union check, not merely load_config success. This is a concrete
unfixed validation gap, not an observed failure of the active local experiment.

The external read-only diagnose_samehost_ports.py under control-plane-draft
reproduces both gaps against the actual current load_config: an example reduced
to disjoint TP2 groups at the same IPv4 is accepted when either router.port or
group_control.port equals the other logical group's worker port. The diagnostic
mocks only Path.read_text to supply JSON in memory; it creates no listener,
process, launch or config change. Acceptance of those invalid configs is the
recorded bug, not a passing launcher safety gate.

## Other representative coverage and final handoff

The actual Qwen3.5 hybrid/Mamba/MoE path also needs explicit representative
verification after maintained entry/config cleanup; native fault tests alone
cannot demonstrate its model runtime. Keep prior baseline source/harness
selection and settings intact. Use cached model/data if present, otherwise
resolve inputs within the authorized allocation workflow without changing shared
environments. A local dense TP1 run does not close hybrid/model-family evidence.

Finally inspect public import aliases, old serialization paths, source maps and
all preserved feature/ablation switches; validate installed package entrypoints,
maintained versus archival launcher references and documentation. Review the
actual table of all accepted before/after runs against ORIGINAL baseline, not
only neighboring revisions, so small drops cannot accumulate unnoticed. Keep
protocol functionality, throughput/variance, accuracy and final resource cleanup
as distinct claims. Missing evidence keeps the full goal active.

## Independent physical-completion audit of the frozen TCP path

Read-only audit againstd265f7946aa3d1369ecdadf9d2d39a77c84e300b confirms that
public progress_diagnostics exposes current counts/phases, not a per-generation
per-rank history of source releases. No existing HTTP/aggregate diagnostic can
replace the required scoped test-only historical observer. Nothing is installed
in the current frozen model case.

The real rank-level GPU source boundaries are the nested commit returned by
AgenticDefaultPhysicalProvider._make_source_direct_handler and its
_host_source_release method. Both invoke bridge.release_after_group_fence before
removing source tracking and registry.retire_after_source_fence. Authority
commit_release must be observed as a live lease becoming absent plus actual
allocator/native release: itsTrue return alone is insufficient because an
already-terminal retry also returnsTrue. NativeRequestMemoryAdapter.release_bound is
the native cleanup boundary; shared Radix pages need not all disappear.

Host restore source commit in make_host_restore_source_handler validates all
target READ receipts, then releases both export and Host snapshot. Observe the
actual SourceLocalHostArena.release live extent removal/free-list return, not
export deletion alone. Terminal Host eviction requires that same per-rank extent
release plus provider.on_committed retiring policy/accounting and terminal state.
Likewise release_final can defer native release; its final sink/composite
_on_final_release and registry retirement are distinct completion evidence.

Group worker _handoff can sendRELEASED before allocator-visible GPU cleanup,
which is deliberately deferred to activate_staged(action="release") in native
TP broadcast order. Therefore neitherRELEASED norACTIVATED ACK alone proves
physical pages are released. application_final HTTP/relay ACK also precedes
eventual reclamation. For a non-no-op terminal probe, establish an actually live
provisional generation first, then prove both ranks' source/Host cleanup and
final coordinator accounting retirement before process shutdown.

A future test-only wrapper must call its original exactly once, preserve return
or exception, record key/attempt/path/group/rank/PID and lease/extent identity,
and distinguish new releases from idempotent retries. Do not replace completion
sinks, add CUDA copies/synchronization, invoke protocol/release actions, acquire
new scheduling locks or block on file I/O in the hot callback. Deferred recording
must treat dropped/overflowed records as evidence failure, not success. No such
observer is implemented or accepted by these design findings.

The retained SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION switch is a legitimate
application-owned terminality scope: AgenticRequestMetadata.classify_output
intentionally returnsUNKNOWN before marker matching when it is enabled. An
explicit separate case can exercise that real provisional/final-callback path,
with an actually live retained generation before the callback. It cannot count
as the genuineTOOL-classification case: keep this switch off for that case and
check the real returned IDs against the declared text/token markers. Neither
test may override classifier results or force output tokens to manufacture
coverage. Record the switch, finish_reason and classification in each case.
