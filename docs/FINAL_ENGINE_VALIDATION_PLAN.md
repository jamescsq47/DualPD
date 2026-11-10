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
Router comparison a TCP model test would be incorrect. Stage7a's maintained
benchmark extraction comparison is independently accepted at11193.29 Decode
token/s (−0.64% versus ORIGINAL),100% reuse and clean final gates; this removes
historical-controller dependencies from the maintained quickcheck, not from all
replay families. Stage7b archival is independently accepted at11279.34 Decode
token/s (+0.13% versus ORIGINAL),100% reuse, full300+1200 and clean final gates.
This is local TP1 evidence, not actual TCP/TP2/hybrid model coverage or complete
operational execution of archived historical recipes.
Nine model/TP dry plans
and real-kernel TP/Mamba fault fixtures are source/functionality evidence, not
actual distributed model forwards or accuracy verification.

## Accepted dense same-host TCP/TP2 probe, 2026-11-10

`dense-tp2-native-r7-20261110` now provides independently audited real-model
evidence for the specified Qwen3-8B two-turn scenarios. P uses GPUs0/1, D uses
GPUs4/5 on the retained node2700 allocation, with distinct logical groups and
the maintained TCP relay, router, worker protocol and HTTP final callbacks.

| Tool wait | Observed parent return | Imported parent, each P rank | Child vs fresh full recompute |
|---|---|---|---|
|0s|D2P Direct|192/195 tokens|13 output IDs and finish reason equal|
|2s|D2P Host store and restore|192/195 tokens|13 output IDs and finish reason equal|

Inputs use the actual native tools template, original TOOL classification,
returned token IDs and a deterministic tool-response suffix; no forced output,
retokenized parent or synthetic lifecycle marker substitutes for the model.
The3-token difference is the allowed page tail, not full recomputation.
Decode HTTP `cached_tokens=0` is not the P-side imported-parent metric.

Four healthy hash-bound final checkpoints (request12),14 Direct source chains,
two-rank Host READ receipts/commit and actual source Host extent reclamation
support the named transfers. Both normally stopped provisional children receive
genuine final callbacks; terminal Host extent release and coordinator retirement
are observed before shutdown, not inferred from callback ACKs. Independent
review verifies final GPU/helpers0, nine free ports, unchanged zero memory
events, original keeper identity and matching postrun source/input freeze.
Peer-disconnect logs occur after the final checkpoint during dependency-ordered
stop; controller exit0 does not imply every stopped child naturally exited0.

### Transport setting and reproduction boundary

For this exact node-local TCP recipe, set the existing `ucx_net_devices` field
to `"mlx5_0:1"` on **both** P and D node entries. The maintained launcher already
maps this field to each worker's `UCX_NET_DEVICES`; no runtime source, shared
environment or global default was changed. Confirm the actual NIC/port is
active on the intended allocation; do not copy this machine-specific name into
unrelated machines or publish it as a universally optimal setting.

The default-device r6 failed during Host READ. Separate CPU-only partial/full
metadata tests pass6/6, while the default-device memfd DRAM-to-VRAM probe stays
PROC at its15s deadline6/6. Its UCX protocol table selects two InfiniBand rails
for the actual Host-to-GPU read. Restricting the same tiny probe to the above
NIC passes6/6 full1MiB payload checks; r7 subsequently completes the real-model
Host path. This supports the configuration remedy, not a precise distinction
between a bad NIC, multirail selection, topology or fabric connectivity. It
does not justify changing ownership/fence logic or NIXL versions.

Artifacts are under the external validation pool's `tcp-tp2-probes/`:
`native-read-diagnostic-summary-20261110.json` and
`dense-tp2-native-r7-20261110/` (runtime result, source/input freeze, checkpoint
journals, runbook, final gates). The reviewed freeze covers5631 files and the
unchanged258-library native provenance. Preparation passes231 tests plus227
subtests,121 existing Host/TP fault tests, and30 independent contract tests.

This is bounded dense functionality, not concurrency/throughput acceptance,
multi-host failure-domain validation, complete allocator/Radix quiescence,
TCP P2D Host spill, hybrid/Mamba/MoE runtime or benchmark accuracy. Those
remaining requirements below stay open; old failed runs are not relabeled.

## Accepted dense TCP/TP2 P2D Host probe, 2026-11-10

`dense-tp2-p2dhost-r8-20261110` subsequently covers the P2D Host case left open
by r7. It retains the same model, TP/GPU layout, fixed NIC, capacities,
tool/admission thresholds and native two-turn client. The sole transport
configuration change is the existing `p2d_late_bind_grace_seconds=0`. This
expires the normal unbound-snapshot timer immediately, without replacing
ownership/fence logic; if D wins the binding race, Direct remains legal and
would not count as Host coverage. In this actual run all six logical
parent/child/reference snapshots take Host.

| Actual per-rank evidence in final checkpoint12 | Count |
|---|---:|
|P2D Host store, successful physical DMA|12|
|P source Req/authority/registry retirement after store|12|
|D Host READ complete and destination ready/bound|12|
|P source Host extent live-to-absent after loaded READ receipts|12|

The two D-rank READ receipts are observed at both source ranks:24 receipt
observations describe12 rank reads, not24 transfers. All six P2D coordinator
records retire. The0s parent still returns D2P Direct; the2s parent returns
D2P Host. Both P ranks import192 of195 parent tokens, and each13-token child
output and finish reason equals its fresh full-recompute reference. Normally
stopped provisional children receive genuine final callbacks; their terminal
Host extents and coordinator state retire before shutdown.

Independent audit verifies four healthy hash-bound checkpoint prefixes, actual
source release and target-ready states, final GPU/helpers0, nine free ports,
unchanged zero memory events and matching postrun source/input freeze. Controller
exit0 and planned dependency-stop logs do not mean all child services exited
naturally. Source extents returned to an arena do not prove that the entire
arena mapping/registration was removed before process shutdown.

Artifacts are in `tcp-tp2-probes/dense-tp2-p2dhost-r8-20261110/` under the same
external validation pool. Preparation passes231 tests plus227 subtests,
102 existing Host/transfer/provider tests and9 independent contract tests.
This closes only the specified dense TCP/TP2 P2D Host functional coverage.
The0grace recipe is not the original BrowseComp performance configuration;
hybrid/Mamba/MoE, multi-host, stress and full-goal acceptance remain unproven.

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

The following paragraphs retain the prerequisites identified before r7. The
dated result above records their scoped implementation and evidence for that
case, not a new public historical-query service or coverage of other models.

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

The historical launcher checked Router and relay listeners only against their
own logical node's worker ports, allowing cross-group collisions on one host.
Stage6 fixes those three checks: worker-port unions include all groups with the
same explicit host_ip, and relay/Router co-location compares host_ip rather
than logical node_id. Real-parser rejection tests and unchanged valid launch
plans cover that configuration fix. It does not discover DNS aliases or
multi-NIC physical-host identity. The actual TCP probe must therefore use one
consistent routable IPv4 for same-host groups and independently review the full
listener union; parser acceptance alone still does not establish live readiness.

The external read-only diagnose_samehost_ports.py under control-plane-draft
reproduced both gaps against the pre-stage6 load_config: an example reduced
to disjoint TP2 groups at the same IPv4 is accepted when either router.port or
group_control.port equals the other logical group's worker port. The diagnostic
mocks only Path.read_text to supply JSON in memory; it creates no listener,
process, launch or config change. Acceptance of those invalid configs is the
recorded historical bug, not a passing launcher safety gate or evidence that
the stage6 fix is absent from the current tree.

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
observer was implemented or accepted at the time of these design findings.
The external test-only observer subsequently used for r7 is covered by the
dated scoped evidence above; these historical requirements are not a claim
that observer preparation alone validates any additional model/path.

Observe physical after-state on exceptional exits too: native cleanup or lease
removal can succeed before a later capacity callback or transport cleanup raises.
An exception alone must not be interpreted as "ownership was retained". The
initial external generic recorder preparation preserved the original exception
but recorded only its before-state on that path. The later external draft now
records protected exceptional after-state and tests partial cleanup followed by
an exception. This closes the generic fixture observation gap, not actual native
integration: real per-rank hooks, producer identification and report delivery
remain uninstalled and require separate acceptance. The original native result
or exception must remain unchanged.

The retained SGLANG_AGENTIC_KV_APP_OWNS_TERMINATION switch is a legitimate
application-owned terminality scope: AgenticRequestMetadata.classify_output
intentionally returnsUNKNOWN before marker matching when it is enabled. An
explicit separate case can exercise that real provisional/final-callback path,
with an actually live retained generation before the callback. It cannot count
as the genuineTOOL-classification case: keep this switch off for that case and
check the real returned IDs against the declared text/token markers. Neither
test may override classifier results or force output tokens to manufacture
coverage. Record the switch, finish_reason and classification in each case.
