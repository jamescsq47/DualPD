# Next engine boundaries: read-only findings during local Router validation

Reference inspected: engine72518d3db7ece7e79fedd759d2bcf0f53d48521e.
This document does not authorize skipping the current actual BrowseComp gate.
No following runtime extraction starts until router-local-r1 completes its full
measurement, source revalidation and resource cleanup with acceptable throughput.

## TCP Router

The current global_pd_router.py is746 lines with29 concrete methods. Its group
protocol and default values differ from the local Router; do not inherit the
local policy components or merge reservation record types. Keep the native
mini-lb app callback decorators, public class identity and main service entry at
the existing root so lazy package imports do not register callbacks twice.

A proportionate boundary is two method-only components, not a file per helper:

| Owner | Methods / responsibilities |
|---|---|
| Concrete TCP Router |8: constructor, close, select_pair, request group annotations, room parsing, ready callback/waiter |
| TCP capacity |12: least-load choice, sample epochs/polling, prompt/provisional charges, feasible Decode selection, reservation expiry/materialization/release |
| TCP HTTP |9: backend timeout, task settlement, generation keys/retry cache, duplicate-selection release, generate/once/stream |

The private TCP Decode reservation dataclass can live with its sole capacity
consumer and remain reexported by the historical root. It must not be aliased to
the structurally different local record. Whole29-method/class/module AST
reconstruction, defining-module runtime bindings, unchanged callback bodies/
decorators, old aliases and native base/MRO must be proved. Existing tests patch
root.send_application_final and the shared aiohttp module; retain those contracts.
Helpers must not import the concrete root back: a -m service entry otherwise
risks loading it twice and registering duplicate callbacks on the shared app.
Relevant tests must explicitly retain causal post-materialization sampling,
TP token charging, capacity waits, idempotent callback retry, frontend-cancel
shielding, backend settlement and streaming/logprob cleanup. The local frozen
throughput case is not evidence that TCP itself serves a full model correctly;
record TCP-native fixtures and actual model-path coverage separately.

## Scheduler

The inspected scheduler.py has an approximately1270-line physical-workset lease/broker block
before its native Scheduler class, plus a separate early-Direct receive record.
The Scheduler's54 agentic-named methods are dispersed among native hooks; merely
moving top-level records is not completion of stage4. Start with the coherent
broker/lease owner, then separate Direct progress, TP coordination and ready/
admission responsibilities while keeping the native event-loop integration
explicit. Exact inventories must be derived from current ASTs, not assumed from
method names alone; mixed native methods retain agentic call sites too.

There are seven explicit Scheduler attribute references inside the current
implementation (including two control-key constants). Some fault tests invoke
Scheduler._agentic_*(SimpleNamespace(...)) and patch native-module metadata.
Consequently a new mixin must not silently require an instance of that mixin or
import Scheduler back during module initialization. Class-specific dispatch,
static descriptors, decorators, constant owners and hook patch sites need an
explicit compatibility design before moving any methods. Preserve annotation
evaluation/import timing: native scheduler has no future-annotations directive.

For every component, map successful handoff, capacity rejection, partial DMA,
timeout, cancellation and shutdown to the existing owner/fence/CAS and release
rules. Preserve eight invariants and require complete original reconstruction,
native TP/Mamba/Host faults, independent audit and a separate frozen comparison.

Read-only workset findings during router-tcp-r1 (engine3d1e524): lease44 lines
and broker1223 lines have no Scheduler global reference. Their external globals
are typing aliases, dataclass/deque/threading, NumPy/Torch and kv_to_page_indices.
Keep those two complete definitions together; leave the separate33-line early-
Direct receive record for its coherent Direct component rather than putting it
inside a generic workset file. Preserve eager annotation evaluation (no new
future-annotations directive), old Scheduler aliases and the native definition/
initialization order.

The existing test_agentic_h2d_decoupling.py readiness-sync test patches
sched_module.kv_to_page_indices specifically for broker.service. Its two-call
assertion must remain intact, retargeting only that defining-module private hook
after extraction. This file is not part of the previous557-native inventory;
explicitly add it for the workset stage. A current search also finds Scheduler
consumers in TP Host commit/cancel, resident Mamba admission, request-owned Mamba,
prompt checkpoint, memory-usage snapshot, decode-memory bridge, multinode shell
and frontend scheduler-priority tests. Review/include these relevant fixtures,
not just the already passing coarse TP/Host inventory.

The first native Scheduler edit also needs an explicit reviewed same-path entry
in dualpd-source-map.json: the current snapshot verifier otherwise correctly
rejects it as a changed native file. Prove reconstruction of the entire original
native module and all remaining mixed methods/initialization, not a filename or
hash waiver. Moving only these two classes is an intermediate gate, not stage4
completion; actual Direct/TP/ready/admission responsibilities still need cleanup.

## Still required afterward

Genuinely remove metadata-only control-plane imports of native Host/DMA/worker
modules; moving a definition without redirecting consumers is insufficient.
Current canonical local Router root, capacity, admission and records import
HostStageState through heavy staging; root/admission import three P2D protocol
names through the full P2D transport. Redirect every one of those imports, not
just the root. The seven remaining storage/metadata definitions can move as one
coherent layer; preserve lazy optional storage imports and historical aliases.
The lifecycle and Host-ledger modules themselves have stdlib-only top-level
dependencies, making them viable control-plane dependencies.

Historical file-entry overlays need explicit treatment: an old selected engine
can lack the newly extracted ledger/protocol modules. Do not fix this by
replacing its native sglang package or duplicating enum/state-table identities.
Design and audit an explicit compatibility path before changing these imports;
the normal canonical Router must prove it imports without Torch/CUDA/Host
workers, while an intentionally historical overlay may retain its old heavy
imports and must be labeled separately. Test both in genuinely fresh processes
with forbidden native imports and separately with the actual old overlay.
The fresh-process canonical proof must cover the complete import closure,
including sglang package initialization and native mini-lb tracing, not only
the new leaf's own imports. If compatibility uses a fallback, match only a
missing selected-overlay module itself; never hide arbitrary nested ImportError
or load a duplicate ledger under a second module name. Scheduler reconstruction
must cover the complete native module, including mixed initialization/event-loop
call sites, decorators/defaults and annotation timing, not just54 named methods.

Retain143 operational environment identifiers, local/TCP defaults and dynamic
versus constructor-time parsing scopes. Separate maintained/archival launchers
without invalidating replay identities, then perform final package/source-map/
functionality/representative-performance and resource-cleanup acceptance.
