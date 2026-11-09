# Control-plane dependency cleanup: next-stage scope

Read-only design during scheduler-components-r1. No serving code in this plan
is applied before that case's throughput, ownership and final cleanup acceptance.

## Current evidence

The maintained local Router root imports HostStageState and SharedHostStagingLedger
through agentic_host_staging. Its local_capacity, local_admission and local_types
components also import the enum through that worker/transport module. Root and
local_admission import only P2D_CUSTOM_PREFILL_DOMAIN, P2D_CUSTOM_SNAPSHOT_ID and
p2d_snapshot_id from p2d_host_staging, which imports torch, physical Host buffers,
worker pools and native KVPoll. None of those three names requires a device.

The existing agentic_host_ledger is already standard-library-only and is the
single implementation of HostStageState and SharedHostStagingLedger. Importing
these exact objects directly can remove a needless transport dependency without
creating another enum, class, ledger or owner. Historical public imports must
continue aliasing these same objects, including old pickle paths.

The current p2d_snapshot_id body is exactly `return f"p2d:{int(bootstrap_room)}"`;
the two custom keys are constant strings. A focused protocol module should own
these actual wire declarations and the unchanged helper, with old import aliases
retained. Do not duplicate literal strings in the Router. Other P2D functions
and state tables are not automatically part of this move: their existing module
globals/private patch sites must be assessed separately rather than moved just
because they appear above a device class.

A fresh-process maintained-path probe on2026-11-09 installed a MetaPathFinder
before importing the canonical Router and rejected both physical staging module
names. It failed exactly at late_binding_router.py:36 importing
sglang.srt.disaggregation.agentic_host_staging, before any preimport of transport
modules. This is an intentionally failing dependency diagnostic, not a regression
test failure or proof of independence. The future test must pass that blocker
and separately demonstrate that P2D wire metadata no longer imports its worker.

TCP Router uses agentic_group_protocol rather than the local Host path. Keep
that distinct authority/protocol and its file-free operation unchanged.

## Historical overlay constraint

router/launch_late_binding_router.py supports a selected old native overlay by
loading only maintained Router files as the private file-package
dualpd_engine_late_binding_router. It must not prepend the active engine to
sys.path, replace sglang.srt.disaggregation, or require the old overlay to contain
the new dualpd package. The pinned legacy_mamba_pinned implementation remains a
separate historical replay source, not an invitation to update old experiments.

Before changing Router imports, derive an explicit maintained-versus-old-overlay
contract. A new canonical ledger/protocol import must not silently break that
existing old-overlay entry. If compatibility resolution is required, it must be
narrow, tested in a genuinely missing-module overlay, preserve selected-overlay
object identity and avoid catching unrelated import/runtime failures. No hidden
copy of ledger classes or enum and no broad sys.path fallback is acceptable.

Independent design review refines the intended bridge to one relative
local_dependencies leaf exporting exactly five existing objects: the enum,
ledger class, two P2D custom keys and ID helper. Four Router consumers should
share those bindings. Only the historical private package may fall back, and
only when ModuleNotFoundError.name exactly equals the requested new canonical
leaf; a missing parent, nested dependency, missing symbol or other error must
propagate. Ordinary canonical imports must never fall back, including packaging
mistakes. Ledger and pure protocol can resolve independently: the three protocol
declarations hold no Host state, so do not couple their fallback choices or move
capacity/committed-state tables into that protocol module.

Resolution tests must exercise all four ledger/protocol availability combinations
in private mode and fail canonical mode when either leaf is missing. A real
selected old overlay must prove exact old enum/ledger/function identities and
old pickle paths; an installed modern overlay with merely blocked dualpd does
not suffice. Canonical fresh-process tests must cover the entire Router import
closure, not just the new helper, with workers/Host buffers/DMA/Scheduler blocked.
Legacy fallback may retain old heavy imports but cannot be reported as the
maintained lightweight path. These are design-review conclusions, not code GO.

## Required proof before applying

1. Exact original declarations/helper AST, old public aliases, serialization
   identity and any dynamic/private global hooks remain valid. The worker and
   Router whole-module inverse proofs must incorporate only the precise import
   changes; no filename exemption or assertion removal.
2. Fresh maintained control-plane imports must block the actual physical Host
   worker/buffer/DMA modules and native Scheduler. Do not preimport those modules
   before installing blockers. Distinguish extension independence from NumPy or
   other imports inherently used by native SGLang/HTTP dependencies; do not claim
   an entirely standard-library Router.
3. Verify both canonical package and historical private-file entry paths. The
   historical test must exercise a selected overlay actually lacking new modules,
   not merely block the dualpd package after preimporting maintained workers.
4. Test ledger claim/release/replay, complete-snapshot states, P2D metadata,
   capacity/full-parent-plus-suffix reservations, cancellation and TP ordering,
   plus import identity and native consumers. No new tasks, workers, queues or
   physical ownership transitions may be introduced.
5. Full CPU/native fault gates, independent eight-invariant code audit, actual
   CUDA checks as relevant, committed clean source, exact-plan audit and a new
   frozen BrowseComp300+1200 comparison against ORIGINAL baseline-r3 are separate
   gates. Existing local checks cannot replace genuine TCP model-path coverage.

These findings define the next action; they are not evidence that stage5 or the
overall cleanup has been implemented or accepted.

## Unapplied compatibility unit draft

An external candidate and stdlib unittest fixture live under
/orcd/scratch/orcd/003/shangy/dualpd/runs/engine-cleanup-validation-20261108/control-plane-draft.
The six tests pass and cover four availability combinations, exact selected
object references, canonical fail-closed behavior, unrelated/nested import and
runtime errors, missing symbols and similarly named packages. Their importer
uses controlled fake objects: this proves only the resolution branches, not
real old-overlay compatibility, the complete maintained import closure or any
native/ownership behavior. No serving tree or transport module was edited.

Independent review reproduced all six initial tests and identified a crossed
failure: resolving protocol before checking ledger attributes could needlessly
load legacy P2D when the ledger interface was already broken. The draft now
binds both ledger objects before resolving protocol. Two additional test groups
cover that crossed failure and propagation of legacy fallback's own exceptions;
all eight groups pass. The four availability combinations additionally assert
the exact import order. An isolated p2d_host_protocol candidate contains only
the two custom keys and unchanged ID function; verify_protocol_draft.py proves
all three complete AST nodes equal the current frozen original and rejects any
other executable code/import. This is preparatory draft evidence only.

The draft source proof now reads immutable engine commit
a550ccaecad5d2b27ebb15a2788c1e1dc0daff8f through git show, rather than a mutable
working file, and compares the entire candidate module against the exact
docstring/future-import plus three original nodes. Relative future imports,
aliases, duplicate/extra nodes or imports cannot pass that exact module proof.

## Full external candidate during the retry

The scratch quota failure prevents new draft/output growth there; the preserved
initial draft remains readable. The full UNAPPLIED seven-file candidate now
lives under /orcd/pool/004/shangy/dualpd/runs/engine-cleanup-validation-20261108/control-plane-draft/candidate/sglang_pd.
It contains exactly four Router consumer files with six import-owner changes,
the reviewed five-binding bridge, the pure three-declaration protocol, and the
worker retaining public imports for those declarations. Capacity/state tables,
all worker bodies and all Router bodies remain original. The external generator
reads immutable a550ccaecad5d2b27ebb15a2788c1e1dc0daff8f and refuses a dirty or
different serving checkout; it only prints patches to the isolated candidate.

verify_candidate.py reconstructs entire four Router modules, checks exact pure
protocol header/body, the entire remaining worker and complete reviewed bridge.
Independent review reproduced its initial baseline and12 destructive mutations,
then identified that merely stripping the worker import did not lock its
initialization position. The checker now compares the complete expected worker
including that import immediately after torch, with a new import-relocation
negative fixture. Three unittest groups cover baseline and13 negative mutations.
This remains source-only preparation, not final formal code/native/GPU GO.

Before application, integrate these exact inverse rules with existing Host and
Router verifiers/maps without weakening their prior whole-body checks. Real
old-overlay missing-leaf execution and fresh canonical full-closure blockers
remain pending. Preserve old worker function aliases and literal old pickle
paths; the function's new natural module is the pure protocol, not a stamp back
to the heavy worker that would force control-plane pickling to import transport.

Candidate test preparation now includes nine portable groups: the original
eight controlled-importer branch groups still pass, and a new fresh-process
canonical Router import test blocks physical Host/P2D/DMA/worker/Scheduler and
frontend imports before the complete entry, then checks consumer aliases and
pure-protocol pickle without loading transport. That full-closure test is
intentionally NOT run against the still-frozen/unapplied serving version.
Two separate real-kernel/native cases are prepared and syntax-checked only:
public worker/protocol/ledger aliases and literal old pickle paths, and a real
selected legacy python tree configured with DUALPD_TEST_LEGACY_ENGINE_ROOT.
The latter requires both new leaves genuinely absent, verifies original-old
source locations, suppresses bytecode writes there and checks native namespace,
path and exact old-object identity through the private entry. Neither case has
run or passed yet; no fake-object branch result replaces them. Complete source
proof integration and actual application still wait for the current GPU run.

## Formal implementation after Scheduler acceptance

Scheduler-components-r2 has independent final acceptance; the freeze is released.
The seven reviewed runtime files are now applied to the maintained engine only.
The bridge's draft-only docstring wording is removed; its complete-source hash
is relocked. The new production verifier references immutablea550 and checks all
four Router modules, six exact import moves, complete worker and protocol, exact
worker import position and the complete bridge. The existing Router inverse
proof calls that gate before undoing the six imports, retaining all100 local and
29 TCP method/body/global-binding checks. The Host verifier includes that proof
and only the explicit P2D worker map transformation. Original source hashes remain
preserved; current hashes are updated for the worker and local Router root.

Nine portable dependency groups and three structural mutation groups are added
to the standard CPU gate; two real-kernel/legacy tests remain separate native
gates. At this point source reconstruction passes, but full CPU/native, final
independent code GO, actual CUDA and the next frozen300+1200 model comparison
are pending. Applied code is not yet final stage acceptance.

Full gate results: CPU813 records (208 frontend+605 engine), real-kernel native
r2 1018 (981 ordinary+37 subtests), and actual H100 registered/bounce roundtrip2.
All final reports have zero failures/errors/skips. The initial native report
retains one new fixture error because the real old namespace package has no
__file__; the correction verifies its exact sole resolved __path__ and preserves
all original identity/source assertions. No runtime fix was needed. Independent
CODE GO binds runtime bundle34823ae87f11255a661a85dcfdda82a7e8a304fce208457573b6a2a6eaf46b0c.
Engine committed clean asd265f7946aa3d1369ecdadf9d2d39a77c84e300b;
74 extension and3 Router map hashes match actual files. Full protocol/native
proofs and all nine launch plans pass. A new exact-plan audit and complete
BrowseComp comparison are still required before performance acceptance.
