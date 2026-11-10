# Standalone extraction and validation

This migration creates sibling repositories `DualPD` and `sglang_pd` inside the
existing workspace. The original slime and SGLang worktrees are retained for
comparison. It does not replace the environments or allocations used by other
experiments.

The current handoff is [CLEANUP_HANDOFF.md](CLEANUP_HANDOFF.md), with all15
measured stage/baseline comparisons in [ENGINE_CLEANUP_RESULTS.md](ENGINE_CLEANUP_RESULTS.md).
Detailed earlier progress below is chronology, not a claim that an old pending
gate is still pending or that historical test counts equal the latest reports.

## Scope and lifecycle mapping

The migration changes Python namespaces, source organization and launch paths.
It does not add an ownership state or change any P_HBM -> P2D_HOST -> D_HBM ->
D2P_HOST transition. Physical fences, durable commits, TP group decisions,
workset accounting, failure, timeout and cancellation semantics remain in the
preserved engine implementation. Both local and TCP runtime paths are retained.

`sglang_pd/dualpd-source-map.json` identifies the original engine commit and
records every relocated implementation/test SHA-256. Initial extraction kept
native integration files byte-identical; subsequent Host/Router/Scheduler
component moves have complete reviewed source/AST reconstruction proofs.
Unmodified allocator/model/forward files retain byte-level comparisons. A type-only HostKVCache
import is deferred in hicache_storage so CPU routers do not need CUDA kernels
solely to import metadata definitions.

The frontend's Sample class, Dataset implementation, processing and trace
utilities were extracted from the original snapshot. GenerateState retains its
singleton, tokenizer/processor, sampling parameters, abort flags and semaphore;
training submission/reward/weight-update code is omitted. Local HTTP pooling,
timeouts, retries, proxy policy and response closure remain unchanged. Ray-based
training dispatch is not part of the standalone inference API.

The project root directly corresponds to the former `examples/pd` directory.
The intermediate source tree and compatibility symlinks have been removed.
Flat harness imports are remapped to `dualpd.*`; no
`slime` import alias or bundled Slime package is installed. The standalone JSON
name `dualpd_root` is accepted alongside the old `slime_root` key, and a conflict
is rejected. Existing experiment inputs and values remain explicit.
Source equivalence additionally accounts for the flattened inference path in
the Retool docstring and the requirements path in a SWE installation error;
these are literal documentation/diagnostic changes, not serving logic.

## Regression gate

### Router ownership cleanup

Both Router implementations and their unit tests now live in the engine's
`python/sglang/srt/disaggregation/dualpd/router/` package. The local
late-binding and TCP/global implementations initially matched the frontend
snapshot byte-for-byte; subsequent component separation preserves their complete
reconstructed method bodies, native callbacks and policy differences. The
frontend keeps only thin launch entries, deployment planning and
the application-side request/ACK protocol. No engine Router imports DualPD.
The package initializer is lazy so importing one variant never activates the
other variant's FastAPI routes.

The explicitly historical Mamba launcher retains its matched ledger protocol.
Its immutable Router revision is vendored inside the engine instead of loaded
from the original Slime Git repository at runtime. It is not the default Router
and is not substituted for the current implementation. Router source hashes
are recorded in `sglang_pd/router-source-map.json`.

Thin local launch entries resolve the sibling engine directory (overridable
with `DUALPD_ENGINE_ROOT`) without prepending it to `sys.path`. If no override
or sibling checkout exists, the shared stdlib-only `runtime/engine_entry.py`
resolves the selected installed SGLang disaggregation package and executes its
engine-owned Router file. The helper itself directly imports only the standard
library; dotted package discovery may initialize the selected SGLang parents,
so the installed fallback is not claimed to be Torch/CUDA-free.
Explicit empty/invalid overrides and incomplete
existing sibling checkouts fail rather than silently selecting another engine.
The two thin scripts load this helper by neighboring file, so direct execution
does not newly require an installed DualPD package. Historical
Mamba/Fused overlays therefore remain selected for SGLang protocol imports;
an old overlay need not contain the new Router namespace. A regression test
blocks that namespace and verifies both launchers still process `--help`
without changing the selected overlay or adding/reordering source search paths
(Torch's generated temporary compilation directory is permitted).

Stage8 installed-entry audit reproduces the original non-editable wheel Router
failure, then successfully runs the candidate CLI with frontend and engine
wheels separately installed into isolated targets. Both actual import origins
are checked; invalid/empty engine overrides still fail. Eleven focused tests
cover source/installed address selection, exact exception propagation and
stdlib-only direct-script execution. Complete source proof still reconstructs
all142 archived files,54 unchanged retained code files and the two exact
post-archive Router entry transforms at that stage; the new helper is whole-file
hash pinned. The later shared-tool path boundary has52 byte-identical retained
code files plus2 exact Router-entry and2 exact shared-tool reconstructions.
This is not a source-tree-free Slurm deployment claim: cluster defaults still
expect an engine source root. Wheels do not include all root reconstruction
source maps/locks or original Git history, and source-verification tools remain
source-checkout tools. Packaging checks do not establish actual TCP/TP2/hybrid
model forwards or throughput acceptance.

Lifecycle mapping for this move: no state transitions, fence/CAS changes,
reservation changes or TP decisions. Success, failure, timeout, cancellation
and shutdown follow the same implementation. Thresholds, sampling, host sizes,
GPU placement and 300+1200/T450 measurement windows remain unchanged.

## Verification commands

### Root-module cleanup

The root contains only `inference.py`, `dualpd.py`, `__init__.py`,
`__main__.py` and `setup.py`. Application-side request metadata/ACK helpers,
metrics, HTTP transport and the CLI live under `runtime/`. They remain client
code: no KV allocator, ownership or DMA state machine is moved out of SGLang.
Mixed-harness generation is in `data/generation.py`; the pre-existing workload
sampling implementation in `data/dispatch.py` remains unchanged.

Root compatibility aliases for BrowseComp, Retool, tools and search were
removed after all active callers were changed to the real `data.*` modules.
Search launchers use `python -m dualpd.data.browsecomp.search_server`.
The local Router deployment shim is under `scripts/new_method/internal/`,
alongside the legacy Mamba shim; Router implementation remains engine-owned.
Source snapshots now preserve the relocated runtime files' relative paths.

Install the updated editable package again to refresh the console entrypoint:
`pip install --no-deps --no-build-isolation -e .`. The command resolves
`dualpd.runtime.cli`, not the removed root `cli.py`. Only the new isolated
environment is refreshed; baseline and original worktree environments are not
modified. Root-layout, active-import, shell-reference, outside-checkout CLI and
historical-overlay checks are part of the CPU gate.

```bash
.venv/bin/python scripts/check_cpu.py
.venv/bin/python scripts/verify_snapshot.py
.venv/bin/python scripts/verify_launch_plans.py
```

The CPU gate covers request metadata, router reservations and retries, workload
sampling, SWE tool loops, HTTP transport, multi-node plans, process cleanup, TP
control protocol, cancellation/fault handling, Host transfer and memory leases.
Logs, JUnit XML and machine-readable return codes are saved in `tests/reports/`.
`verify_snapshot.py` is a migration-only reference check requiring the retained
original sibling snapshots. Production imports and the CPU gate do not need them.

The first internal engine cleanup moves the unchanged Host ledger to
`agentic_host_ledger.py` and engine tests to `dualpd/tests/`. Historical Host
imports remain class-identical aliases. The source map retains original hashes,
with current hashes for nine explicitly reviewed source/test transformations;
`sglang_pd/tools/verify_dualpd_cleanup.py` performs strict AST and environment
identifier checks against the pre-cleanup commit. The multi-node launch-plan
comparison now also allows removal of the unused
`SGLANG_AGENTIC_MEMORY_AUTHORITY_V2` helper export; real activation remains
`SGLANG_AGENTIC_MULTINODE_ENABLED`. All operational flags and defaults remain.

An old router test expected `remote_read_epoch`, which is absent from the source
engine. It now checks the actual existing recovery_claim_id, owner, state and
per-rank pinned claims instead. No engine ownership behavior was changed for it.

The broader inherited test inventory is **not fully green in this environment**:
some imports require libcuda, exploratory probes require smolagents and external
datasets, old baseline AST tests point at retired /homes/siqic environments, and
several historical experiment-controller tests assume separate processes but
mutate shared globals during collection. The dedicated CPU gate does not claim
to validate those optional or archival paths. They remain available for their
original environments; failures discovered during broad exploration are recorded
here instead of being silently reported as successful checks.

## GPU acceptance

CPU checks and byte/AST equivalence establish extraction compatibility, not
measured GPU equivalence. GPU validation must pass an independent audit first,
use an available allocation, and record:

- Dense and hybrid multi-turn Direct and Host correctness, including TP>1.
- Source release and durable handoff ownership conservation.
- Identical sampling/workload/placement/capacity values in old and new plans.
- Full 300 s warmup + 1200 s measurement for final BrowseComp throughput.
- New SWE performance/accuracy claims additionally require T450 and preceding
  tokens/rounds/accuracy with unchanged task order. These have not been remeasured
  by this BrowseComp-based cleanup; preserving source/harness contracts is not
  a new measured accuracy result.
- Post-run GPU, port, container and FUSE cleanup.

Initial extraction checks alone did not establish end-to-end model performance.
Subsequent staged cleanup has accepted real local Qwen3-8B TP1 BrowseComp
comparisons on node2700; exact revisions, full windows, original-baseline
comparisons, ownership evidence and final gates are recorded in
`SEQUENTIAL_ENGINE_CLEANUP.md` and
`../runs-host/ENGINE_CLEANUP_QUICKCHECK_H100.md`. That local result alone does
not establish TCP/TP2/hybrid forwards or SWE accuracy preservation. Subsequent
dense r7/r8 and hybrid r9 probes provide separate, bounded real-model TCP/TP2
evidence, documented in [FINAL_ENGINE_VALIDATION_PLAN.md](FINAL_ENGINE_VALIDATION_PLAN.md)
and [HYBRID_RUNTIME_VALIDATION.md](HYBRID_RUNTIME_VALIDATION.md). The latter
preserves the original stable-prefix compatibility mode, not full-parent
page-tail-only reuse. Focused faults/CUDA tests, model functionality, throughput
and benchmark accuracy remain distinct evidence classes.
All eight lifecycle acceptance criteria in
`AGENTIC_PD_DESIGN_INVARIANTS.md` remain binding. Deleting an
apparently obsolete transport branch is a separate change requiring reachability
evidence and lifecycle/fault tests; this migration does not infer obsolescence
from `legacy`/`v2` naming or from a default-off experiment switch.

## Historical experiment replay

The shared BrowseComp matrix implementation is tracked in
`scripts/replay/cluster/matrix_controller.py`. Existing model-specific wrappers can use it
without loading code from a scratch run directory. The maintained staged
quickcheck no longer imports that historical controller: its fixed recipe and
measurement helpers live in `scripts/benchmarks/`. Classification and relocation
of201 original launcher files are accepted with immutable source reconstruction,
layered import/argv contracts and a separate full BrowseComp comparison. This
does not mean all historical Slurm/Docker recipes were executed, or imply that
their scripts or retained feature switches can be deleted.
Frozen DRAM rerun controllers
still intentionally require their immutable launch_config/code_identity and
engine snapshot. Those are experiment data dependencies, not project libraries.
Keep them immutable when comparing with published results.
