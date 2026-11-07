# Standalone extraction and validation

This migration creates sibling repositories `DualPD` and `sglang_pd` inside the
existing workspace. The original slime and SGLang worktrees are retained for
comparison. It does not replace the environments or allocations used by other
experiments.

## Scope and lifecycle mapping

The migration changes Python namespaces, source organization and launch paths.
It does not add an ownership state or change any P_HBM -> P2D_HOST -> D_HBM ->
D2P_HOST transition. Physical fences, durable commits, TP group decisions,
workset accounting, failure, timeout and cancellation semantics remain in the
preserved engine implementation. Both local and TCP runtime paths are retained.

`sglang_pd/dualpd-source-map.json` identifies the original engine commit and
records every relocated implementation/test SHA-256. Native scheduler, allocator,
model and forward integration files remain unchanged. A type-only HostKVCache
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
late-binding and TCP/global implementations are byte-identical to the frontend
snapshot. The frontend keeps only thin launch entries, deployment planning and
the application-side request/ACK protocol. No engine Router imports DualPD.
The package initializer is lazy so importing one variant never activates the
other variant's FastAPI routes.

The explicitly historical Mamba launcher retains its matched ledger protocol.
Its immutable Router revision is vendored inside the engine instead of loaded
from the original Slime Git repository at runtime. It is not the default Router
and is not substituted for the current implementation. Router source hashes
are recorded in `sglang_pd/router-source-map.json`.

Thin local launch entries resolve the sibling engine directory (overridable
with `DUALPD_ENGINE_ROOT`) without prepending it to `sys.path`. Historical
Mamba/Fused overlays therefore remain selected for SGLang protocol imports;
an old overlay need not contain the new Router namespace. A regression test
blocks that namespace and verifies both launchers still process `--help`
without changing the selected overlay or adding/reordering source search paths
(Torch's generated temporary compilation directory is permitted).

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
- SWE T450 and preceding tokens/rounds/accuracy with unchanged task order.
- Post-run GPU, port, container and FUSE cleanup.

No end-to-end GPU model-serving or performance acceptance is claimed by the extraction itself.
Focused TP/Mamba/Host/fault unit regressions additionally passed on node2700
with an allocated H100 and CUDA libraries; this is not end-to-end model serving.
The exact results and execution boundary are recorded in ../tests/reports/README.md.
All eight lifecycle acceptance criteria in
`AGENTIC_PD_DESIGN_INVARIANTS.md` remain binding. Deleting an
apparently obsolete transport branch is a separate change requiring reachability
evidence and lifecycle/fault tests; this migration does not infer obsolescence
from `legacy`/`v2` naming or from a default-off experiment switch.

## Historical experiment replay

The shared BrowseComp matrix implementation is tracked in
`scripts/cluster/matrix_controller.py`. Existing model-specific wrappers can use it
without loading code from a scratch run directory. Frozen DRAM rerun controllers
still intentionally require their immutable launch_config/code_identity and
engine snapshot. Those are experiment data dependencies, not project libraries.
Keep them immutable when comparing with published results.
