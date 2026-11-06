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

Historical `examples/pd` paths resolve within the new checkout through a
compatibility symlink. Flat harness imports are remapped to `dualpd.*`; no
`slime` import alias or bundled Slime package is installed. The standalone JSON
name `dualpd_root` is accepted alongside the old `slime_root` key, and a conflict
is rejected. Existing experiment inputs and values remain explicit.

## Regression gate

```bash
.venv/bin/python scripts/check_cpu.py
.venv/bin/python scripts/verify_snapshot.py
.venv/bin/python scripts/verify_launch_plans.py
```

The CPU gate covers request metadata, router reservations and retries, workload
sampling, SWE tool loops, HTTP transport, multi-node plans, process cleanup, TP
control protocol, cancellation/fault handling, Host transfer and memory leases.
Logs, JUnit XML and machine-readable return codes are saved in `validation/`.
`verify_snapshot.py` is a migration-only reference check requiring the retained
original sibling snapshots. Production imports and the CPU gate do not need them.

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
The exact results and execution boundary are recorded in ../validation/README.md.
All eight lifecycle acceptance criteria in
`../src/dualpd/docs/AGENTIC_PD_DESIGN_INVARIANTS.md` remain binding. Deleting an
apparently obsolete transport branch is a separate change requiring reachability
evidence and lifecycle/fault tests; this migration does not infer obsolescence
from `legacy`/`v2` naming or from a default-off experiment switch.

## Historical experiment replay

The shared BrowseComp matrix implementation is tracked in
`tools/dualpd/matrix_controller.py`. Existing model-specific wrappers can use it
without loading code from a scratch run directory. Frozen DRAM rerun controllers
still intentionally require their immutable launch_config/code_identity and
engine snapshot. Those are experiment data dependencies, not project libraries.
Keep them immutable when comparing with published results.
