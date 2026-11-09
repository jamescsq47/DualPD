# Local request-generation KV launchers

This directory contains reusable topology entries and retained historical
recipes. Router, allocator, transport and lifecycle implementations belong to
the sibling `sglang_pd` engine.

## Maintained execution chain

The current staged regression enters through
`scripts/cluster/engine_cleanup_quickcheck.py`. Its recipe and measurement
helpers are in `scripts/benchmarks/`; see that directory's README. It explicitly
selects Qwen3-8B TP1 4P:4D/c256, 680 source-order samples, seed2026,
temperature0, tool/admission1s, total D→P512/P→D256GiB Host capacity and
300s business warmup+1200s measurement.

Reusable execution components remain:

- `run_1p3d_case.sh`, `run_1p5d_case.sh`, `run_2p6d_numa_case.sh`,
  `run_4p4d_numa_case.sh` and the TP2 topology wrappers construct case arguments.
- `internal/run_agentic_pipeline.sh` and `run_pd_servers.sh` supervise model/
  workload services and dependency-ordered cleanup.
- `internal/agentic_case_environment.sh` builds16 common case argv entries.
- `internal/apply_experiment_profile.sh` preserves explicit caller overrides
  and fails before startup on trusted profile-loader errors.
- `internal/launch_late_binding_router.py` is a thin engine launch bridge.

Case-layer historical fallback thresholds remain2s/2s; the maintained
regression explicitly selects1s/1s. Only4P4D applies the optional profile at this
case layer. Fingerprint, allocation, NUMA and resource gates remain mandatory.

## Host and transfer configuration

Host payloads use ordinary CPU DRAM through memfd; small control metadata uses
run-scoped `/dev/shm` directories. The current regression configures128GiB D→P
and64GiB P→D per physical P rank, four ranks total. Both directions retain
independent capacity/ownership. Slow Host landing and later Prefill placement
are independent; an arena's P domain does not bind all recovery computation.

The current recipe explicitly selects registered final-extent DMA,8GiB windows,
a768GiB per-process registered-cache bound, startup prewarm/barrier, two lanes
and4096-token chunks. The cache bound is not an extra physical arena allocation.
Before-DMA unsupported registration/batch operations retain the bounce fallback.
Actual ablations and Direct/Host/fence/workset/TP semantics remain governed by
`docs/AGENTIC_PD_DESIGN_INVARIANTS.md`, not redefined here.

## Retained historical recipes

Model/workload-named, fused/checkpoint and pinned-Mamba launchers/support remain
for replay. In particular, `run_qwen3_8b_tp1_browsecomp_4p4d.sh` defaults to
c384,1200s warmup,2s/2s and different Host/cache values. It is not the current
regression entry. The existing profile YAML likewise records an earlier trial,
not a substitute for the maintained recipe.

Preserve recorded parameters and pinned inputs. Historical dependency closure
and archival remain pending stage7 work; these retained recipes do not imply new
acceptance of every historical model/TP/default configuration.
