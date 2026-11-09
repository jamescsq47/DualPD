# Standalone extraction verification

Validated in the new isolated `DualPD/.venv`; source worktrees and their Python
environments were retained. Runtime SGLang version is 0.5.10.post1.

Current CPU gate: **613 passing JUnit records** (208 frontend, 403 engine and
two engine subtests). The 25 new frontend tests cover the reusable allocation
regression runner: exact audited source/config, untracked source detection,
single per-job lock, real nested-shell TERM cleanup and already-exited leader
draining, complete JSONL polling, resource gates and throughput comparison.
`engine-cleanup-runner.xml` is a separate overlapping rerun, not 25 extra records.
Independent exact-plan audit GO applies to baseline-r3; the DMA extraction also
has a code audit GO, pending its own exact GPU plan audit. Job28285059 is RUNNING
on node2700. Baseline-r3 completed300+1200 seconds with Decode11265.01 token/s,
Prefill75367.90 token/s, zero failures and a clean resource gate. DMA performance
has not yet been measured.
See ../../runs-host/ENGINE_CLEANUP_QUICKCHECK_H100.md for live-run artifact paths.

| Check | Result | Evidence |
|---|---|---|
| Standalone frontend + launch regression | 208 passed | frontend.xml, frontend.log |
| Engine CPU lifecycle/control/lease/Router regression | 403 passed + 2 passed subtests (405 JUnit records) | engine.xml, engine.log |
| H100-host TP/Mamba/Host/fault regression | 415 passed | h100-regression.xml |
| Engine original/relocated source verification | 4411 files byte-identical, 64 mapped files hash-identical, 9 reviewed AST transformations, 3 Router implementations hash-identical | ../../scripts/verify_snapshot.py, ../../../sglang_pd/tools/verify_dualpd_cleanup.py, ../../../sglang_pd/dualpd-source-map.json, ../../../sglang_pd/router-source-map.json |
| Frontend serving algorithms | 12 core files AST-identical apart from imports and two flattened documentation/diagnostic paths | ../../scripts/verify_snapshot.py |
| Launch-plan comparison | 9 configurations identical except root paths, Router module entry, path-dependent config hash and removal of one no-op helper flag | ../../scripts/verify_launch_plans.py |
| Standalone package build | Wheel built; runtime configs, scripts and design document present | pip wheel and wheel member inspection |

The H100-host regression used Slurm job **28109209 on node2700**, with **1 H100,
8 CPUs and 64 GiB**. It completed successfully in approximately 11 seconds of
pytest execution. It tested TP coordination and hybrid lifecycle via focused
fixtures; this is not a TP>1 model-serving or throughput measurement. No model
servers, search services, task containers or persistent watchdogs were launched.
The job exited and its allocation was released.

The 9 launch-plan configurations cover Qwen3 TP1/2/4/8, Qwen3.5 hybrid TP1/2/4/8,
and MiniMax TP8. They verify generated commands/environment values rather than
claiming all those model configurations were run. Actual BrowseComp low-concurrency
input validation also passed; its new-method plan retains thresholds 1/1 seconds,
300 s warmup, 1200 s measurement and selects the new `sglang_pd/python`.

Independent static/state-machine audit returned **GO**. It independently checked
source equivalence, module identity, standalone imports from /tmp, Sample
serialization and the type-only HiCache import cleanup.

There is **one accepted standalone BrowseComp baseline**, not acceptance of
the subsequent DMA extraction or new SWE accuracy results. Previous
T450/throughput/accuracy tables are retained historical
measurements, not new measurements of these directories. See ../../docs/EXTRACTION.md
for inherited optional/archival test limitations discovered by broad test scans.

The project was subsequently flattened so DualPD itself is the interface root.
Frontend and engine CPU gates were rerun after that layout change. The H100 XML
above records the preceding extraction validation; it is not a new model run.

## Router ownership follow-up

Local/TCP Router implementations, service startup logic and Router unit tests
were moved to the engine. Frontend counts decrease because these tests now run
in the engine group; no Router coverage was removed. That migration's CPU gate had
556 passing JUnit records. It includes four engine-only import/isolation checks
and two historical-overlay launch checks. Four additional 30B-A3B ratio/launcher
regressions and both changed shell launchers' `bash -n` checks also passed.

The independent follow-up audit returned **GO** after identifying and fixing a
historical-overlay module-location issue. Both current Router implementations
and the pinned legacy Mamba implementation remain byte-identical to their
source commits; launch wrappers preserve the selected SGLang overlay. No GPU
experiment or end-to-end performance measurement was run for this follow-up.

## Root-module cleanup follow-up

The root now contains five Python files: `inference.py` and four package/install
entry files. Client KV protocol, metrics, HTTP transport and CLI live under
`runtime/`; mixed-harness generation lives in `data/generation.py`. The existing
workload sampler in `data/dispatch.py` is unchanged. Five obsolete root alias
files were removed; all active callers use the real workload modules instead.

The CPU gate now has 566 passing JUnit records (183 frontend, 381 engine and two
subtests). New checks cover root layout, relocated imports, stale Python/Shell
references, five outside-checkout `--help` entrypoints and the refreshed console
script. All 87 Shell launchers pass `bash -n`; all nine model/TP launch plans
still compare equal under the documented path/entry remapping. Engine/source
equivalence passes with no unexpected differences. Independent read-only audit
returned **GO**. The engine checkout was not modified by this follow-up.

The rebuilt wheel contains all eight checked relocated runtime/service paths,
omits all 11 removed root modules, and declares `dualpd.runtime.cli:main` as its
console entry. Build artifacts and the wheel are retained under
`/tmp/dualpd-wheel-check.hcu3pq/`, outside the project root.

Only `DualPD/.venv` was reinstalled to refresh its console script. CPU tests were
run outside the filesystem sandbox because FlashInfer imports write a user
cache log and transport tests use localhost sockets. This does not initialize
GPU model services or constitute a new 300+1200/T450 measurement.

## First internal engine cleanup

The current supported CPU gate has 577 passing JUnit records (183 frontend,
392 engine, two subtests). It includes the new ledger/alias/pickle tests and
memory-scheduler-bridge tests. An additional fresh multi-node launcher check
passed all 22 tests after the no-op flag was removed.

The Host staging implementation decreased from 10131 to 7676 lines by moving
the identical ledger state/CAS code to its own 2485-line, CUDA-independent
module. The 41 existing engine tests were relocated, not deleted. Only an
assertion-only test of the unused helper flag was removed. The AST gate proves
the nine ledger nodes and remaining DMA/worker behavior unchanged; literal
environment identifier inventory retains all 143 remaining identifiers.

Extended CPU discovery recorded **506 passes and 9 failures**, all nine due to
`libcuda.so.1` loading inside two application-termination test functions.
`engine-extended-discovery.xml` retains those failures. Selecting the supported
CPU cases gives **506 passes, 9 deselected** in `engine-extended-cpu.xml`; these
overlap the main gate and must not be added as unique coverage. The command
omits 13 modules whose collection requires the CUDA driver and the token-content
suite whose helper import has the same dependency. Full collection and the
separate frontend Host suite also hit that driver dependency; the latter is
recorded in `host-cleanup-cpu.xml`. None of these driver-dependent checks is
claimed as a pass or silently weakened to obtain a green result.

Independent static audit returned **GO**, including all eight ownership
invariants. No GPU tests or benchmark measurements were run in this cleanup.
See ../../../sglang_pd/docs/DUALPD_CLEANUP.md for retained switches, exact
removed helpers, import/pickle compatibility boundaries and remaining work.
