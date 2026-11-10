# Code-cleanup handoff

The standalone organization and sequential BrowseComp cleanup are implemented.
This is not a rerun of all historical jobs or repair of every inherited algorithm
limitation. Original Slime/SGLang worktrees remain comparison sources; this work
does not modify shared `pd`/baseline environments or merge their selections.

## Responsibilities

| Location | Responsibility |
|---|---|
|DualPD root|`inference.py` plus four package/install entries; no `examples/pd` wrapper|
|`runtime/`, `data/`, `configs/`|Application requests/tool/final ACKs, harnesses/tools/verifiers, configuration|
|`scripts/cluster/`, `scripts/new_method/`, `scripts/benchmarks/`|Maintained deployment, thin engine entries, fixed regression|
|`scripts/replay/`|Historical orchestration, source identities and operational warnings|
|Engine `disaggregation/dualpd/`|Ledger/protocol, DMA/registration/buffers/writer, authority and worker lifecycle|
|Engine Router/Scheduler components|Separate local/TCP routing policies, workset broker, Direct/TP/ready logic and explicit native hooks|

Public imports remain `sglang.srt.disaggregation.agentic_*`. Historical aliases,
serialization identities, selected overlays and actual ablations/defaults remain.
Source proof preserves194 environment identifiers (not194 boolean switches).
Metadata-only control code no longer imports DMA/workers just for protocol types.

## Verification

Each Host, Router, Scheduler, dependency, configuration, launcher/replay and
package-entry stage has source/contract/fault review and its own full BrowseComp
comparison. [SEQUENTIAL_ENGINE_CLEANUP.md](SEQUENTIAL_ENGINE_CLEANUP.md) indexes
these gates; [ENGINE_CLEANUP_RESULTS.md](ENGINE_CLEANUP_RESULTS.md) compares all15
baseline/stage results against the original pre-refactor standalone baseline.

At runtime-code frontend `fab82e1`, engine `d265f7946aa3d1369ecdadf9d2d39a77c84e300b`,
full CPU passes315 frontend tests+308 subtests and568 engine tests+37 subtests:
1228 JUnit records, no failures/errors/skips,7 non-fatal warnings. Whole-source
verification finds4402 byte-identical engine files and no unexpected differences;
changed files have complete reviewed reconstruction. Replay proves201 classified
files/142 migrations and52 byte-identical retained code files plus2 exact Router
entries+2 exact shared tools. Native1018/CUDA2 are earlier accepted same-engine
gates, not new executions by the docs-only handoff.

Actual isolated double-wheel imports/Router CLI and old-overlay/pin contracts
cover installation/entry selection. Dense TCP/TP2 r7 covers Direct/D2P Host;
r8 covers P2D Host. Hybrid r9 covers the original stable-prefix Direct/D2P Host
mode with exact child/reference outputs and scoped physical release chains.
See [FINAL_ENGINE_VALIDATION_PLAN.md](FINAL_ENGINE_VALIDATION_PLAN.md).

The final `shared-tool-paths-r1` retains Qwen3-8B TP1 4P:4D/c256, application
seed2026/temp0, original inputs/limits, tool/admit1s and total D2P512/P2D256 GiB.
It completes300.725907s warmup+1200.001417s measurement: Decode11239.01 token/s
versus original11265.01 (−0.2308%), Prefill75288.22, parent reuse100%, failures/
OOM0. Individual regression runs do not establish statistical equivalence.

Independent full archive scan:8260 unique stable rows;6687 measured protocol
successes=2360 completed+4327 truncated, not6687 correct/naturally completed tasks.
Eight-log audit: P2D Host five sets11567 equal; D2P Direct21473 sent=received=
mapped D release; Host134 durable=D release. Seven measured Host snapshots
restore/release;127 become durable only during post-window drain and have no
individual restore evidence. Event sets do not prove latency/instantaneous
ownership or complete local P2D Direct source-release chains.

Fresh independent/main SSH confirm controller1148718/launcher1148742 absent,
GPU/ports/Apptainer/FUSE empty, six memory events0 and original keeper retained.
Bounded post-window termination is not OOM or natural child exit0. The seven-day
allocation28285059/node2700 remains retained; recheck Slurm and obtain a fresh
exact-plan audit for another GPU validation.

Artifacts: `/orcd/pool/004/shangy/dualpd/runs/engine-cleanup-validation-20261108/shared-tool-paths-r1/`
contains plan/result, request-integrity, ownership-summary, acceptance-audit and
logs. Result SHA256:
`f9c7ee874bfbace806a2cba03d8fc57595bdbb3307fa2bad02d72f495f2fab77`.

## Explicit limits

- Preserving SWE source/harness/config contracts is not a new T450/accuracy
  measurement. No new measured SWE accuracy equivalence is claimed.
- Historical source/import/argv contracts do not mean all old Slurm/Docker jobs
  were executed. Fixed PIDs/jobs/credentials remain inert historical identities.
- Hybrid r9 deliberately returns256/321 tokens under the old frozen prompt
  checkpoint, recomputing the generated suffix; it is not a full-parent
  page-tail-only pass. [Details](HYBRID_RUNTIME_VALIDATION.md). Changing this is
  separate algorithm/lifecycle work, not a silent cleanup fix.
- Hybrid P2D Host, stress/multi-host, every model/TP/switch combination and global
  shared-Radix/recurrent-allocator quiescence are not proved. Optional-Relay
  failure concerns in the sequential notes are retained, not claimed fixed.
- Wheel checks prove import/CLI/Router selection, not source-free Slurm or Git
  reconstruction tools shipped in wheels. Models/data, images, site paths and
  signature policy remain explicit inputs; no permissive policy is supplied.

These limitations accompany stronger future claims, without expanding this
organization task into new algorithm development or an all-history rerun.
