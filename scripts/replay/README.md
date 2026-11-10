# Historical experiment replay

This directory separates recorded experiment orchestration from maintained
launchers. `manifest.json` maps all201 original launcher files:142 historical
files move here and59 maintained/core/shared/compatibility files stay at their
canonical addresses. Original hashes refer to immutable frontend6bc1523.

Current source/contract and full BrowseComp archival acceptance is recorded in
`../../docs/CLEANUP_HANDOFF.md` and `../../docs/SEQUENTIAL_ENGINE_CLEANUP.md`.
The current retained-code proof has52 byte-identical files,2 exact Router-entry
updates and2 exact shared-tool path updates. No changed retained file is waived.
This is not authorization to execute the historical operational recipes below.

- `cluster/`: historical matrix/autopilot/watchdog, model-specific Slurm
  recipes, old V1 smoke and their tests. Keep these siblings together because
  experiments reuse controller module objects across model families.
- `new_method/`: model/experiment-specific local recipes and sequence tests.
  Common topology cases, pipelines and compatibility/checkpoint/Router helpers
  remain in `../new_method/`; their runtime settings are not unified here.

The maintained entrypoints remain `../cluster/{multinode,qwen35_slurm,
qwen35_native_pd_slurm,qwen35_colocated_slurm,process_supervisor}.py` and
`../cluster/engine_cleanup_*`. The fixed BrowseComp regression uses
`../benchmarks/` directly, never these historical controllers.

`cluster/_core.py` loads only seven named canonical site/shared tools. It
preserves bare/public module object identity, checks physical origins, rejects
split identities and does not insert an engine or core directory into sys.path.
Historical siblings use explicit package/direct selection. Package mode requires
the actual selected DualPD package to be importable. Bootstrap must not race
unrelated ordinary imports; this is not a general import-hook service.

Moving a file does not authorize running it. Many entries contain old job IDs,
nodes, IPs, fixed PIDs, run directories and queue destinations. In particular,
the old reload guard can send TERM to a fixed PID. Never execute historical
watchdogs/guards/sbatch to discover their behavior, and never reuse historical
credentials for a new run. Native overlays, frozen snapshots, model/data paths
and old harness selections remain historical identities, not current defaults.

PPD helpers and eight older BrowseComp sbatch recipes now have precise helper
address repairs, recorded as14 additional whole-file reconstruction edit sets.
Five PPD controllers also resolve their summary helper in the maintained tree.
The original Slime harness and external PR source/SIF selections remain distinct
and unchanged: no blanket path substitution. Independent audit accepted these
14 exact address repairs, not execution of historical jobs. Old batch pytest
paths and pinned dependencies may still be unavailable; historical operational
replay is not claimed validated by these static checks.

The following records migration preparation, before its subsequent independent
code and full300+1200 BrowseComp acceptance; its pending wording is historical.
Complete immutable source reconstruction
and the five-shell/28-case full environment+argv A/B pass. Seven more wrappers
pass22 comparisons at the common-case boundary, not the final pipeline; only
WORKLOAD_CONFIG/RUN_DIR path spellings are resolved there. The six complex
entrypoints now pass15 byte-exact configuration-prefix comparisons before the
first mkdir, with topology/lane/plugin-existence rejection checks. Colocated's
real EXIT cleanup runs with an asserted empty service registry and a forbidden
kill sentinel, not real service teardown. The27B plugin path is fixture-local
in disposable copies and its empty sentinel proves existence handling only,
not native-plugin validity. Docker, model startup, source snapshots and complex
operational suffixes are not executed. These explicitly different coverage
depths must not be described as18 full historical experiment replays.
Fingerprint checks in the first A/B are stubbed; its maintained support bytes
are asserted equal to immutable sources. Independent scoped test audit accepts
the three contract suites, not overall GPU or historical runtime acceptance.
See `../../docs/SEQUENTIAL_ENGINE_CLEANUP.md`. This directory is not a claim of
actual TCP/TP2/hybrid model coverage or permission to delete retained switches.
