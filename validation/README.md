# Standalone extraction verification

Validated in the new isolated `DualPD/.venv`; source worktrees and their Python
environments were retained. Runtime SGLang version is 0.5.10.post1.

| Check | Result | Evidence |
|---|---|---|
| Standalone frontend + launch regression | 295 passed + 2 passed subtests (297 JUnit records) | frontend.xml, frontend.log |
| Engine CPU lifecycle/control/lease regression | 248 passed | engine.xml, engine.log |
| H100-host TP/Mamba/Host/fault regression | 415 passed | h100-regression.xml |
| Engine original/relocated source verification | 4420 files byte-identical, 73 relocated modules/tests hash-identical | ../scripts/verify_snapshot.py, ../../sglang_pd/dualpd-source-map.json |
| Frontend serving algorithms | 12 core files AST-identical apart from imports | ../scripts/verify_snapshot.py |
| Launch-plan comparison | 9 configurations identical except root paths and path-dependent config hash | ../scripts/verify_launch_plans.py |
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

There is **no new full BrowseComp or SWE end-to-end performance/accuracy
acceptance**. Previous T450/throughput/accuracy tables are retained historical
measurements, not new measurements of these directories. See ../docs/EXTRACTION.md
for inherited optional/archival test limitations discovered by broad test scans.
