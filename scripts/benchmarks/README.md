# Maintained benchmark components

These modules support the current staged BrowseComp regression. They do not
import historical matrix/autopilot controllers or start services at import.

- `browsecomp_qwen3_8b_h100.py` constructs the fixed accepted Qwen3-8B TP1,
  4P:4D/c256 recipe: 680 source-order samples, seed2026, temperature0,
  context40960/output36864, tool/admission1s, 300s business warmup+1200s
  measurement, D→P512/P→D256GiB total arenas and congestion32/8. It preserves
  every recorded environment key, even unused historical diagnostic keys.
  Cached deployment model/data paths and the original analysis Python remain
  explicit; this is not a generic portable model profile.
- `measurement.py` contains the unchanged GPU/port/cgroup observations and
  full-window/tail/OOM summary contracts. `ensure_analysis` receives its Python
  and project root explicitly. The controller preserves the original fixed
  analysis Python; it does not silently replace it with its isolated Python.

The maintained entry remains `scripts/cluster/engine_cleanup_quickcheck.py`.
It loads only these two allowed files from the selected `DUALPD_ROOT`, checks
physical controller/component identity, and retains caller-selected path
spelling in launch values. It supports direct isolated-Python invocation from
outside the checkout without relying on an installed `dualpd` package. No
engine namespace or Python search path is prepended by this loader.

`scripts/verify_benchmark_components.py` proves the complete three runtime
modules against immutable frontend152a620, plus unchanged historical matrix
sources. Complete environment/argv A/B, synthetic failure/cleanup fixtures,
negative source mutations and actual accepted-log parity are separate checks.
They do not substitute for native/actual GPU or full300+1200 acceptance.

Historical launchers have not been archived by this extraction. Their replay
closure, pinned inputs, path identities and ablation switches must be preserved
in the next part of stage7; actualTCP/TP2/hybrid and final packaging remain open.
