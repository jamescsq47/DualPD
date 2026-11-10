# DualPD

`DualPD/` itself is the former `slime/examples/pd/` directory. Entry scripts,
workloads and configs live directly here. There is no `src/dualpd/` wrapper or
`examples/pd/` compatibility directory.

| Location | Contents |
|---|---|
| Root Python files | `inference.py` and Python package/install entrypoints only |
| `data/` | Workload loaders, agent loops, tools and verifiers |
| `configs/` | Workload, serving and experiment configuration |
| `scripts/` | Launchers, analysis and automation; cluster tools are in `scripts/cluster/` |
| `runtime/` | Client KV protocol, metrics, HTTP, CLI, sample/state and trace utilities |
| `tests/` | Regression tests; reports are in `tests/reports/` |
| `docs/` | Design and usage documentation |
| `runs-host/` | Existing experiment result documents |
| `patches/` | Dependency/compatibility patches retained for reproducibility |

From this directory:

```bash
.venv/bin/python -m pip install --no-deps -e .
source scripts/env.sh
.venv/bin/python inference.py --help
.venv/bin/python -m dualpd doctor
.venv/bin/python -m dualpd slurm --help
.venv/bin/python scripts/check_cpu.py
```

The sibling `../sglang_pd` is the modified engine. Install it in the isolated
environment with `pip install --no-deps --no-build-isolation -e ../sglang_pd/python`
(the build needs setuptools-scm). Baseline experiments retain their explicitly
configured baseline Python and engine. Models, datasets, retrieval caches and
container images remain external inputs.

Routing implementations belong to `sglang_pd`, under
`python/sglang/srt/disaggregation/dualpd/router/`: `late_binding_router.py`
for the local path and `global_pd_router.py` for the TCP path. DualPD only
selects configuration and launches them;
`scripts/new_method/internal/launch_late_binding_router.py` is a thin entrypoint,
not a second implementation. `runtime/agentic_kv_request.py` remains on the
application side because it builds requests and publishes tool/final ACKs.

Harness dispatch is in `data/generation.py`; workload selection remains in
`data/dispatch.py`. BrowseComp and Retool use their real modules under
`data/browsecomp/` and `data/retool/`, not root-level compatibility aliases.
Run the local search service with
`python -m dualpd.data.browsecomp.search_server --help`.

See `docs/EXTRACTION.md` for migration scope and `tests/reports/README.md` for
validation evidence. The original source worktrees remain available for comparison.

The organization and verification handoff is
[CLEANUP_HANDOFF.md](docs/CLEANUP_HANDOFF.md). The staged before/after results are in
[ENGINE_CLEANUP_RESULTS.md](docs/ENGINE_CLEANUP_RESULTS.md). Actual TCP/TP2
coverage and remaining gates are tracked in
[FINAL_ENGINE_VALIDATION_PLAN.md](docs/FINAL_ENGINE_VALIDATION_PLAN.md);
[HYBRID_RUNTIME_VALIDATION.md](docs/HYBRID_RUNTIME_VALIDATION.md) distinguishes
the preserved stable-prompt Mamba compatibility mode from full-parent reuse.
