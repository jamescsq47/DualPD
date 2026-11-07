# DualPD

`DualPD/` itself is the former `slime/examples/pd/` directory. Entry scripts,
workloads and configs live directly here. There is no `src/dualpd/` wrapper or
`examples/pd/` compatibility directory.

| Location | Contents |
|---|---|
| Root Python files | Main inference, routing and request protocol interfaces |
| `data/` | Workload loaders, agent loops, tools and verifiers |
| `configs/` | Workload, serving and experiment configuration |
| `scripts/` | Launchers, analysis and automation; cluster tools are in `scripts/cluster/` |
| `runtime/` | Standalone sample, generation state, HTTP and trace utilities |
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

See `docs/EXTRACTION.md` for migration scope and `tests/reports/README.md` for
validation evidence. The original source worktrees remain available for comparison.
