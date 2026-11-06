# DualPD

Standalone agentic PD serving, workload harnesses, experiment orchestration and
resource analysis. The engine is the sibling `../sglang_pd` repository.
Slime is no longer imported by the inference runtime.

## Install and launch

Use the existing compatible SGLang environment to create an isolated environment:

```bash
/path/to/pd/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip install setuptools-scm
.venv/bin/python -m pip install --no-deps --no-build-isolation -e ../sglang_pd/python
source scripts/env.sh
.venv/bin/python -m dualpd doctor
.venv/bin/python -m dualpd inference --help
.venv/bin/python -m dualpd slurm --help
```

For a clean environment, install this project's dependencies and the matching
engine fork. SWE dependencies are optional (`pip install -e '.[swe]'`);
BrowseComp retrieval requires its existing embedding service and cached corpus.
No model, dataset or container image is bundled.

The supported cluster workflow is an **editable source installation**. The
inference and router entrypoints are packaged; the cluster commands additionally
need the checkout's `tools/dualpd` scripts. Models, dataset roots, Slurm allocation,
GPU placement, containers and baseline Python environments remain explicit inputs.
`scripts/env.sh` selects the modified engine: source it only for that method.
Baseline launchers use their separately configured clean baseline environments.

## Layout

| Directory | Purpose |
|---|---|
| `src/dualpd/runtime` | Inference sample, tokenizer/sampling state, dataset loading, HTTP and trace |
| `src/dualpd/data` | BrowseComp, SWE-bench, Retool and other workload adapters |
| `src/dualpd/inference.py` | Open/closed-loop load generator and metric recording |
| `src/dualpd/late_binding_router.py` | Local shared-arena router |
| `tools/dualpd` | TCP router, launch/control tools, matrices, watchdogs and analysis |
| `configs` | Existing workload schedules and serving profiles |
| `results` | Existing result documents and retained small artifacts |
| `examples/pd` | Compatibility symlink to `src/dualpd` for existing launch commands |

The former README and all experiment instructions are retained in
`src/dualpd/README.md`; historical paths and defaults in that document are
historical evidence, not the standalone installation guide.

## Compatibility and validation

See `docs/EXTRACTION.md` for the scope, source identities, supported engine paths,
and the distinction between CPU regression checks and GPU performance validation.
The migration preserves harness prompts, sampling settings, request-generation
metadata, lifecycle notifications, warmup/measurement windows and metric schemas.
Historical experiment controllers that replay a frozen run still require that
run's immutable artifacts. The shared BrowseComp matrix controller is now tracked
as `tools/dualpd/matrix_controller.py` rather than imported from scratch.

The old `slime/` and `sglang-src/pd/` directories remain available for comparison.
Do not combine their Python modules with this checkout in the same process.
