# DualPD main experiments

`run_qwen3_8b.sh browsecomp` and `run_qwen3_8b.sh swe` are the public
single-node entries; see the root README for input preparation and commands.
Editable concurrency defaults are c512 for BrowseComp and c64 for SWE.
Explicit environment values override defaults.

The two Qwen3-32B TP2 entries select the corresponding physical GPU groups.
Set MODEL_PATH, PD_ENV_BIN and WORKLOAD_CONFIG explicitly when using them.
`run_swe.sh` exposes the allocated-node SWE main experiment frontend; use
`--help` for model, TP, topology, data and job options.

Shared worker/search/Router startup and cleanup live in `runtime/launch/`,
not in this public experiment directory. Routing itself belongs to SGLang.

D→P and P→D capacities are independent and specified per Prefill rank. The
public 4P:4D entry uses 128/64 GiB per P rank (512/256 GiB node totals).
Native HiCache/Mooncake storage and native Decode offload remain baseline-only.

## Model-specific SWE experiments (allocated Slurm nodes)

These entries restore the formal launch configurations from commit
`61e3342`, rather than requiring users to reconstruct them from generic CLI
options. P:D below counts logical TP engines, not physical GPUs.

| Script suffix after `run_` | Nodes | TP / EP | P:D | Historical concurrency settings |
| --- | ---: | --- | --- | --- |
| `qwen35_9b_tp1_2p6d_swe.sh` | 1 | 1 / 1 | 2:6 | 128, 192, 256 |
| `qwen38_27b_tp2_1p3d_swe.sh` | 1 | 2 / 1 | 1:3 | 64, 128, 192 |
| `qwen38_27b_tp2_two_node_2p6d_swe.sh` | 2 | 2 / 1 | 2:6 total | 128, 192, 256 |
| `qwen35_35b_a3b_tp2_1p3d_swe.sh` | 1 | 2 / 1 | 1:3 | 64, 128, 192 |
| `qwen35_35b_a3b_tp2_2p2d_swe.sh` | 1 | 2 / 1 | 2:2 | 64, 128, 192, 256 |
| `qwen35_122b_a10b_tp8_1p1d_swe.sh` | 2 | 8 / 1 | 1:1 | 64, 128, 192, 256 |
| `minimax_m27_tp8ep8_1p1d_swe.sh` | 2 | 8 / 8 | 1:1 | 64, 96, 128, 192, 256; later 320, 384, 448 |

Every row has the same named entry under `scripts/baseline/`, selecting
`no_reverse`, `hicache` or `colocated`. Configurations live in
`configs/model-profiles/` with their original source-script provenance.
The concurrency lists document historical settings, not an automatic matrix
or a guarantee that every method completed every listed setting.

For example, after installing DualPD in the frontend environment:

```bash
bash scripts/new-method/run_qwen35_35b_a3b_tp2_1p3d_swe.sh launch \
  --job-id JOB_ID --prefill-node NODE --decode-node NODE \
  --python /path/to/dualpd-env/bin/python \
  --model /path/to/models/Qwen3.5-35B-A3B \
  --data-root /path/to/datasets \
  --swe-sif-dir /path/to/swebench-verified-sif \
  --run-dir /path/to/results/qwen35-35b-1p3d-c128 \
  --concurrency 128
```

Use two distinct allocated nodes for two-node entries. Add `--dry-run` to
print the exact command sequence without Slurm/SSH/GPU access. `launch`
prepares, starts control, waits for control, starts workers, waits for models,
then starts router and workload. It is a single-case entry, not the historical
site-specific watchdog: use `status` and `stop` with the same job/run options.
It does not automatically stop at T450; the workload runs up to 500 tasks.
Later commands need the same explicit concurrency/placement options.

Qwen3.5-9B keeps its PR51 fenced-shell harness configuration; the other rows
use the structured-tool harness, with the original overlay variant for the
35B 2P:2D campaign. MiniMax retains TP8/EP8 and its own reasoning/tool parsers.
The tuned 35B 2P:2D entry retains D2P/P2D lanes 4/3. Its public
`Ttool` default is 1s as requested; the historical campaign used 30s,
which can be selected explicitly with `--fast-tool-seconds 30`.

## Local Docker / BrowseComp recipes

The restored `run_qwen35_9b_tp1_browsecomp_4p4d.sh`,
`run_qwen35_27b_tp2_browsecomp_4p4d.sh` and
`run_qwen35_27b_tp2_swe_openenv_4p4d_full.sh` retain the original local
GPU grouping, model-specific prompt/protocol and sampling settings. Set
`MODEL_PATH`, `PD_DATA_ROOT`, `PD_ENV_BIN` and a fresh `RUN_DIR`;
the default modified engine is `3rdparty/sglang`. BrowseComp retains the
300s + 1200s window. Their small historical Host arenas are not the H100
Slurm recipes' Host budgets. The local public timing defaults are now 1s;
historical 2s timing can be selected explicitly via environment overrides.
Do not equate the historical 27B fenced-shell colocated recipe with its
structured-tool main recipe without aligning the harness first.
