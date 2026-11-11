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
