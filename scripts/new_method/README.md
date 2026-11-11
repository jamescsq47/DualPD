# DualPD main experiments

`run_qwen3_8b.sh browsecomp` and `run_qwen3_8b.sh swe` are the public
single-node entries; see the root README for input preparation and commands.
Editable concurrency defaults are c512 for BrowseComp and c64 for SWE.
Explicit environment values override defaults.

Generic `run_*case.sh` files construct P:D topology. TP2 wrappers select logical
GPU groups. For those lower-level entries, set MODEL_PATH, PD_ENV_BIN and an
appropriate WORKLOAD_CONFIG explicitly. The `internal/` directory supervises
workers, search and Router processes and shares the same cleanup path.
It is not a second Router implementation.

D→P and P→D capacities are independent and specified per Prefill rank. The
public 4P:4D entry uses 128/64 GiB per P rank (512/256 GiB node totals).
Native HiCache/Mooncake storage and native Decode offload remain baseline-only.
