# Native baselines

Use a separate clean baseline Python/engine, never the modified submodule.
The environment gate rejects custom agentic modules before GPU startup.

`run_qwen3_8b_browsecomp.sh no_reverse` selects native PD without reverse reuse;
`run_qwen3_8b_browsecomp.sh hicache` selects native HiCache + Mooncake with
Decode offload. Set BASELINE_ENV_BIN, MODEL_PATH, PD_DATA_ROOT and RUN_DIR.
The editable concurrency default is c512. Full commands are in the root README.

`run_pd_case.sh` and `run_colocated_case.sh` are reusable launch components.
Cluster baseline commands are in `../cluster/qwen35_native_pd_slurm.py` and
`qwen35_colocated_slurm.py`. Keep sampling, model limits and physical-GPU
memory fractions aligned with the main method. No custom method policy is
added to the native baseline engine.
