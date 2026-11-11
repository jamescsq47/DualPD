# Native baselines

Use a separate clean baseline Python/engine, never the modified submodule.
The environment gate rejects custom agentic modules before GPU startup.

`run_qwen3_8b_browsecomp.sh no_reverse` selects native PD without reverse reuse;
`run_qwen3_8b_browsecomp.sh hicache` selects native HiCache + Mooncake with
Decode offload. Set BASELINE_ENV_BIN, MODEL_PATH, PD_DATA_ROOT and RUN_DIR.
The editable concurrency default is c512. Full commands are in the root README.

`run_swe_no_reverse.sh`, `run_swe_hicache.sh` and `run_swe_colocated.sh` expose
the allocated-node SWE baseline frontends. Use `--help` for model/TP, topology,
baseline Python/engine and job options. Their common implementation lives in
`runtime/launch/`, not this directory. Keep sampling, model limits and physical-GPU
memory fractions aligned with the main method. No custom method policy is
added to the native baseline engine.
