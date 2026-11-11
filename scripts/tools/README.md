# Input preparation and result analysis

- `prepare_workloads.py browsecomp`: convert the upstream question Parquet.
- `prepare_swe_bench.py`: export official SWE-bench Verified tasks to JSONL.
- `prefetch_swe_bench_images.py`: prefetch official per-task Docker images.
- `check_environments.py`: fail closed if modified/baseline engines are mixed.
- `analyze_pd_offload.py`: final-run PD/Host observations.
- `analyze_swe_bench_run.py`: task completion, T450/T500 and verifier results.
- `analyze_agentic_kv_v1.py`: request-generation lifecycle observations.
- `load_experiment_profile.py`: checked optional launcher profiles.
- `select_rate.py`: summarize a caller-configured request-rate sweep.

Use each command's `--help`. Inputs and outputs are external run directories;
raw logs, result tables, development probes and test fixtures are not shipped.
