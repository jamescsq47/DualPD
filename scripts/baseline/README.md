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

## Model-specific entries

Qwen3.5-9B / 27B / 35B-A3B / 122B-A10B, Qwen3.8-27B and MiniMax-M2.7
launch configurations are retained here; see the
[model/topology table](../new-method/README.md#model-specific-swe-experiments-allocated-slurm-nodes).
For the Slurm entries, choose one baseline method before the action:

```bash
bash scripts/baseline/run_minimax_m27_tp8ep8_1p1d_swe.sh hicache launch \
  --job-id JOB_ID \
  --python /path/to/clean-baseline-env/bin/python \
  --sglang-root /path/to/clean-sglang-baseline \
  --workload-python /path/to/dualpd-env/bin/python \
  --model /path/to/models/MiniMax-M2.7 \
  --data-root /path/to/datasets \
  --swe-sif-dir /path/to/swebench-verified-sif \
  --workload-job-id JOB_ID --workload-node PREFILL_NODE \
  --storage-job-id JOB_ID --storage-node DECODE_NODE \
  --run-dir /path/to/results/minimax-hicache-c128 \
  --concurrency 128
```

Use `no_reverse` or `colocated` in place of `hicache`. Add
`--dry-run` to inspect the commands without launching. Baseline node placement
comes from the allocation. The same Slurm job must contain all nodes unless
the backend's explicit extra-allocation options are used.
For the strict two-node MiniMax/122B recipes, the workload is on P and the
Mooncake store on D. The mixed two-node 27B campaign instead used the workload
on the second node and the store on the first. Set both placements explicitly;
the wrapper refuses to launch two-node HiCache without them.

The historical recipes used different clean baseline revisions. The 27B/35B
No-reverse and Colocated runs require the existing GDN-contiguous and Mamba-slot
correctness fixes. Do not substitute the custom-method engine or compare an
uncorrected baseline. Legacy hybrid HiCache revisions may require the original
compatibility layer; the wrapper accepts `--mooncake-compat /path/to/layer`
and `--mooncake-master-binary /path/to/mooncake_master` but does not silently
patch or modify the baseline. Match the engine's scheduler/radix flag spelling
using `--mamba-cache-strategy-flag`.

Local Qwen3.5-27B Docker/BrowseComp baseline scripts are also restored, with
input/environment paths supplied by the caller. They retain their historical
configurations; notably the Docker Colocated entry uses a fenced-shell profile
and is not automatically equivalent to the structured-tool PD entry.
