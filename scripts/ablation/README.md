# BrowseComp ablations

Use the same installation and external inputs as the main method:

```bash
MODEL_PATH=/path/to/models/Qwen3-8B \
PD_DATA_ROOT=/path/to/datasets \
RUN_DIR=/path/to/results/no-d2p \
  bash scripts/ablation/run_browsecomp.sh no_d2p
```

| Variant | Change from the public main-method entry |
| --- | --- |
| `full` | No ablation; uses the main entry's existing policy defaults |
| `no_d2p` | Recompute later turns; retain the custom P→D path |
| `fast_failure_recompute` | Recompute when Direct admission fails rather than retrying via Host |
| `d2p_direct_bounded` | Disable D→P Host while retaining the 1s tool cutoff |
| `direct_only` | Disable D→P Host staging; allow Direct attempts for tool waits and recompute on failed admission |
| `host_only` | Force D→P Host recovery instead of Direct |
| `p2d_direct` | Disable P→D Host staging but retain late binding |
| `p2d_prebind` | Prebind D before Prefill; disable P→D Host staging (TP=1) |
| `p2d_native_prealloc` | D physical preallocation before P, settle reverse placement first; no P→D Host; transfer ingress limit 0 |
| `p2d_paired_credit` | Native preallocation plus capacity-backed P credit; no P→D Host |
| `random_routing` | Use the retained random-routing ablation |
| `dram_small` | Four P ranks × D→P16/P→D32 GiB = **whole-node D→P64/P→D128 GiB** |

Concurrency and the 300+1200-second window come from the public main entry.
Set other arena sizes or timing thresholds explicitly for sensitivity studies;
sizes remain per P rank. Use a new RUN_DIR and fresh shell for each run so
previous policy overrides cannot contaminate the next case. These entries
expose retained engine switches; cleanup does not constitute new GPU acceptance
of every variant or justify comparing different model/TP settings.

## Dedicated experiment entries

These reuse the public main/baseline launchers, not the old node-specific
controllers, frozen run paths or watchdogs. No benchmark results are bundled.

| Entry | Purpose | Default concurrency |
| --- | --- | ---: |
| `run_qwen3_8b_browsecomp_paths.sh` | D→P Direct/Host/recompute and random-routing controls | 512 |
| `run_qwen3_8b_browsecomp_p2d.sh` | P→D Host-off, prebind, native-preallocation and paired-credit comparisons | 576 |
| `run_qwen3_8b_browsecomp_dram.sh` | Host capacity study, including native HiCache controls and resource curves | 512 (fixed) |

For example, with the same MODEL_PATH and PD_DATA_ROOT as above:

```bash
RUN_DIR=/path/to/results/host-only \
  bash scripts/ablation/run_qwen3_8b_browsecomp_paths.sh host_only

MAX_INFLIGHT=512 RUN_DIR=/path/to/results/p2d-host-off \
  bash scripts/ablation/run_qwen3_8b_browsecomp_p2d.sh p2d_direct

RUN_DIR=/path/to/results/native-preallocation \
  bash scripts/ablation/run_qwen3_8b_browsecomp_p2d.sh p2d_native_prealloc
```

All are Qwen3-8B TP1, logical 4P:4D, with the same data order, sampling,
GPU memory fractions and 300s warmup + 1200s measurement as the current main
entry. Path controls start from current public policies and arena defaults,
not an implicit historical checkout. The old path study's P→D128 GiB total
can be selected with `P2D_HOST_ARENA_GIB_PER_P=32`; the current public default
is P→D256 GiB total. Do not compare runs with different budgets as a one-knob
path ablation. Historical hyphenated path names remain accepted by
`run_browsecomp.sh`; `direct_only` (effectively unlimited tool cutoff) and
`d2p_direct_bounded` (1s cutoff) intentionally remain distinct.

The native-preallocation entry includes the retained route-settlement setting
and unbounded transfer ingress (`PD_MAX_TRANSFER_INFLIGHT=0`); paired-credit
adds P-side capacity credit. These are multi-setting comparison recipes, not
an assertion that changing one flag gives the same semantics. GPU/container
lifecycle remains owned by the existing case launcher. No automatic retry or
partial-window acceptance is introduced.

## DRAM capacity study and curves

```bash
RUN_DIR=/path/to/results/dram-new-small \
  bash scripts/ablation/run_qwen3_8b_browsecomp_dram.sh new_small

BASELINE_ENV_BIN=/path/to/clean-baseline-env/bin \
RUN_DIR=/path/to/results/dram-hicache-reduced \
  bash scripts/ablation/run_qwen3_8b_browsecomp_dram.sh hicache_reduced
```

| Variant | D→P total / P→D total (GiB) | Per-P D→P / P→D (GiB) | Native P / D Host size per rank (GB) |
| --- | --- | --- | --- |
| `new_full` | 512 / 256 | 128 / 64 | — |
| `new_half` | 256 / 128 | 64 / 32 | — |
| `new_small` | **64 / 128** | **16 / 32** | — |
| `hicache_full` | — | — | 128 / 64 |
| `hicache_reduced` | — | — | 96 / **52** |
| `hicache_p64d64` | — | — | 64 / 64 |

All three HiCache controls retain the baseline's Mooncake segment argument
`"256 GB"`; new-method arenas use GiB. The reduced D size is 52 GB, the
accepted final historical setting, not the initial invalid 48/51 GB proposal.
Native Host capacity assertions are unchanged. The accidental per-P64/128
experiment is not the `new_small` entry: `new_small` is the requested
**whole-node** D→P64/P→D128 GiB experiment.

DRAM entries retain the recorded congestion-recompute policy (32/8 thresholds),
1s timing and 4P:4D layout. Only capacities vary among the three new-method
cases. Use fresh shells/run directories so unrelated caller overrides cannot
change the serving recipe. The one-case observation runner refuses to append
to a populated directory and propagates launch failures; it does not skip or
shorten measurement windows.

The existing opt-in pool observer and sampler now write:

- `host_resources_2s.csv`: node/cgroup memory and pool-byte curves;
- `host_pools_2s.csv`: separate D→P/P→D or P/D Host pool counters;
- `process_memory_60s.jsonl`: actual PSS scans (not 2s PSS measurements);
- `raw_prometheus_2s.jsonl.gz` and `dram-pools/`: raw measurement evidence;
- `host_resource_summary.json`: steady-window means, sampled peaks and coverage.

Summaries require the completed closed-loop boundaries and at least 95%
pool/PSS coverage; missing observation is not silently reported as zero usage.
PSS includes model/runtime allocations and is **not** the Host KV arena usage.
Shared memfd pages must not be summed as per-process RSS. The observer only
reads counters; it changes no engine cache/transfer policy. Reorganized and
legacy engine module paths are both recognized.

Mixed-Math/other unpublished workloads, abandoned probes, bandwidth tests and
historical retry/watchdog state are intentionally not restored. No new SWE
model ablation results or timing-sensitivity measurements are implied by
these reusable launch entries.
