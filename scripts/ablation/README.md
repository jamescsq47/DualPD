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
| `direct_only` | Disable D→P Host staging; allow Direct attempts for tool waits and recompute on failed admission |
| `host_only` | Force D→P Host recovery instead of Direct |
| `p2d_direct` | Disable P→D Host staging but retain late binding |
| `p2d_prebind` | Prebind D before Prefill; disable P→D Host staging (TP=1) |
| `p2d_native_prealloc` | Native D preallocation with P/D submission; disable P→D Host staging |
| `random_routing` | Use the retained random-routing ablation |
| `dram_small` | Four P ranks × D→P16/P→D32 GiB = **whole-node D→P64/P→D128 GiB** |

Concurrency and the 300+1200-second window come from the public main entry.
Set other arena sizes or timing thresholds explicitly for sensitivity studies;
sizes remain per P rank. Use a new RUN_DIR and fresh shell for each run so
previous policy overrides cannot contaminate the next case. These entries
expose retained engine switches; cleanup does not constitute new GPU acceptance
of every variant or justify comparing different model/TP settings.
