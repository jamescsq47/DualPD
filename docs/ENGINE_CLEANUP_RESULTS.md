# Sequential cleanup: measured comparisons

These rows are read from each case's saved `result.json`, not estimates or
adjacent-revision comparisons. All use the retained node2700, Qwen3-8B TP1,
4P:4D/c256, the fixed BrowseComp regression inputs and at least300s business
warmup followed by1200s measurement. The reference is always `baseline-r3`.
Each case records zero measured failures, zero memory-event deltas and passing
final cleanup; those gates do not imply benchmark answer accuracy.

| Stage / case directory | Prefill token/s | Decode token/s | Decode vs original | Parent reuse |
|---|---:|---:|---:|---:|
|Original `baseline-r3`|75367.90|11265.01|reference|100%|
|`host-dma-r1`|74816.50|11246.55|−0.164%|99.9921%|
|`host-registration-r1`|75281.01|11220.31|−0.397%|100%|
|`host-buffers-r1`|75462.39|11254.37|−0.094%|100%|
|`host-writer-r1`|74889.97|11214.24|−0.451%|100%|
|`router-local-r1`|75201.48|11206.66|−0.518%|100%|
|`router-tcp-r1`|75023.20|11203.87|−0.543%|100%|
|`scheduler-workset-r1`|74549.10|11207.29|−0.512%|100%|
|`scheduler-components-r2`|74979.20|11238.17|−0.238%|100%|
|`control-dependencies-r1`|75081.77|11223.49|−0.369%|100%|
|`case-configuration-r1`|75041.43|11212.44|−0.467%|100%|
|`benchmark-components-r1`|75079.41|11193.29|−0.637%|100%|
|`replay-archive-r1`|75472.07|11279.34|+0.127%|100%|
|`installed-router-entry-r1`|74964.76|11200.41|−0.573%|100%|
|`shared-tool-paths-r1`|75288.22|11239.01|−0.231%|100%|

The first eight rows (through `scheduler-workset-r1`) are under
`/orcd/scratch/orcd/003/shangy/dualpd/runs/engine-cleanup-validation-20261108/`;
later cases are under
`/orcd/pool/004/shangy/dualpd/runs/engine-cleanup-validation-20261108/`.
The table is in implementation order, not filename sort order. The slight
non-unit reuse in the early DMA case is retained rather than rounded to100%.

The final Decode measurement is0.231% below the original. These are individual
regression runs, not repeated-trial confidence intervals, so the small variation
must not be presented as a statistically established optimization or regression.
Despite its name, `router-tcp-r1` is the local TP1 performance comparison after
TCP Router code extraction, not actual TCP model-path execution. Actual TP2 TCP
Direct/Host functionality has separate dense r7/r8 and hybrid r9 probes; their
two-turn outputs cannot be inserted into this throughput table.

See [SEQUENTIAL_ENGINE_CLEANUP.md](SEQUENTIAL_ENGINE_CLEANUP.md) for stage tests,
source proofs and independent acceptance, [FINAL_ENGINE_VALIDATION_PLAN.md](FINAL_ENGINE_VALIDATION_PLAN.md)
for representative protocol coverage, and [HYBRID_RUNTIME_VALIDATION.md](HYBRID_RUNTIME_VALIDATION.md)
for the inherited stable-prefix compatibility limitation. This summary does
not turn representative tests into measurements of all historical experiments.
