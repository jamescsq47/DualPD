# Published workloads and data licenses

This directory contains **harness code, not raw datasets**. Only two workloads
are supported: `browsecomp/` and `swe_bench_verified/`. Download data and
container images separately as described in the root [README](../README.md).
Shared configuration, sampling and loading live in `runtime/workloads/`.

| Workload/resource | Public source | Access and upstream license information |
| --- | --- | --- |
| SWE-bench Verified, 500 test instances | [Official dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | Publicly downloadable benchmark. The [SWE-bench implementation](https://github.com/SWE-bench/SWE-bench/blob/main/LICENSE) is MIT; original task repositories and container dependencies retain their own licenses. The dataset card does not declare a separate blanket dataset license; do not equate the code license with all task content. |
| BrowseComp question split, 680 training questions | [FoldAgent release](https://github.com/sunnweiwei/FoldAgent) and its linked data archive | Publicly downloadable benchmark split. [FoldAgent code](https://github.com/sunnweiwei/FoldAgent/blob/main/LICENSE) is Apache-2.0; that alone does not establish a separate license for the linked archive or third-party question content. Use the upstream data terms; this repository does not redistribute it. |
| BrowseComp local retrieval corpus | [Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus) | Dataset card declares MIT. Original document contents may retain third-party rights. |
| Precomputed corpus embeddings | [miaolu3/browsecomp-plus](https://huggingface.co/datasets/miaolu3/browsecomp-plus) | Hosting dataset card declares Apache-2.0; underlying corpus rights remain separate. |

These are external public benchmarks, not DPD-owned data released under
DPD's Apache-2.0 license. Check upstream terms before redistribution. Model
weights likewise retain the licenses on their model cards.

BrowseComp tools call a local search/open service over the fixed corpus.
SWE-bench tools execute shell commands in isolated Docker or Apptainer task
environments; hidden tests are injected only after agent termination.
`swe_bench_verified/sandbox.py` owns sandbox setup/teardown and `verifier.py`
owns official task grading. No benchmark payloads, task images, trajectories
or internal experiment results are bundled here.

Preparation commands also live alongside their workload, not under `scripts`:

```bash
python -m dualpd.data.browsecomp.prepare --help
python -m dualpd.data.swe_bench_verified.prepare --help
python -m dualpd.data.swe_bench_verified.images.prefetch_docker --help
python -m dualpd.data.swe_bench_verified.images.prefetch_oci --help
python -m dualpd.data.swe_bench_verified.images.convert_to_sif --help
python -m dualpd.data.swe_bench_verified.images.load_oci --help
```
