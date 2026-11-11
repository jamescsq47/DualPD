# Multi-node and Slurm experiments

The root README covers local single-node commands. For distributed runs:

- `multinode.py` / `multinode.sh`: configuration, launch plans and TCP runtime.
- `qwen35_slurm.py`: main-method Slurm deployment, model/TP and topology options.
- `qwen35_native_pd_slurm.py`: native PD / HiCache + Mooncake baseline.
- `qwen35_colocated_slurm.py`: colocated baseline.
- `process_supervisor.py`: scoped process-group identity, stop and cleanup.
- `dram_resource_sampler.py`, `dram_complete_frame_analysis.py`: DRAM samples
  and complete-frame analysis for memory ablations.
- `dram_observer/sitecustomize.py`: opt-in pool byte counters for DRAM studies;
  enabled only when PD_DRAM_TRACE_DIR is set and its directory is on PYTHONPATH.
  This observes pools without changing their cache policy.
- OCI/SIF helpers: prepare official task images for Apptainer sites.
- `collect_hardware_topology.sh`: passive GPU/NUMA/network inventory.

```bash
python -m dualpd multinode --help
python -m dualpd slurm --help
python scripts/cluster/qwen35_native_pd_slurm.py --help
python scripts/cluster/qwen35_colocated_slurm.py --help
```

Copy `multinode.example.json` outside the repository and replace all paths,
node addresses, network devices and the control token. Inspect a launch plan
before starting services. Two-node runs require accessible model/data paths,
peer network connectivity and a separate baseline environment when applicable.
The Slurm frontends require an existing exclusive allocation, with the node
and job flags supplied explicitly. They do not submit or renew allocations.

SWE configurations use Docker by default, with separate Apptainer and overlay
profiles. For SIF conversion, set SWE_SIF_DATASET, SWE_SIF_OCI_LAYOUT,
SWE_SIF_DIR, SWE_SIF_TEMP_BASE, SWE_SIF_LOG_DIR and SWE_SIF_POLICY to your site
paths. The policy must be a readable site-approved containers/image signature
policy; the release does not ship an insecure default policy.
