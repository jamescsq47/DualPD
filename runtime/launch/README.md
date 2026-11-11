# Shared service launch implementation

Public experiment commands live only in `scripts/new-method/`,
`scripts/baseline/` and `scripts/ablation/`. This directory is their reusable
implementation, not another collection of experiment recipes.

- `local/`: P:D topology construction, model/search/Router startup and cleanup.
- `baseline/`: clean native PD and colocated service startup.
- `cluster/`: TCP/Slurm deployment and exact process-group supervision.
- `environment.py`: modified/baseline separation gate.
- `profiles.py`: checked optional experiment-profile loading.
- `common.sh`: shared process lifecycle functions.

Use the public scripts with `--help` for Slurm model, TP and topology flags.
`configs/deployment/multinode.json` is a template: replace paths, node addresses
and the control token. Existing exclusive Slurm allocations are required by
the Slurm frontends; they do not submit or renew allocations.

Docker task images are the default user setup. Prepare Apptainer images through
`dualpd.data.swe_bench_verified.images` if your site uses the Apptainer workload
profiles. Site network, storage, baseline interpreter and engine paths must
be supplied explicitly. No hidden switch substitutes the modified engine in
a native baseline run.
