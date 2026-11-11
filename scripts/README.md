# Experiment commands

Run from the DualPD root after installation.

| Directory | Retained purpose |
| --- | --- |
| `new_method/` | Main-method public entry and reusable local P:D/TP topologies |
| `baseline/` | Native No-reverse, HiCache + Mooncake and colocated launch components |
| `ablations/` | One-variable BrowseComp ablation entry |
| `cluster/` | TCP multi-node/Slurm launch, process supervision and resource observations |
| `tools/` | Dataset preparation, environment checks and final-run analysis |
| `common/` | Process-group lifecycle and cleanup shared by launchers |

Concurrency defaults are editable in the public launch scripts. Model/data and
output paths are supplied by the caller. Historical replay controllers,
watchdogs, bandwidth microbenchmarks and smoke/test scripts are not distributed.
