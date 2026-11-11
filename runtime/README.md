# Application runtime

This is the shared **client-side runtime**, not SGLang's model-serving engine.

- `inference.py`: workload driver, concurrency, measurement windows and outputs.
- `workloads/`: workload YAML parsing, sample loading and deterministic dispatch.
- `client/`: request-generation wire metadata, tool/final notifications, HTTP
  transport, sample types, tokenization and generation state for both harnesses.
- `cli.py`, `engine_entry.py`: command dispatch and the engine launch bridge.
- `launch/`: shared local/Slurm process startup and cleanup used by experiments.
- `analysis/`, `observation/`: final-run summaries and non-policy-changing
  metrics/tracing/resource samples for measurement and memory ablations.

GPU KV allocation, Router scheduling, DMA and Host storage implementation live
in `3rdparty/sglang`, not here. Run the driver with
`python -m dualpd inference --help`; the public launchers configure it for you.

## Engine boundary

Client metadata and tool/final notification timing remain application-owned.
Local ACKs call the pinned engine's public
`sglang.srt.disaggregation.agentic_client.publish_local_ack`; applications never
construct an internal marker store. Multi-node final ACKs still go through the
existing Router HTTP callback, with unchanged timeout/retry and failure policy.
The engine adapter performs the same local marker publication as before; it
does not allocate/release KV or invent a new ownership transition.

`sglang.srt.disaggregation.agentic_*` are intentional public imports, even
though their files live in the engine's `dualpd/` source directory. SGLang
extends the package search path to keep one state-class/module identity.
Do not mechanically rewrite those imports to `disaggregation.dualpd.*`.

For the modified engine, Host arena managers register read-only byte counters
with its opt-in `agentic_host_observation` module. DualPD consumes the records
and samples PSS/node/cgroup memory; it does not monkey-patch those managers.
`observation/host_pools/sitecustomize.py` is only an external adapter for stock
baseline `HostKVCache`, never an engine policy patch or a custom-engine hook.
Both paths retain the same raw counter/curve schema and measurement windows.

Use the updated pinned engine for the local ACK adapter and native Host
observation; updating only the frontend while retaining an older custom engine
does not supply those interfaces. Clean baseline engines remain separate.
