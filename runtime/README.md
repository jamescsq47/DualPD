# Application runtime

This is the shared **client-side runtime**, not SGLang's model-serving engine.

- `inference.py`: workload driver, concurrency, measurement windows and outputs.
- `workloads/`: workload YAML parsing, sample loading and deterministic dispatch.
- `agentic_kv_request.py`: request-generation metadata and tool/final ACKs.
- `http_utils.py`, `model_http_transport.py`: model HTTP requests.
- `types.py`, `generation.py`, `dataset.py`, `processing_utils.py`: sample,
  tokenizer and generation state used by the two harnesses.
- Metrics/tracing modules: measurement and request observations.
- `cli.py`, `engine_entry.py`: command dispatch and the engine launch bridge.
- `launch/`: shared local/Slurm process startup and cleanup used by experiments.
- `analysis/`, `observation/`: final-run summaries and non-policy-changing
  resource samples needed by the measurement and memory-ablation workflows.

GPU KV allocation, Router scheduling, DMA and Host storage implementation live
in `3rdparty/sglang`, not here. Run the driver with
`python -m dualpd inference --help`; the public launchers configure it for you.
