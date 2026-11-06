# SGLang PR #45423 decode-local-prefill validation (H100)

## Status

Validated on 2026-10-29. The PR can serve follow-up turns directly on the decode server, reuse the decode-side radix cache, and compute only the uncached suffix. Ordinary P→D requests and decode-local-prefill requests can share one decode server.

This validates the engine mechanism. PR #45423 does not provide the router policy that decides when a request should use decode-local prefill or which decode replica owns its conversation cache; the benchmark harness must provide conversation affinity and set `disagg_local_prefill=true`.

## Long-lived allocation

| Field | Value |
|---|---|
| Slurm job | `27220658` (`ppd45423-lab`) |
| Node | `node1701` |
| Resources | 8×H100 80GB, 224 CPU, 1800GB host memory, exclusive |
| Start | 2026-10-29 07:26:40 UTC |
| End | 2026-11-05 06:26:40 UTC |

Only one seven-day node is reserved initially. A second node should be requested only for a cross-node P/D experiment.

A quiet 10-minute watchdog monitors the allocation and queues an alert to this Codex thread only if the Slurm job disappears or stops running. Its state is `/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/allocation-watchdog-state.json`.

## Isolated environment

| Component | Value |
|---|---|
| PR source | `/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/sglang` |
| PR commit | `bf3e4122565e7cd3131d6b8d0c5c9f8cf4eda6b1` |
| Runtime image | `/orcd/pool/004/shangy/dualpd/containers/sglang-v0.5.22-ppd45423.sif` |
| Image source commit | `baa9befa381511cd4d56f83980951e5a45fffc6d` |
| Python / Torch / CUDA | Python 3.12.15 / Torch 2.14.1 / CUDA 13.0 |
| SGLang kernel | 0.4.9 (passes the PR runtime minimum-version gate) |
| FlashInfer / Transformers | 0.7.1 / 5.19.0 |
| Rust extension cache | `/orcd/pool/004/shangy/dualpd/ppd45423-cache` |
| Model | `/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-8B` |
| Transfer backend | NIXL, one-node GPU0=P and GPU1=D |

The existing `sglang-src/pd`, `sglang-src/pd_baseline`, and their Python environments were not modified.

## Validation results

| Check | Result | Evidence |
|---|---:|---|
| Decode-local-prefill unit tests | PASS | 15 passed, 5 subtests passed in 30.00s |
| Multi-turn decode cache reuse | PASS | Four cases; each second turn reported at least 1056 cached tokens |
| Uncached suffix execution | PASS | A 1457-token second-turn prompt reused 1056 tokens and computed 401 tokens as 128+128+128+17 chunks |
| Output parity | PASS | First-token KL against a cache-salted ordinary PD reference was below 0.05 |
| Mixed ordinary PD + local prefill | PASS | 32 requests at 16 client threads; all reached length finish and both P/D health checks returned 200 |
| Wrong-side request rejection | PASS | Prefill server returned HTTP 400 for `disagg_local_prefill=true` |
| Full Qwen3-8B E2E suite | PASS | 3 passed in 140.17s |
| Final cleanup gate | PASS | Zero GPU processes, experiment ports, and Apptainer/FUSE helpers |

Logs:

- `/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/logs/unit-v0522.log`
- `/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/logs/e2e-qwen3-8b-v0522-r2.log`
- `/orcd/scratch/orcd/003/shangy/dualpd/ppd45423/logs/build-rust-tree-r2.log`

## Reproduction commands

```bash
# Build or verify the source-matched Rust radix-tree extension once.
slime/tools/dualpd/ppd45423_build_rust_tree.sh

# PR unit tests.
slime/tools/dualpd/ppd45423_run_unit.sh

# Qwen3-8B engine-level PPD validation.
slime/tools/dualpd/ppd45423_run_qwen3_8b_e2e.sh
```

## Next benchmark step

Add conversation-to-decode affinity to the existing agentic harness. Route the first turn through ordinary P→D; for later turns, send the full conversation directly to the owning D with `disagg_local_prefill=true`. Compare this method with ordinary PD/No-reverse and HiCache under the same 300-warmup + 1200-steady-state protocol. Record the local-prefill selection rate, decode cached-token ratio, suffix-prefill tokens, TTFT, decode throughput, and total throughput.
