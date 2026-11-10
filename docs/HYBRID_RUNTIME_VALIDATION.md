# Hybrid runtime validation: preserved compatibility, not full-tail reuse

## Actual model probe r9

The external case `tcp-tp2-probes/hybrid-tp2-native-r9-20261110/` under
`/orcd/pool/004/shangy/dualpd/runs/engine-cleanup-validation-20261108/`
runs the cached Qwen3.5-35B-A3B model on the retained node2700 allocation
28285059. P uses GPUs0/1, D uses GPUs4/5; both groups use TP2 and the maintained
TCP protocol. It is a short functional probe, not a BrowseComp performance or
SWE accuracy comparison.

The client validates the actual native Qwen3.5 XML `lookup(key=alpha)` tool
call through the original TOOL classification. It constructs the next input
from the returned token IDs plus a deterministic tool-result suffix, without
rerendering history, forcing output or substituting a synthetic lifecycle marker.
The fresh full-recompute reference uses the same next input and sampling.

| Requested tool wait | Parent input | Parent output | Logical parent | Child input | Child/reference output | Exact IDs and finish reason |
|---|---:|---:|---:|---:|---:|---|
|0s|297|24|321|344|13 / 13|Equal|
|2s|297|24|321|344|13 / 13|Equal|

The requested waits alone do not prove routes. Independent offline review of
four healthy hash-bound final checkpoint12 prefixes establishes actual D2P
Direct for0s and Host store/restore for2s. All14 Direct source chains pass
(12 P2D plus2 D2P rank chains). On both D ranks, Host-store DMA completion154
precedes native Req/Mamba release158, authority retirement159 and source
commit161. Both target ranks provide loaded READ receipts165/166 before source
Host extent release169 and commit170. P0/P1 restore I/O172/146 precedes ready
commit178/152; P0 coordinator retirement180 clears the pending marker/policy.
Four child/reference terminal generations have actual extent and policy
retirement before shutdown, not just HTTP final-callback ACKs. These sequence
numbers are worker-local, not a cross-process global ordering.

Independent review gives a bounded hybrid Direct/D2P Host functional GO under
the preserved stable-prefix compatibility contract, releasing this source
freeze. It does not authorize another GPU launch or imply full-parent reuse,
hybrid P2D Host coverage, global allocator quiescence, throughput or accuracy.
The external `runtime-result.json` records the scoped result and journal counts.

Controller termination is exit0. The saved final gate reports GPU processes0,
Apptainer/FUSE helpers0, all nine test ports bindable, unchanged zero memory
events and the original allocation keeper. `postrun-source-freeze-check.json`
reports `matches: true`. Planned dependency-ordered service stops do not imply
that each child service naturally exited0. No engine/shared-environment change
was made for this probe.

## Why the 256-token return is not a page-tail pass

Both P ranks report an imported parent of256 tokens for each child. The Host
case's D snapshot also has256 tokens. The D parent release observation records
`kv_committed_len=321`. Thus321 logical parent tokens minus256 imported tokens
is65; even excluding one sampled token gives320 minus256 =64, still greater
than the page64 tail allowance63. Output equality does not satisfy the strict
full-parent reuse criterion by itself.

The reason is the existing opt-in stable-prompt Mamba checkpoint mode, documented
in [QWEN35_STABLE_PROMPT_CHECKPOINT.md](QWEN35_STABLE_PROMPT_CHECKPOINT.md).
`p2d_mamba_destination_indices` freezes the page-aligned prompt checkpoint;
`frozen_mamba_checkpoint` disables subsequent Decode checkpoint tracking in
`schedule_batch` and `_mamba_prefix_cache_update`.
`snapshot_token_count_for_req` returns that validated frozen boundary. The
observed256-token return is consistent with initial prompt297 aligned to64;
the frozen field itself was not directly journaled, so this causal connection
combines runtime observations with source inspection.

The original compatibility design deliberately retains a checkpoint before
history that a later chat-template rendering may change. Its contract explicitly
requires recomputing the generated suffix; it is not a promise to reuse the
entire generated parent except for one page tail. This behavior predates the
extraction and must not silently change during semantics-preserving cleanup.
In particular, simply enabling Decode tracking would overwrite the stable
state needed by the existing SWE harness. Turning the mode off, editing the
prompt/page size or shortening the output would not fix that compatibility
requirement and would not validate the original configuration.

Consequently this probe supplies real hybrid model/output evidence, but is
**not** a full-parent page-tail reuse GO. The strict invariant and this legacy
stable-prefix mode have different scopes; their tension is now explicit rather
than hidden by the dense-model success. A dual-checkpoint design or an explicit
recompute accounting change would be a separate algorithm/lifecycle change,
requiring its own capacity/retraction/TP/fault tests and audit. No such change
is included in this cleanup result, and the overall final acceptance stays open.
