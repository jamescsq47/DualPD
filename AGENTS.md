# DualPD development

This checkout publishes BrowseComp and SWE-bench Verified only. Keep raw data,
model weights, credentials, run outputs and local test reports out of Git.
The original slime and sglang-src worktrees are comparison/backup sources;
do not modify them as part of release cleanup.

runtime/ is the application client, not the GPU engine. Transport, routing,
allocator and Scheduler implementations belong to the pinned SGLang submodule.
Preserve sampling, prompts, tool protocols, task verification and measurement
windows when changing file organization. No cleanup authorizes changing them.

Preserve snapshot ownership: transfer only complete snapshots; do not release
source HBM or Host extents before durable completion and physical DMA fences.
Claim, path selection, commit and release must be atomic across TP ranks.
Direct, Host recovery, tool execution and Forward progress must remain decoupled.
Shared radix prefixes retain reference-safe lifetimes. Explicit recomputation
must be distinguished from genuine KV loss. Cancel/failure paths must converge
before reusing buffers. Never substitute the modified engine in baseline runs.

Validate imports, launch plans and lifecycle/fault cases outside the release
tree before GPU experiments. Lifecycle changes require independent audit before
GPU validation. CPU checks do not establish throughput or accuracy acceptance.
BrowseComp acceptance needs 300 s warmup + 1200 s measurement; SWE uses the
recorded T450 window. Preserve the upstream engine's own development tests.
