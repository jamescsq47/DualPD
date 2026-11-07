#!/usr/bin/env python3
"""Fixed-shape MiniMax P-side Forward/restore interference benchmark.

This is a diagnostic microbenchmark, not an end-to-end performance result.  It
loads the real model through SGLang's ``ModelRunner`` and replays aggregate
``(batch, new tokens, cached prefix tokens)`` shapes observed in P service
logs.  For every shape it measures:

* complete-context recomputation with production-sized 8192-token chunks;
* the same new-token Forward without a prefix (new-token compute baseline);
* the new-token Forward with its prefix already resident in P HBM; and
* that resident-prefix Forward while a production registered Host->HBM KV
  batch copy restores a disjoint snapshot on the copy stream.

The copied snapshot uses the real model KV pool and
``SharedMHAHostSnapshot``.  It is disjoint from the Forward request so the
experiment measures copy-engine/HBM contention rather than corrupting the KV
being read by Attention.  Cross-node NIXL time is intentionally measured from
the production ownership-handoff trace, not emulated in this single-node
process.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import os
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist

from sglang.bench_one_batch import load_model
from sglang.srt.disaggregation.agentic_host_staging import (
    H2DLaunchFence,
    LayerFirstD2HStaging,
    SharedMHAHostSnapshot,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    get_bool_env_var,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
)


GIB = 1 << 30


@dataclasses.dataclass(frozen=True)
class Shape:
    label: str
    batch_size: int
    new_tokens: int
    cached_tokens: int

    @classmethod
    def parse(cls, value: str) -> "Shape":
        try:
            label, batch, new, cached = value.split(":")
            shape = cls(label, int(batch), int(new), int(cached))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "shape must be LABEL:BATCH:NEW_TOTAL:CACHED_TOTAL"
            ) from exc
        if (
            shape.batch_size <= 0
            or shape.new_tokens <= 0
            or shape.cached_tokens < 0
            or shape.new_tokens % 64
            or shape.cached_tokens % 64
        ):
            raise argparse.ArgumentTypeError(
                "batch must be positive and token totals must be non-negative "
                "multiples of the 64-token production page size"
            )
        return shape


DEFAULT_SHAPES = (
    # Exact service-log observations nearest the marginal p25/p50/p75 points.
    "agentic_p25:1:704:14656",
    "agentic_p50:3:1536:29504",
    "agentic_p75:5:3008:49088",
    "hicache_p25:1:576:640",
    "hicache_p50:1:2304:7744",
    "hicache_p75:2:7616:17280",
    # The dominant HiCache recompute chunk (784 observations in c448).
    "hicache_mode:1:8192:256",
)


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, params: EvictParams):
        del params


def split_pages(total_tokens: int, batch_size: int) -> list[int]:
    """Split a page-aligned aggregate token count across a fixed batch."""

    pages, remainder = divmod(total_tokens // 64, batch_size)
    return [64 * (pages + (index < remainder)) for index in range(batch_size)]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def stats(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def maybe_prepare_mlp_sync(batch: ScheduleBatch, runner) -> None:
    if require_mlp_sync(runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=runner.server_args.dp_size,
            attn_tp_size=get_attention_tp_size(),
            attn_cp_size=runner.attn_cp_size,
            tp_group=runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(runner.server_args),
            disable_overlap_schedule=runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


def make_forward_batch(reqs: list[Req], runner, *, return_worker_batch=False):
    tree_cache = TreeCacheNamespace(
        page_size=runner.server_args.page_size,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    maybe_prepare_mlp_sync(batch, runner)
    worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    return (forward_batch, worker_batch) if return_worker_batch else forward_batch


def make_token_layout(
    prefix_lengths: list[int], new_lengths: list[int], seed: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Create one shared token corpus for every counterfactual condition."""

    rng = np.random.default_rng(seed)
    prefixes, suffixes, complete = [], [], []
    for prefix_length, new_length in zip(prefix_lengths, new_lengths):
        prefix = rng.integers(
            0, 200000, size=prefix_length, dtype=np.int32
        ).tolist()
        suffix = rng.integers(0, 200000, size=new_length, dtype=np.int32).tolist()
        prefixes.append(prefix)
        suffixes.append(suffix)
        complete.append(prefix + suffix)
    return prefixes, suffixes, complete


def make_requests(token_ids_list: list[list[int]], seed: int) -> list[Req]:
    sampling = SamplingParams(temperature=0, max_new_tokens=1)
    requests = []
    for index, token_ids in enumerate(token_ids_list):
        req = Req(
            rid=f"fixed-{seed}-{index}",
            origin_input_text="",
            origin_input_ids=token_ids,
            sampling_params=sampling,
        )
        req.fill_ids = []
        req.logprob_start_len = -1
        requests.append(req)
    return requests


def tp_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def tp_max(value: float, device) -> float:
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


@torch.no_grad()
def run_forward(runner, forward_batch, stream) -> tuple[float, float]:
    """Return TP-max GPU event milliseconds and synchronized wall milliseconds."""

    tp_barrier()
    started = time.perf_counter()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        begin.record(stream)
        runner.forward(forward_batch)
        end.record(stream)
    end.synchronize()
    wall_ms = (time.perf_counter() - started) * 1000.0
    return tp_max(float(begin.elapsed_time(end)), runner.device), tp_max(
        wall_ms, runner.device
    )


def append_chunk(reqs: list[Req], target_lengths: list[int], runner) -> list[Req]:
    """Fill one 8192-token chunk in source order, like PrefillAdder."""

    remaining = [target - len(req.fill_ids) for req, target in zip(reqs, target_lengths)]
    if not any(remaining):
        return []
    budget = 8192
    appended = []
    for req, target, left in zip(reqs, target_lengths, remaining):
        if left <= 0 or budget <= 0:
            continue
        if left >= budget and appended:
            break
        take = min(left, budget)
        take = take // 64 * 64
        if take <= 0:
            continue
        previous = len(req.fill_ids)
        req.fill_ids = req.origin_input_ids[: previous + take]
        if previous:
            req.prefix_indices = runner.req_to_token_pool.req_to_token[
                req.req_pool_idx, :previous
            ].to(req.prefix_indices.dtype)
            req.is_chunked = 1
        req.set_extend_input_len(take)
        budget -= take
        appended.append(req)
    return appended


@torch.no_grad()
def materialize_prefix(
    runner, reqs: list[Req], target_lengths: list[int], stream, *, measure: bool
) -> tuple[list[float], list[float]]:
    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    while True:
        active = append_chunk(reqs, target_lengths, runner)
        if not active:
            break
        forward_batch = make_forward_batch(active, runner)
        if measure:
            current_gpu, current_wall = run_forward(runner, forward_batch, stream)
            gpu_ms.append(current_gpu)
            wall_ms.append(current_wall)
        else:
            run_forward(runner, forward_batch, stream)
        for req in active:
            req.set_extend_input_len(0)
    return gpu_ms, wall_ms


def prepare_fixed_forward(
    runner,
    prefix_lengths: list[int],
    token_ids_list: list[list[int]],
    stream,
    seed: int,
) -> tuple[list[Req], ForwardBatch, object]:
    total_lengths = [len(token_ids) for token_ids in token_ids_list]
    reqs = make_requests(token_ids_list, seed)
    if any(prefix_lengths):
        materialize_prefix(runner, reqs, prefix_lengths, stream, measure=False)
    for req, prefix, total in zip(reqs, prefix_lengths, total_lengths):
        req.fill_ids = req.origin_input_ids[:total]
        if prefix:
            req.prefix_indices = runner.req_to_token_pool.req_to_token[
                req.req_pool_idx, :prefix
            ].to(req.prefix_indices.dtype)
            req.is_chunked = 1
        req.set_extend_input_len(total - prefix)
    forward_batch, worker_batch = make_forward_batch(
        reqs, runner, return_worker_batch=True
    )
    return reqs, forward_batch, worker_batch


def benchmark_full_recompute(
    runner,
    token_ids_list: list[list[int]],
    stream,
    repeats: int,
    seed: int,
) -> dict:
    samples_gpu: list[float] = []
    samples_wall: list[float] = []
    chunk_counts: list[int] = []
    total_lengths = [len(token_ids) for token_ids in token_ids_list]
    for repeat in range(repeats + 1):
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()
        reqs = make_requests(token_ids_list, seed)
        gpu_parts, wall_parts = materialize_prefix(
            runner, reqs, total_lengths, stream, measure=True
        )
        if repeat:
            samples_gpu.append(sum(gpu_parts))
            samples_wall.append(sum(wall_parts))
            chunk_counts.append(len(gpu_parts))
    return {
        "gpu_ms": stats(samples_gpu),
        "wall_ms": stats(samples_wall),
        "chunk_count": chunk_counts,
    }


def benchmark_resident(
    runner,
    prefix_lengths: list[int],
    token_ids_list: list[list[int]],
    stream,
    repeats: int,
    seed: int,
) -> tuple[dict, list[Req], ForwardBatch, object]:
    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    reqs, forward_batch, worker_batch = prepare_fixed_forward(
        runner, prefix_lengths, token_ids_list, stream, seed
    )
    run_forward(runner, forward_batch, stream)
    gpu_samples, wall_samples = [], []
    for _ in range(repeats):
        gpu, wall = run_forward(runner, forward_batch, stream)
        gpu_samples.append(gpu)
        wall_samples.append(wall)
    return {
        "gpu_ms": stats(gpu_samples),
        "wall_ms": stats(wall_samples),
    }, reqs, forward_batch, worker_batch


@torch.no_grad()
def benchmark_production_forward_envelope(
    runner, worker_batch, stream, repeats: int
) -> dict:
    """Time the production timer envelope, including ForwardBatch.init_new."""

    gpu_samples, wall_samples = [], []
    for repeat in range(repeats + 1):
        tp_barrier()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            begin.record(stream)
            forward_batch = ForwardBatch.init_new(worker_batch, runner)
            runner.forward(forward_batch)
            end.record(stream)
        end.synchronize()
        if repeat:
            gpu_samples.append(
                tp_max(float(begin.elapsed_time(end)), runner.device)
            )
            wall_samples.append(
                tp_max((time.perf_counter() - started) * 1000.0, runner.device)
            )
    return {"gpu_ms": stats(gpu_samples), "wall_ms": stats(wall_samples)}


def benchmark_host_dma_overlap(
    runner,
    forward_batch,
    stream,
    restore_tokens: int,
    repeats: int,
    rank: int,
    root: str,
    copy_stream_priority: int,
    chunk_tokens: int,
) -> dict:
    pool = getattr(runner.token_to_kv_pool, "full_kv_pool", runner.token_to_kv_pool)
    restore_tokens = max(64, ((restore_tokens + 63) // 64) * 64)
    device_indices = runner.token_to_kv_pool_allocator.alloc(restore_tokens)
    if device_indices is None:
        raise RuntimeError(f"cannot reserve {restore_tokens} disjoint DMA tokens")
    host_indices = device_indices.detach().cpu().numpy().astype(np.int64, copy=True)
    byte_size = (
        2
        * restore_tokens
        * int(pool.layer_num)
        * int(pool.head_num)
        * int(pool.head_dim)
        * int(pool.store_dtype.itemsize)
    )
    staging = LayerFirstD2HStaging(pool, chunk_tokens)
    copy_stream = torch.cuda.Stream(priority=copy_stream_priority)
    current_device_index = torch.cuda.current_device()
    Path(root).mkdir(parents=True, exist_ok=True)
    path = os.path.join(root, f"rank-{rank}-pid-{os.getpid()}.kv")
    snapshot = SharedMHAHostSnapshot(
        path=path,
        token_count=restore_tokens,
        device_pool=pool,
        byte_size=byte_size,
        create=True,
    )
    try:
        if not snapshot.prepare_registered_host_dma():
            raise RuntimeError("registered final-extent DMA is unavailable")

        # Populate the Host extent once; this setup transfer is outside all
        # reported H2D/Forward measurements.
        for start in range(0, restore_tokens, chunk_tokens):
            end = min(start + chunk_tokens, restore_tokens)
            fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
            event, _ = snapshot.start_backup_range_from_device(
                device_indices[start:end],
                destination_start=start,
                stream=copy_stream,
                staging=staging,
                launch_fence=fence,
                source_indices_host=host_indices[start:end],
            )
            event.synchronize()

        def launch_chunk(start):
            end = min(start + chunk_tokens, restore_tokens)
            fence = H2DLaunchFence(event=torch.cuda.Event(enable_timing=True))
            event, copy_refs = snapshot.start_load_range_from_bounce_to_device(
                device_indices[start:end],
                source_start=start,
                stream=copy_stream,
                staging=staging,
                launch_fence=fence,
                device_indices_host=host_indices[start:end],
            )
            # The descriptor arrays in copy_refs and the launch fence must
            # remain alive until the asynchronous registered-batch copy ends.
            return snapshot._last_h2d_start_event, event, fence, copy_refs

        def run_h2d_pipeline(first_launched=None):
            torch.cuda.set_device(current_device_index)
            gpu_ms = 0.0
            for start in range(0, restore_tokens, chunk_tokens):
                begin, event, fence, copy_refs = launch_chunk(start)
                if first_launched is not None:
                    first_launched.set()
                    first_launched = None
                event.synchronize()
                gpu_ms += float(begin.elapsed_time(event))
                del fence, copy_refs
            return gpu_ms

        def local_forward():
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            with torch.cuda.stream(stream):
                begin.record(stream)
                runner.forward(forward_batch)
                end.record(stream)
            end.synchronize()
            return (
                float(begin.elapsed_time(end)),
                (time.perf_counter() - started) * 1000.0,
            )

        # Warm descriptor coalescing, registered windows, and copy kernels.
        run_h2d_pipeline()
        dma_only_ms = []
        dma_only_wall_ms = []
        for _ in range(repeats):
            tp_barrier()
            started = time.perf_counter()
            gpu_ms = run_h2d_pipeline()
            wall = (time.perf_counter() - started) * 1000.0
            dma_only_ms.append(tp_max(gpu_ms, runner.device))
            dma_only_wall_ms.append(tp_max(wall, runner.device))

        paired_resident_ms = []
        paired_contention_ms = []
        overlap_forward_ms = []
        overlap_forward_wall_ms = []
        overlap_dma_ms = []
        overlap_makespan_ms = []
        overlap_external_wait_ms = []
        for repeat in range(repeats):
            tp_barrier()
            if repeat % 2 == 0:
                resident_gpu, _ = local_forward()
            wall_started = time.perf_counter()
            first_launched = threading.Event()
            pipeline_result = {}

            def progress_pipeline():
                try:
                    pipeline_result["gpu_ms"] = run_h2d_pipeline(first_launched)
                except BaseException as exc:
                    pipeline_result["exception"] = exc
                    first_launched.set()

            progress_thread = threading.Thread(target=progress_pipeline)
            progress_thread.start()
            try:
                first_launched.wait()
                gpu, forward_wall = local_forward()
            finally:
                # Never release the snapshot or device lease while a copy is
                # live, including when Forward itself raises.
                progress_thread.join()
            if "exception" in pipeline_result:
                raise pipeline_result["exception"]
            dma_gpu = pipeline_result["gpu_ms"]
            makespan = (time.perf_counter() - wall_started) * 1000.0
            if repeat % 2:
                resident_gpu, _ = local_forward()
            # Every local event is complete before TP reductions; collectives
            # cannot leak into Forward or overlap makespan timing.
            resident_gpu = tp_max(resident_gpu, runner.device)
            gpu = tp_max(gpu, runner.device)
            forward_wall = tp_max(forward_wall, runner.device)
            dma_gpu = tp_max(dma_gpu, runner.device)
            makespan = tp_max(makespan, runner.device)
            paired_resident_ms.append(resident_gpu)
            paired_contention_ms.append(gpu - resident_gpu)
            overlap_forward_ms.append(gpu)
            overlap_forward_wall_ms.append(forward_wall)
            overlap_dma_ms.append(dma_gpu)
            overlap_makespan_ms.append(makespan)
            overlap_external_wait_ms.append(max(0.0, makespan - forward_wall))

        return {
            "restore_tokens": restore_tokens,
            "bytes_per_rank": byte_size,
            "aggregate_tp_bytes": byte_size * dist.get_world_size()
            if dist.is_initialized()
            else byte_size,
            "dma_only_gpu_ms": stats(dma_only_ms),
            "dma_only_wall_ms": stats(dma_only_wall_ms),
            "dma_only_per_rank_GiB_s": stats(
                [byte_size / GIB / (value / 1000.0) for value in dma_only_ms]
            ),
            "dma_only_aggregate_tp_GiB_s": stats(
                [
                    byte_size
                    * (dist.get_world_size() if dist.is_initialized() else 1)
                    / GIB
                    / (value / 1000.0)
                    for value in dma_only_ms
                ]
            ),
            "overlap_forward_gpu_ms": stats(overlap_forward_ms),
            "paired_resident_forward_gpu_ms": stats(paired_resident_ms),
            "paired_dma_contention_gpu_ms": stats(paired_contention_ms),
            "overlap_forward_wall_ms": stats(overlap_forward_wall_ms),
            "overlap_dma_gpu_ms": stats(overlap_dma_ms),
            "overlap_makespan_wall_ms": stats(overlap_makespan_ms),
            "forward_external_wait_wall_ms": stats(overlap_external_wait_ms),
            "copy_stream_priority_effective": copy_stream.priority,
            "pipeline_semantics": "one 1024-token chunk launched after prior chunk completion",
        }
    finally:
        snapshot.close(unlink=True)
        runner.token_to_kv_pool_allocator.free(device_indices)


def run_worker(server_args, bench_args, port_args, gpu_id: int, rank: int) -> None:
    expected = {
        "tp_size": 8,
        "ep_size": 8,
        "page_size": 64,
        "chunked_prefill_size": 8192,
        "context_length": 131072,
    }
    mismatches = {
        key: (getattr(server_args, key, None), value)
        for key, value in expected.items()
        if getattr(server_args, key, None) != value
    }
    if mismatches:
        raise RuntimeError(f"non-production benchmark configuration: {mismatches}")
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, rank
        )
    wrapped_runner, _ = load_model(server_args, port_args, gpu_id, rank)
    runner = wrapped_runner.torch_runner
    forward_stream = torch.cuda.Stream(priority=bench_args.forward_stream_priority)
    results = []
    try:
        for shape_index, shape in enumerate(bench_args.shape):
            prefix_lengths = split_pages(shape.cached_tokens, shape.batch_size)
            new_lengths = split_pages(shape.new_tokens, shape.batch_size)
            _prefixes, suffixes, complete = make_token_layout(
                prefix_lengths,
                new_lengths,
                bench_args.seed + shape_index * 100,
            )
            full = (
                None
                if bench_args.skip_full_recompute
                else benchmark_full_recompute(
                    runner,
                    complete,
                    forward_stream,
                    bench_args.repeats,
                    bench_args.seed + shape_index * 100 + 1,
                )
            )
            new_only, _, _, _ = benchmark_resident(
                runner,
                [0] * shape.batch_size,
                suffixes,
                forward_stream,
                bench_args.repeats,
                bench_args.seed + shape_index * 100 + 2,
            )
            resident, _, resident_batch, resident_worker_batch = benchmark_resident(
                runner,
                prefix_lengths,
                complete,
                forward_stream,
                bench_args.repeats,
                bench_args.seed + shape_index * 100 + 3,
            )
            production_envelope = benchmark_production_forward_envelope(
                runner,
                resident_worker_batch,
                forward_stream,
                bench_args.repeats,
            )
            dma = benchmark_host_dma_overlap(
                runner,
                resident_batch,
                forward_stream,
                shape.cached_tokens,
                bench_args.repeats,
                rank,
                os.path.join(bench_args.host_root, shape.label),
                bench_args.copy_stream_priority,
                bench_args.h2d_chunk_tokens,
            )
            resident_ms = dma["paired_resident_forward_gpu_ms"]["p50"]
            new_only_ms = new_only["gpu_ms"]["p50"]
            overlap_ms = dma["overlap_forward_gpu_ms"]["p50"]
            results.append(
                {
                    "shape": dataclasses.asdict(shape),
                    "per_request_prefix_tokens": prefix_lengths,
                    "per_request_new_tokens": new_lengths,
                    "full_context_recompute": full,
                    "new_tokens_without_prefix": new_only,
                    "hbm_resident_prefix_forward": resident,
                    "production_forward_envelope": production_envelope,
                    "host_h2d_and_forward": dma,
                    "decomposition_p50_gpu_ms": {
                        "new_token_compute_no_prefix_counterfactual": new_only_ms,
                        "long_prefix_incremental_path_residual_signed": (
                            resident_ms - new_only_ms
                        ),
                        "cached_kv_attention_upper_bound": max(
                            0.0, resident_ms - new_only_ms
                        ),
                        "dma_hbm_contention_signed": dma[
                            "paired_dma_contention_gpu_ms"
                        ]["p50"],
                        "dma_hbm_contention": max(
                            0.0,
                            dma["paired_dma_contention_gpu_ms"]["p50"],
                        ),
                        "resident_total": resident_ms,
                        "overlapped_total": overlap_ms,
                    },
                }
            )
            if rank == 0:
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        if rank == 0:
            payload = {
                "kind": "minimax_p_forward_decomposition",
                "model_path": server_args.model_path,
                "tp_size": server_args.tp_size,
                "ep_size": server_args.ep_size,
                "page_size": server_args.page_size,
                "context_length": server_args.context_length,
                "chunked_recompute_tokens": 8192,
                "host_h2d_chunk_tokens": bench_args.h2d_chunk_tokens,
                "forward_stream_priority_requested": bench_args.forward_stream_priority,
                "copy_stream_priority_requested": bench_args.copy_stream_priority,
                "forward_stream_priority_effective": forward_stream.priority,
                "attention_backend": server_args.attention_backend,
                "dtype": str(server_args.dtype),
                "kv_cache_dtype": str(server_args.kv_cache_dtype),
                "shape_origin": "exact aggregate triples observed in c448 P logs",
                "per_request_vector": "page-aligned balanced synthetic split; aggregate triple is exact",
                "attribution_note": (
                    "resident-new_only is an upper bound on cached-KV/long-prefix Attention; "
                    "MiniMax MoE routing and collectives are a residual confounder"
                ),
                "sglang_git_sha": subprocess.check_output(
                    ["git", "-C", "/home/shangy/dualpd/sglang-src/pd", "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "repeats": bench_args.repeats,
                "results": results,
            }
            Path(bench_args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(bench_args.output).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--shape",
        action="append",
        type=Shape.parse,
        default=None,
        help="repeatable LABEL:BATCH:NEW_TOTAL:CACHED_TOTAL",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--forward-stream-priority", type=int, default=-5)
    parser.add_argument("--copy-stream-priority", type=int, default=-1)
    parser.add_argument("--h2d-chunk-tokens", type=int, default=1024)
    parser.add_argument(
        "--skip-full-recompute",
        action="store_true",
        help="skip the expensive recompute arm for distribution sampling",
    )
    parser.add_argument("--host-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    args.shape = args.shape or [Shape.parse(value) for value in DEFAULT_SHAPES]
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    server_args = ServerArgs.from_cli_args(args)
    server_args.cuda_graph_max_bs = max(shape.batch_size for shape in args.shape)
    server_args.disable_cuda_graph = True
    server_args.disable_piecewise_cuda_graph = True
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    workers = []
    try:
        for rank in range(server_args.tp_size):
            with maybe_reindex_device_id(rank) as gpu_id:
                process = multiprocessing.Process(
                    target=run_worker,
                    args=(server_args, args, port_args, gpu_id, rank),
                )
                process.start()
                workers.append(process)
        failed = False
        for process in workers:
            process.join()
            failed |= process.exitcode != 0
        if failed:
            raise SystemExit("one or more TP benchmark workers failed")
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
