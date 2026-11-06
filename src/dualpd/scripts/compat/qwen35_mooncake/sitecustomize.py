"""Run-local SGLang 0.5.10 compatibility for hybrid Mamba Mooncake HiCache.

This module is loaded only when its directory is explicitly prepended to
PYTHONPATH.  It intentionally avoids modifying the pd_baseline environment.
The compatibility methods are a narrow backport of the upstream SGLang
hybrid-pool Mooncake implementation for PoolName.MAMBA.
"""

from __future__ import annotations

import logging
import json
import os
import threading
import time
from typing import List, Optional

import numpy as np
import torch

from io_serialization import serialized_store_call
from spawn_retry import install_posix_spawn_eagain_retry


# Install before SGLang/Triton imports.  This changes only the transient
# EAGAIN branch of Python's private posix_spawn fast path; successful compiler
# launches and all non-EAGAIN failures retain their stock behavior.
install_posix_spawn_eagain_retry()

from offload_admission import (
    HybridCapacityBudget,
    HybridOffloadAdmission,
    exact_mamba_snapshot_length,
    host_capacity_shortfall,
    is_structurally_attached,
    needs_internal_prefix_backup,
    restore_sources_available,
    stable_mamba_track_index,
)

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.memory_pool_host import (
    HostPoolGroup,
    MambaPoolHost,
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
    PoolEntry,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeBaseStore,
    MooncakeStore,
)
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet


logger = logging.getLogger(__name__)


# The baseline snapshot predates the Qwen3.5/Qwen3.8 GDN fallback-layout
# correctness fix from da60b8134.  With TP=2 the fallback splitter returns
# strided ``a``/``b`` views, while the GDN kernel reads them as dense tensors.
# This is mostly hidden at lower pressure but corrupts generations at c256
# (the SWE harness then terminates them as ``no_command``).  Keep the baseline
# source tree immutable and backport the two required contiguous conversions
# through the run-local compatibility layer.
_original_fix_query_key_value_ordering = (
    Qwen3_5GatedDeltaNet.fix_query_key_value_ordering
)


def _fix_query_key_value_ordering_with_contiguous_gates(self, *args, **kwargs):
    query, key, value, z, b, a = _original_fix_query_key_value_ordering(
        self, *args, **kwargs
    )
    return query, key, value, z, b.contiguous(), a.contiguous()


# Mooncake 0.3.12's native store aborts the process when a single store is
# entered concurrently by the HiCache backup and prefetch threads. Serialize
# only the three native calls on each store instance; workers and store
# instances remain independent, and all stock Mooncake semantics are retained.
_original_put_batch_zero_copy_impl = MooncakeStore._put_batch_zero_copy_impl
_original_get_batch_zero_copy_impl = MooncakeStore._get_batch_zero_copy_impl
_original_batch_exist = MooncakeStore._batch_exist


def _serialized_put_batch_zero_copy_impl(self, *args, **kwargs):
    return serialized_store_call(
        self, _original_put_batch_zero_copy_impl, *args, **kwargs
    )


def _serialized_get_batch_zero_copy_impl(self, *args, **kwargs):
    return serialized_store_call(
        self, _original_get_batch_zero_copy_impl, *args, **kwargs
    )


def _serialized_batch_exist(self, *args, **kwargs):
    return serialized_store_call(self, _original_batch_exist, *args, **kwargs)


def _mamba_get_hybrid_pool_buffer(self: MambaPoolHost):
    return [self.temporal_buffer, *self.conv_buffer]


def _mamba_get_page_buffer_meta(self: MambaPoolHost, indices):
    """Return zero-copy Mooncake pointers for page-first Mamba state."""
    assert len(indices) % self.page_size == 0
    if self.layout not in ("page_first", "page_first_direct"):
        raise ValueError(
            f"Mamba Mooncake zero-copy requires page_first layout, got {self.layout}"
        )

    temporal_elems = int(np.prod(self.temporal_state_shape))
    conv_elems = [int(np.prod(shape)) for shape in self.conv_state_shapes]
    temporal_bytes = (
        self.page_size
        * self.num_mamba_layers
        * self.temporal_dtype.itemsize
        * temporal_elems
    )
    conv_bytes = [
        self.page_size
        * self.num_mamba_layers
        * self.conv_dtype.itemsize
        * elems
        for elems in conv_elems
    ]
    temporal_base = self.temporal_buffer.data_ptr()
    conv_bases = [buffer.data_ptr() for buffer in self.conv_buffer]

    ptrs = []
    sizes = []
    for index in indices.tolist()[:: self.page_size]:
        if temporal_elems > 0:
            ptrs.append(
                temporal_base
                + index
                * self.num_mamba_layers
                * temporal_elems
                * self.temporal_dtype.itemsize
            )
            sizes.append(temporal_bytes)
        for component, elems in enumerate(conv_elems):
            ptrs.append(
                conv_bases[component]
                + index
                * self.num_mamba_layers
                * elems
                * self.conv_dtype.itemsize
            )
            sizes.append(conv_bytes[component])
    return ptrs, sizes


_original_mamba_init = MambaPoolHost.__init__
_original_copy_tensor_all_layers_lf_pf = (
    MambaPoolHost._copy_tensor_all_layers_lf_pf
)
_original_copy_tensor_pf_lf = MambaPoolHost._copy_tensor_pf_lf


def _mamba_init_with_component_sizes(self: MambaPoolHost, *args, **kwargs):
    override = os.getenv("SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB")
    if override:
        positional = list(args)
        if len(positional) >= 3:
            positional[2] = int(override)
        else:
            kwargs["host_size"] = int(override)
        args = tuple(positional)
    _original_mamba_init(self, *args, **kwargs)
    self.temporal_state_elem_size = int(np.prod(self.temporal_state_shape))
    self.conv_state_elem_sizes = [
        int(np.prod(shape)) for shape in self.conv_state_shapes
    ]


def _copy_tensor_all_layers_lf_pf_with_cuda_dst_indices(
    src_layers,
    dst,
    src_indices,
    dst_indices,
    num_layers,
    device,
    io_backend,
):
    """Backport the hybrid-cache kernel's CUDA destination-index fix.

    SGLang 0.5.10 keeps host-pool indices on CPU, while the fused
    layer-first -> page-first Mamba copy kernel requires both index tensors on
    CUDA.  Upstream now performs this asynchronous device copy immediately
    before launching the kernel.
    """
    if io_backend == "kernel" and dst_indices.device.type != "cuda":
        dst_indices = dst_indices.to(src_indices.device, non_blocking=True)
    return _original_copy_tensor_all_layers_lf_pf(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        num_layers,
        device,
        io_backend,
    )


def _copy_tensor_pf_lf_with_cuda_src_indices(
    src,
    dst,
    src_indices,
    dst_indices,
    layer_id,
    num_layers,
    io_backend,
):
    """Backport the symmetric Host->GPU Mamba index-device fix."""
    if io_backend == "kernel" and src_indices.device.type != "cuda":
        src_indices = src_indices.to(dst_indices.device, non_blocking=True)
    return _original_copy_tensor_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        layer_id,
        num_layers,
        io_backend,
    )


_original_register_mem_pool_host = MooncakeStore.register_mem_pool_host


def _register_mem_pool_host(self: MooncakeStore, mem_pool_host):
    if isinstance(mem_pool_host, HostPoolGroup):
        # The legacy v1 Mooncake path remains responsible for the KV anchor.
        mem_pool_host = mem_pool_host.anchor_entry.host_pool
    return _original_register_mem_pool_host(self, mem_pool_host)


def _register_mem_host_pool_v2(
    self: MooncakeStore, host_pool, host_pool_name
):
    if host_pool_name == PoolName.KV:
        return
    if host_pool_name != PoolName.MAMBA:
        raise ValueError(
            f"Qwen3.5 compatibility only supports the Mamba side pool, got {host_pool_name}"
        )
    if not hasattr(self, "registered_pools"):
        self.registered_pools = {}
    self.registered_pools[host_pool_name] = host_pool
    for buffer in host_pool.get_hybrid_pool_buffer():
        if buffer.numel() > 0:
            MooncakeBaseStore.register_buffer(self, buffer)


def _mamba_component_keys(self: MooncakeStore, page_keys: List[str], transfer):
    host_pool = self.registered_pools[transfer.name]
    suffixes = []
    if host_pool.temporal_state_elem_size > 0:
        suffixes.append(f"_{self.mha_suffix}_temporal")
    suffixes.extend(
        f"_{self.mha_suffix}_conv_{index}"
        for index in range(len(host_pool.conv_buffer))
    )
    return [f"{key}{suffix}" for key in page_keys for suffix in suffixes], len(
        suffixes
    )


def _batch_exists_v2(
    self: MooncakeStore,
    keys: List[str],
    pool_transfers: Optional[List[PoolTransfer]] = None,
    extra_info: Optional[HiCacheStorageExtraInfo] = None,
) -> PoolTransferResult:
    kv_pages = self.batch_exists(keys, extra_info)
    hit_count = {PoolName.KV: kv_pages} if kv_pages else {}
    final_pages = kv_pages

    for transfer in pool_transfers or []:
        if final_pages == 0:
            break
        if transfer.name != PoolName.MAMBA:
            raise ValueError(f"Unsupported hybrid pool: {transfer.name}")
        component_keys, multiplier = _mamba_component_keys(self, keys, transfer)
        exists = self._batch_exist(component_keys)
        page_exists = [
            all(
                value == 1
                for value in exists[index * multiplier : (index + 1) * multiplier]
            )
            for index in range(kv_pages)
        ]
        boundary = 0
        if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
            try:
                boundary = page_exists.index(False)
            except ValueError:
                boundary = kv_pages
        elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
            trailing = max(1, len(transfer.keys) if transfer.keys else 1)
            for prefix_len in range(kv_pages, 0, -1):
                if all(
                    page_exists[index]
                    for index in range(max(0, prefix_len - trailing), prefix_len)
                ):
                    boundary = prefix_len
                    break
        if boundary:
            hit_count[transfer.name] = boundary
        final_pages = min(final_pages, boundary)
    return PoolTransferResult(final_pages, hit_count)


def _batch_io_v2(self: MooncakeStore, transfers: List[PoolTransfer], is_set: bool):
    results = {}
    for transfer in transfers:
        if transfer.name != PoolName.MAMBA:
            raise ValueError(f"Unsupported hybrid pool: {transfer.name}")
        host_pool = self.registered_pools[transfer.name]
        keys = transfer.keys or []
        host_indices = transfer.host_indices
        assert keys
        assert len(keys) == len(host_indices) // host_pool.page_size
        component_keys, multiplier = _mamba_component_keys(self, keys, transfer)
        ptrs, sizes = host_pool.get_page_buffer_meta(host_indices)
        if is_set:
            exists = self._batch_exist(component_keys)
            io_results = [0 if value == 1 else -1 for value in exists]
            missing = [index for index, value in enumerate(exists) if value != 1]
            if missing:
                put_results = self._put_batch_zero_copy_impl(
                    [component_keys[index] for index in missing],
                    [ptrs[index] for index in missing],
                    [sizes[index] for index in missing],
                )
                for index, value in zip(missing, put_results):
                    io_results[index] = value
        else:
            io_results = self._get_batch_zero_copy_impl(component_keys, ptrs, sizes)
        # The legacy helper assumes ordinary MHA K/V pairs.  Regroup using the
        # actual number of Mamba components (temporal + conv buffers) instead.
        results[transfer.name] = [
            (
                all(value == 0 for value in group)
                if is_set
                else all(value > 0 for value in group)
            )
            for group in (
                io_results[index : index + multiplier]
                for index in range(0, len(io_results), multiplier)
            )
        ]
    return results


def _batch_get_v2(self: MooncakeStore, transfers, extra_info=None):
    return _batch_io_v2(self, transfers, is_set=False)


def _batch_set_v2(self: MooncakeStore, transfers, extra_info=None):
    return _batch_io_v2(self, transfers, is_set=True)


# Decode-side offload in SGLang 0.5.10 only accepts a plain MHA/MLA pool.
# Qwen3.5 exposes HybridLinearKVPool, so build the same paired KV + Mamba host
# pools used by HiMambaRadixCache and archive both under the same final hash.
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (  # noqa: E402
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.kv_events import OffloadedState  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402


_original_decode_offload_init = DecodeKVCacheOffloadManager.__init__
_original_decode_offload = DecodeKVCacheOffloadManager.offload_kv_cache
_original_decode_check_progress = DecodeKVCacheOffloadManager.check_offload_progress
_original_decode_check_offload = DecodeKVCacheOffloadManager._check_offload_progress
_original_decode_check_backup = DecodeKVCacheOffloadManager._check_backup_progress
_original_decode_release = DecodeKVCacheOffloadManager._release_finished_req
_original_himamba_evict_device_leaf = HiMambaRadixCache._evict_device_leaf
_original_himamba_evict_host_leaf = HiMambaRadixCache._evict_host_leaf
_original_himamba_load_back = HiMambaRadixCache.load_back
_original_himamba_loading_check = HiMambaRadixCache.loading_check


def _evict_host_leaf_skipping_retired_candidate(self, node):
    """Ignore a stale Host eviction index without touching its allocation.

    A candidate can outlive its structural tree attachment when an async
    storage completion races with a radix split/replacement.  The stock path
    frees Host buffers before checking the parent link, which both aborts and
    risks a double free.  Identity-check the link first; a retired candidate
    is removed only from the auxiliary indexes and the eviction loop may
    continue with another leaf.
    """
    if is_structurally_attached(node, self.get_child_key_fn):
        return _original_himamba_evict_host_leaf(self, node)

    self._discard_from_leaf_sets(node)
    if self.mamba_host_lru_list.in_list(node):
        self.mamba_host_lru_list.remove_node(node)
    now = time.monotonic()
    if now >= getattr(self, "_qwen35_stale_host_leaf_log_after", 0.0):
        logger.warning(
            "Discarded retired HiMamba Host eviction candidate without "
            "freeing buffers: node_id=%s",
            node.id,
        )
        self._qwen35_stale_host_leaf_log_after = now + 5.0
    return 0


def _evict_device_leaf_preserving_internal_prefix(self, node):
    """Preserve split radix prefixes during write-through device eviction.

    ``evictable_full_device_leaves`` means that a node has no *device-resident*
    children, not necessarily that ``node.children`` is empty.  A split prefix
    can therefore become eligible after its child is demoted to Host.  The
    stock write-through path tries to delete it as a physical leaf and aborts
    on ``non-leaf``.  Backing up and demoting this uncommon internal node is
    the only semantics-preserving operation: deleting it would orphan the
    Host/Mooncake-resident descendant.
    """
    if needs_internal_prefix_backup(
        backuped=node.backuped,
        child_count=len(node.children),
        write_policy=self.cache_controller.write_policy,
    ):
        needed_kv = len(node.value)
        needed_mamba = int(
            node.mamba_value is not None and node.mamba_host_value is None
        )
        is_tp0 = self.tp_world_size == 1 or torch.distributed.get_rank(
            group=self.tp_group
        ) == 0

        # Reclaim old Host cache before attempting the device->Host demotion.
        # Use a TP-wide MAX so every rank evicts the same requested amount even
        # when asynchronous completions briefly skew local free capacities.
        kv_shortfall = torch.tensor(
            [
                host_capacity_shortfall(
                    available=self.full_kv_pool_host.available_size(),
                    needed=needed_kv,
                )
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                kv_shortfall, op=torch.distributed.ReduceOp.MAX, group=self.tp_group
            )
        if int(kv_shortfall.item()) > 0:
            self.evict_host(int(kv_shortfall.item()))

        # KV Host eviction may also free the associated Mamba Host state, so
        # compute the recurrent-state shortfall after the KV reclaim.
        mamba_shortfall = torch.tensor(
            [
                host_capacity_shortfall(
                    available=self.mamba_pool_host.available_size(),
                    needed=needed_mamba,
                )
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                mamba_shortfall,
                op=torch.distributed.ReduceOp.MAX,
                group=self.tp_group,
            )
        if int(mamba_shortfall.item()) > 0:
            self.evict_mamba_host(int(mamba_shortfall.item()))

        # Host eviction can remove the last descendant of this candidate and
        # cascade through an unprotected device tombstone.  Never operate on a
        # retired object; if it merely became a physical leaf, use the stock
        # write-through leaf eviction instead of needlessly backing it up.
        if not is_structurally_attached(node, self.get_child_key_fn):
            return 0, 0
        if len(node.children) == 0:
            return _original_himamba_evict_device_leaf(self, node)

        # Host protection may still make every old entry temporarily
        # unreclaimable.  In that case fail closed for this candidate and let
        # the eviction heap try another one.
        ready = torch.tensor(
            [
                int(self.full_kv_pool_host.available_size() >= needed_kv),
                int(self.mamba_pool_host.available_size() >= needed_mamba),
            ],
            dtype=torch.int,
        )
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(
                ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        if not bool(torch.all(ready).item()):
            now = time.monotonic()
            if is_tp0 and now >= getattr(
                self, "_qwen35_internal_backup_log_after", 0.0
            ):
                logger.warning(
                    "Deferring split internal radix-prefix eviction because "
                    "Host staging is protected/full: node_id=%s kv=%s mamba=%s",
                    node.id,
                    needed_kv,
                    needed_mamba,
                )
                self._qwen35_internal_backup_log_after = now + 5.0
            # The outer eviction heap will try other candidates.
            return 0, 0

        if is_tp0:
            logger.info(
                "Backing up split internal radix prefix before device eviction: "
                "node_id=%s children=%s",
                node.id,
                len(node.children),
            )
        self.write_backup(node, write_back=True)
        self.writing_check(write_back=True)
        if not node.backuped:
            raise RuntimeError(
                f"Internal radix-prefix backup failed after TP admission: "
                f"node_id={node.id}"
            )
        return self._evict_to_host(node)
    return _original_himamba_evict_device_leaf(self, node)


def _load_back_with_host_source_protection(self, node, mem_quota=None, req=None):
    """Protect Host sources for the complete synchronous restore decision.

    ``load_back`` may evict device KV to make room for the restore.  That
    nested device eviction can itself reclaim Host cache.  Without an explicit
    Host reference, it can select one of this restore's own source nodes and
    leave ``host_value=None`` before the commit loop.  Protect the full chain;
    if an older queued hit was already evicted, report a miss so
    ``init_load_back`` recomputes from the nearest surviving ancestor.
    """
    nodes_to_load = []
    cursor = node
    while cursor.evicted:
        nodes_to_load.insert(0, cursor)
        cursor = cursor.parent

    mamba_required = bool(node.mamba_backuped and node.mamba_evicted)
    local_ready = (
        node.id not in self.ongoing_load_back
        and restore_sources_available(
            kv_sources_present=(
                item.host_value is not None for item in nodes_to_load
            ),
            mamba_required=mamba_required,
            mamba_present=node.mamba_host_value is not None,
        )
    )
    ready = torch.tensor([int(local_ready)], dtype=torch.int)
    if self.tp_world_size > 1:
        torch.distributed.all_reduce(
            ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
    if not bool(ready.item()):
        now = time.monotonic()
        if now >= getattr(self, "_qwen35_stale_restore_log_after", 0.0):
            logger.warning(
                "Treating retired HiMamba Host restore source as a cache miss: "
                "node_id=%s kv_nodes=%s mamba_required=%s",
                node.id,
                len(nodes_to_load),
                mamba_required,
            )
            self._qwen35_stale_restore_log_after = now + 5.0
        return None

    protected = []
    for item in nodes_to_load:
        protect_mamba = item is node and mamba_required
        self._protect_host_node(item, protect_mamba=protect_mamba)
        protected.append((item, protect_mamba))
    if mamba_required and node not in nodes_to_load:
        self._protect_host_node(node, protect_mamba=True)
        protected.append((node, True))

    try:
        result = _original_himamba_load_back(self, node, mem_quota, req=req)
        if result is not None and node.id in self.ongoing_load_back:
            pending = getattr(self, "_qwen35_load_host_protections", None)
            if pending is None:
                pending = self._qwen35_load_host_protections = {}
            if node.id in pending:
                raise RuntimeError(
                    f"duplicate Host restore protection for node_id={node.id}"
                )
            pending[node.id] = protected
            protected = []
        return result
    finally:
        for item, release_mamba in reversed(protected):
            self._release_host_node(item, release_mamba=release_mamba)


def _loading_check_releasing_host_sources(self):
    """Release restore-source references only after the H2D DMA fence."""
    try:
        return _original_himamba_loading_check(self)
    finally:
        pending = getattr(self, "_qwen35_load_host_protections", None)
        if pending:
            active = set(self.ongoing_load_back)
            for node_id in list(pending):
                if node_id in active:
                    continue
                protected = pending.pop(node_id)
                for item, release_mamba in reversed(protected):
                    self._release_host_node(item, release_mamba=release_mamba)


def _decode_offload_init_hybrid(
    self,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    tp_group,
    tree_cache,
    server_args,
):
    hybrid_kv = token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(hybrid_kv, HybridLinearKVPool):
        return _original_decode_offload_init(
            self,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            tp_group,
            tree_cache,
            server_args,
        )
    if not isinstance(req_to_token_pool, HybridReqToTokenPool):
        raise ValueError("Hybrid decode offload requires HybridReqToTokenPool")

    self.req_to_token_pool = req_to_token_pool
    self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    self.page_size = server_args.page_size
    self.server_args = server_args
    self.request_counter = 0
    self.tree_cache = tree_cache
    env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
    self.offload_stride = (
        self.page_size
        if env_stride is None or env_stride <= 0
        else max(self.page_size, (env_stride // self.page_size) * self.page_size)
    )

    full_kv = hybrid_kv.full_kv_pool
    kv_host_cls = MLATokenToKVPoolHost if hybrid_kv.use_mla else MHATokenToKVPoolHost
    full_host = kv_host_cls(
        full_kv,
        server_args.hicache_ratio,
        server_args.hicache_size,
        self.page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    mamba_host = MambaPoolHost(
        req_to_token_pool.mamba_pool,
        server_args.hicache_ratio,
        server_args.hicache_size,
        allocator_type=server_args.hicache_storage_backend,
        layout=server_args.hicache_mem_layout,
    )
    full_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
    mamba_mapping = dict(req_to_token_pool.mamba_map)
    transfer_layer_num = len(set(full_mapping) | set(mamba_mapping))

    def kv_layer_mapper(layer_id):
        return full_mapping.get(layer_id) if 0 <= layer_id < transfer_layer_num else None

    def mamba_layer_mapper(layer_id):
        return mamba_mapping.get(layer_id) if 0 <= layer_id < transfer_layer_num else None

    host_group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=full_host,
                device_pool=full_kv,
                layer_mapper=kv_layer_mapper,
                is_primary_index_anchor=True,
            ),
            PoolEntry(
                name=PoolName.MAMBA,
                host_pool=mamba_host,
                device_pool=req_to_token_pool.mamba_pool,
                layer_mapper=mamba_layer_mapper,
            ),
        ]
    )
    self.decode_host_mem_pool = host_group
    self._hybrid_mamba_host_pool = mamba_host
    self._qwen35_hybrid_decode_offload = True
    self.tp_group = tp_group
    self.tp_world_size = torch.distributed.get_world_size(group=tp_group)

    extra_config = {}
    if server_args.hicache_storage_backend_extra_config:
        extra_config = json.loads(server_args.hicache_storage_backend_extra_config)
    self.cache_controller = HybridCacheController(
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        mem_pool_host=host_group,
        page_size=self.page_size,
        tp_group=tp_group,
        io_backend=server_args.hicache_io_backend,
        load_cache_event=threading.Event(),
        storage_backend=server_args.hicache_storage_backend,
        model_name=server_args.served_model_name,
        storage_backend_extra_config=extra_config,
        transfer_layer_num=transfer_layer_num,
    )
    req_to_token_pool.register_layer_transfer_counter(
        self.cache_controller.layer_done_counter
    )
    hybrid_kv.register_layer_transfer_counter(self.cache_controller.layer_done_counter)
    self.ongoing_offload = {}
    self.ongoing_backup = {}
    self.offloaded_state = {}
    self.offload_inflight = {}
    self._hybrid_mamba_by_ack = {}
    self._hybrid_offload_admission = HybridOffloadAdmission()
    self._hybrid_capacity_budget = HybridCapacityBudget()
    self._hybrid_capacity_deferrals = 0
    # Capacity retry is token-based rather than wall-clock based so all TP
    # ranks make the same local fast-path decision between consensus points.
    self._hybrid_capacity_retry_after = {}
    logger.info(
        "Enabled run-local hybrid decode offload: stride=%s, KV+Mamba storage",
        self.offload_stride,
    )


def _decode_offload_hybrid(self, req):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_offload(self, req)
    if req.req_pool_idx == -1 or req.req_pool_idx is None or not req.output_ids:
        return False
    if req.mamba_pool_idx is None:
        return False
    # A Mamba state is a large request-wide snapshot.  Keep it admitted until
    # the corresponding Host->Mooncake backup completes; otherwise a long
    # decode can allocate one full snapshot every 64 tokens and outrun the
    # storage worker even though only a bounded number of requests are active.
    admission = self._hybrid_offload_admission
    if not admission.can_submit(req.rid):
        return False
    token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
    if token_indices.dim() == 0 or token_indices.numel() == 0:
        return False

    all_tokens = req.origin_input_ids + req.output_ids[:-1]
    # With the extra-buffer Mamba scheduler, ``mamba_pool_idx`` is the live
    # recurrent state.  The only state that is safe to associate with a radix
    # boundary is the most recently completed ping-pong checkpoint, whose
    # exact token length is recorded in ``mamba_last_track_seqlen``.  Using
    # the live state here stores a newer state under an older hash; a later
    # HiCache restore then resumes from the wrong recurrence and generations
    # degenerate into no-command/max-token outputs at high concurrency.
    checkpoint_len = req.mamba_last_track_seqlen
    if checkpoint_len is None:
        return False
    checkpoint_len = int(checkpoint_len)
    if checkpoint_len > len(all_tokens):
        raise RuntimeError(
            "Mamba checkpoint length exceeds processed token length: "
            f"checkpoint={checkpoint_len} processed={len(all_tokens)} "
            f"request={req.rid}"
        )
    if checkpoint_len < self._hybrid_capacity_retry_after.get(req.rid, 0):
        return False
    prefill_len = len(req.origin_input_ids) // self.page_size * self.page_size
    state = self.offloaded_state.get(req.rid)
    if state is None:
        hashes = self._compute_prefix_hash(req.origin_input_ids[:prefill_len])
        state = OffloadedState(
            prefill_len=prefill_len,
            inc_len=0,
            last_hash=hashes[-1] if hashes else None,
        )
        self.offloaded_state[req.rid] = state
    new_count = checkpoint_len - state.prefill_len - state.inc_len
    # The ping-pong checkpoint is already a real track boundary.  Archive all
    # tokens since the prior durable checkpoint; unlike KV, recurrent state
    # cannot be sliced back to an earlier prefix.
    aligned = exact_mamba_snapshot_length(new_count, self.offload_stride)
    if aligned == 0:
        return False

    start = state.prefill_len + state.inc_len
    end = start + aligned
    incremental_tokens = all_tokens[start:end]
    incremental_indices = token_indices[start:end]

    # Capacity is MIN-reduced once in check_offload_progress(), then consumed
    # deterministically as requests are visited in the same order on all TP
    # ranks.  This preserves collective ordering without one all-reduce per
    # request (which can starve HTTP health handling at c128).
    if not self._hybrid_capacity_budget.reserve(len(incremental_indices), 1):
        self._hybrid_capacity_retry_after[req.rid] = (
            checkpoint_len + self.offload_stride
        )
        self._hybrid_capacity_deferrals += 1
        if self.tp_world_size == 1 or torch.distributed.get_rank(
            group=self.tp_group
        ) == 0:
            if admission.should_log_global_capacity_failure():
                logger.warning(
                    "Hybrid Host staging saturated; deferred %s snapshots "
                    "and will retry after %s tokens",
                    self._hybrid_capacity_deferrals,
                    self.offload_stride,
                )
                self._hybrid_capacity_deferrals = 0
        return False

    self.request_counter += 1
    ack_id = self.request_counter
    # Copy the stable ping-pong checkpoint, never the live recurrent slot.
    # Fence this small request-wide state before the scheduler can reuse that
    # ping-pong slot at a later track boundary.  Attention KV remains on the
    # controller's asynchronous D2H path.
    #
    # Mamba is a single request-level snapshot, so copy and fence only that
    # side pool here.  Attention KV stays on the controller's asynchronous
    # stream and therefore retains the intended transfer/Forward overlap.
    if req.mamba_ping_pong_track_buffer is not None:
        stable_track_idx = stable_mamba_track_index(
            req.mamba_next_track_idx,
            len(req.mamba_ping_pong_track_buffer),
        )
        mamba_device_index = req.mamba_ping_pong_track_buffer[stable_track_idx]
    else:
        mamba_device_index = req.mamba_pool_idx
    mamba_device_indices = mamba_device_index.unsqueeze(0)
    mamba_host_indices = self._hybrid_mamba_host_pool.alloc(1)
    if mamba_host_indices is None:
        raise RuntimeError(
            "Hybrid Mamba host allocation failed after TP capacity consensus "
            f"for request {req.rid}"
        )
    try:
        self._hybrid_mamba_host_pool.backup_from_device_all_layer(
            self.req_to_token_pool.mamba_pool,
            mamba_host_indices,
            mamba_device_indices,
            self.server_args.hicache_io_backend,
        )
        torch.cuda.current_stream().synchronize()
        mamba_transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=mamba_host_indices,
            device_indices=mamba_device_indices,
        )
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )
    except BaseException:
        self._hybrid_mamba_host_pool.free(mamba_host_indices)
        raise
    if host_indices is None:
        self._hybrid_mamba_host_pool.free(mamba_host_indices)
        raise RuntimeError(
            "Hybrid host allocation failed after TP capacity consensus for "
            f"request {req.rid}"
        )

    self._hybrid_capacity_retry_after.pop(req.rid, None)
    admission.mark_submitted(req.rid)
    self._mark_offload_started(req.rid)
    self.ongoing_offload[ack_id] = (
        req,
        host_indices,
        incremental_tokens,
        time.time(),
        start,
        end,
    )
    self._hybrid_mamba_by_ack[ack_id] = mamba_transfer
    state.inc_len += aligned
    return True


def _decode_check_progress_hybrid(self):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_check_progress(self)

    cc = self.cache_controller
    qsizes = torch.tensor(
        [len(cc.ack_write_queue), cc.ack_backup_queue.qsize()],
        dtype=torch.int,
    )
    if self.tp_world_size > 1:
        torch.distributed.all_reduce(
            qsizes, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
    n_write, n_backup = map(int, qsizes.tolist())
    self._check_offload_progress(n_write)
    self._check_backup_progress(n_backup)

    # Progress may have freed Host slots.  Publish one conservative capacity
    # snapshot for all offload decisions until the next scheduler iteration.
    anchor_host = self.decode_host_mem_pool.anchor_entry.host_pool
    capacities = torch.tensor(
        [
            anchor_host.available_size(),
            self._hybrid_mamba_host_pool.available_size(),
        ],
        dtype=torch.int64,
    )
    if self.tp_world_size > 1:
        torch.distributed.all_reduce(
            capacities, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
    self._hybrid_capacity_budget.reset(*map(int, capacities.tolist()))


def _decode_check_offload_hybrid(self, finish_count):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_check_offload(self, finish_count)
    while finish_count > 0:
        ack = self.cache_controller.ack_write_queue.pop(0)
        ack.finish_event.synchronize()
        for ack_id in ack.node_ids:
            req, host_indices, tokens, started, start, _end = self.ongoing_offload.pop(
                ack_id
            )
            mamba_transfer = self._hybrid_mamba_by_ack.pop(ack_id)
            self._mark_offload_finished(req.rid)
            state = self.offloaded_state.get(req.rid)
            prior_hash = state.last_hash if state is not None else None
            page_hashes = self._compute_prefix_hash(tokens, prior_hash)
            mamba_transfer.keys = [page_hashes[-1]]
            mamba_transfer.hit_policy = PoolHitPolicy.TRAILING_PAGES
            backup_id = self.cache_controller.write_storage(
                host_indices,
                tokens,
                hash_value=page_hashes,
                extra_pools=[mamba_transfer],
            )
            self.ongoing_backup[backup_id] = (
                req.rid,
                host_indices,
                started,
                mamba_transfer.host_indices,
            )
            if state is not None:
                state.last_hash = page_hashes[-1]
            if req.finished() and not self._has_inflight_offload(req.rid):
                self._release_finished_req(req, state.prefill_len if state else start)
        finish_count -= 1


def _decode_check_backup_hybrid(self, finish_count):
    if not getattr(self, "_qwen35_hybrid_decode_offload", False):
        return _original_decode_check_backup(self, finish_count)
    for _ in range(finish_count):
        op = self.cache_controller.ack_backup_queue.get()
        rid, host_indices, _started, mamba_host_indices = self.ongoing_backup.pop(
            op.id
        )
        self.decode_host_mem_pool.free(host_indices)
        if mamba_host_indices is not None:
            self._hybrid_mamba_host_pool.free(mamba_host_indices)
        self._hybrid_offload_admission.mark_backup_finished(rid)


def _decode_release_hybrid(self, req, start_offset):
    if (
        getattr(self, "_qwen35_hybrid_decode_offload", False)
        and req.req_pool_idx is not None
        and req.req_pool_idx != -1
        and req.mamba_pool_idx is not None
    ):
        self.req_to_token_pool.free_mamba_cache(req)
        self._hybrid_offload_admission.forget(req.rid)
        self._hybrid_capacity_retry_after.pop(req.rid, None)
    return _original_decode_release(self, req, start_offset)


MambaPoolHost.__init__ = _mamba_init_with_component_sizes
Qwen3_5GatedDeltaNet.fix_query_key_value_ordering = (
    _fix_query_key_value_ordering_with_contiguous_gates
)
HiMambaRadixCache._evict_device_leaf = (
    _evict_device_leaf_preserving_internal_prefix
)
HiMambaRadixCache._evict_host_leaf = (
    _evict_host_leaf_skipping_retired_candidate
)
HiMambaRadixCache.load_back = _load_back_with_host_source_protection
HiMambaRadixCache.loading_check = _loading_check_releasing_host_sources
MambaPoolHost._copy_tensor_all_layers_lf_pf = staticmethod(
    _copy_tensor_all_layers_lf_pf_with_cuda_dst_indices
)
MambaPoolHost._copy_tensor_pf_lf = staticmethod(
    _copy_tensor_pf_lf_with_cuda_src_indices
)
MambaPoolHost.get_hybrid_pool_buffer = _mamba_get_hybrid_pool_buffer
MambaPoolHost.get_page_buffer_meta = _mamba_get_page_buffer_meta
MooncakeStore._put_batch_zero_copy_impl = _serialized_put_batch_zero_copy_impl
MooncakeStore._get_batch_zero_copy_impl = _serialized_get_batch_zero_copy_impl
MooncakeStore._batch_exist = _serialized_batch_exist
MooncakeStore.register_mem_pool_host = _register_mem_pool_host
MooncakeStore.register_mem_host_pool_v2 = _register_mem_host_pool_v2
MooncakeStore.batch_exists_v2 = _batch_exists_v2
MooncakeStore.batch_get_v2 = _batch_get_v2
MooncakeStore.batch_set_v2 = _batch_set_v2
DecodeKVCacheOffloadManager.__init__ = _decode_offload_init_hybrid
DecodeKVCacheOffloadManager.offload_kv_cache = _decode_offload_hybrid
DecodeKVCacheOffloadManager.check_offload_progress = _decode_check_progress_hybrid
DecodeKVCacheOffloadManager._check_offload_progress = _decode_check_offload_hybrid
DecodeKVCacheOffloadManager._check_backup_progress = _decode_check_backup_hybrid
DecodeKVCacheOffloadManager._release_finished_req = _decode_release_hybrid

logger.warning("Loaded run-local Qwen3.5 hybrid Mooncake compatibility layer")
