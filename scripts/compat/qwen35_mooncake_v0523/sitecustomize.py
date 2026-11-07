"""Run-local Qwen3.8 hybrid decode-offload bridge for SGLang v0.5.23.

The v0.5.21 compatibility layer already carries the request-lifecycle,
stable-Mamba-checkpoint, storage-hash, and TP-workset fixes used by the
validated Qwen3.8 runs.  SGLang v0.5.23 changed its host-pool construction
API, so load those narrow hooks and replace only the hybrid decode-manager
initializer with the current native hybrid-Mamba stack builder.

This module is injected through the experiment manifest's PYTHONPATH.  It
does not modify the SGLang checkout or the shared baseline environment.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from weakref import WeakKeyDictionary

import torch


_HERE = Path(__file__).resolve().parent
_V0521_DIR = _HERE.parent / "qwen35_mooncake_v0521"
if str(_V0521_DIR) not in sys.path:
    sys.path.insert(0, str(_V0521_DIR))

_BASE_SPEC = importlib.util.spec_from_file_location(
    "_qwen35_mooncake_v0521_base", _V0521_DIR / "sitecustomize.py"
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError("cannot load the validated v0.5.21 compatibility hooks")
_base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_base)

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (  # noqa: E402
    DecodeKVCacheOffloadManager,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.disaggregation.kv_events import OffloadedState  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import (  # noqa: E402
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (  # noqa: E402
    build_hybrid_mamba_stack,
)
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer  # noqa: E402
from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_memory,
    get_schedule,
    get_serving,
)


logger = logging.getLogger(__name__)

_original_check_offload_progress = (
    DecodeKVCacheOffloadManager.check_offload_progress
)


def _init_hybrid_decode_offload_v0523(
    self,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    tp_group,
    tree_cache,
):
    """Initialize FULL+MAMBA decode offload with the v0.5.23 builders."""
    hybrid_kv = token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(hybrid_kv, HybridLinearKVPool):
        return _base._original_init(
            self,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            tp_group,
            tree_cache,
        )
    if not isinstance(req_to_token_pool, HybridReqToTokenPool):
        raise ValueError("Hybrid decode offload requires HybridReqToTokenPool")

    memory = get_memory()
    self.req_to_token_pool = req_to_token_pool
    self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    self.page_size = get_schedule().page_size
    self.request_counter = 0
    self.tree_cache = tree_cache
    self.kv_cache = hybrid_kv
    env_stride = envs.SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE.get()
    self.offload_stride = (
        self.page_size
        if env_stride is None or env_stride <= 0
        else max(self.page_size, (env_stride // self.page_size) * self.page_size)
    )
    self.tp_group = tp_group
    self.tp_world_size = torch.distributed.get_world_size(group=tp_group)

    extra_config = {}
    if memory.hicache_storage_backend_extra_config:
        try:
            extra_config = json.loads(memory.hicache_storage_backend_extra_config)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid hicache storage backend extra config JSON: {exc}"
            ) from exc

    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=self.page_size,
        tp_cache_group=tp_group,
    )
    full_mapping = {
        global_layer - hybrid_kv.start_layer: pool_layer
        for global_layer, pool_layer in
        hybrid_kv.full_attention_layer_id_mapping.items()
    }
    host_group, controller, _config = build_hybrid_mamba_stack(
        params=params,
        decls=hybrid_kv.host_pool_decls(),
        mamba_pool=req_to_token_pool.mamba_pool,
        full_layer_mapping=full_mapping,
        start_layer=hybrid_kv.start_layer,
        load_cache_event=threading.Event(),
        storage_backend=memory.hicache_storage_backend,
        use_mla=hybrid_kv.use_mla,
        model_name=get_serving().served_model_name,
        storage_backend_extra_config=extra_config,
    )

    # The hybrid builder owns a unified transfer-layer counter.  Both the KV
    # and recurrent-state pools must observe it before any D2H transfer starts.
    hybrid_kv.register_layer_transfer_counter(controller.layer_done_counter)
    req_to_token_pool.register_layer_transfer_counter(controller.layer_done_counter)

    self.decode_host_mem_pool = host_group
    self.cache_controller = controller
    self.ongoing_offload = {}
    self.ongoing_backup = {}
    self.offloaded_state = WeakKeyDictionary()
    self.offload_inflight = WeakKeyDictionary()
    self._qwen35_hybrid_decode = True
    self._qwen35_active_mamba_indices = None
    self._qwen38_backup_busy = WeakKeyDictionary()
    self._qwen38_backup_owners = {}
    # Refreshed, and TP-MIN reduced, by check_offload_progress().
    self._qwen38_host_capacity = [0, 0]
    logger.warning(
        "Enabled v0.5.23 Qwen3.8 hybrid decode offload bridge: "
        "KV=%.1f GiB Mamba=%.1f GiB per rank",
        memory.hicache_size,
        float(
            os.environ.get(
                "SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB", memory.hicache_size
            )
        ),
    )


def _offload_hybrid_mamba_v0523(self, req):
    """Archive KV only at an exact, synchronously protected Mamba checkpoint."""
    if not getattr(self, "_qwen35_hybrid_decode", False):
        return _base._original_offload(self, req)
    if req.kv.req_pool_idx in (None, -1) or not req.output_ids:
        return False
    # Keep at most one host/storage snapshot per request.  Without this gate a
    # long request consumes one large Mamba host slot every track interval.
    if req in self._qwen38_backup_busy:
        return False

    checkpoint_len = req.kv.mamba_last_track_seqlen
    track_buffer = req.kv.mamba_ping_pong_track_buffer
    track_index = req.kv.mamba_last_track_idx
    if checkpoint_len is None or track_buffer is None or track_index is None:
        return False

    token_indices = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
    if token_indices.dim() == 0 or token_indices.numel() == 0:
        return False
    all_tokens = req.origin_input_ids + req.output_ids[:-1]
    checkpoint_len = int(checkpoint_len)
    if checkpoint_len > len(all_tokens):
        raise RuntimeError(
            "Mamba checkpoint exceeds processed token length: "
            f"checkpoint={checkpoint_len} processed={len(all_tokens)} "
            f"request={req.rid}"
        )

    prefill_len = self._prefill_offloaded_len(req)
    state = self.offloaded_state.get(req)
    if state is None:
        prefill_hashes = self._compute_prefix_hash(
            req, req.origin_input_ids[:prefill_len]
        )
        state = OffloadedState(
            last_hash=prefill_hashes[-1] if prefill_hashes else None
        )
        self.offloaded_state[req] = state
    incremental_len = checkpoint_len - prefill_len - state.inc_len
    if incremental_len < self.offload_stride or incremental_len % self.offload_stride:
        return False

    kv_capacity, mamba_capacity = self._qwen38_host_capacity
    if incremental_len > kv_capacity or mamba_capacity < 1:
        return False
    self._qwen38_host_capacity[0] -= incremental_len
    self._qwen38_host_capacity[1] -= 1

    start = prefill_len + state.inc_len
    incremental_tokens = all_tokens[start:checkpoint_len]
    incremental_indices = token_indices[start:checkpoint_len].to(torch.int64)
    mamba_device_indices = track_buffer[track_index].reshape(1)
    mamba_host_pool = self.decode_host_mem_pool.get_pool(PoolName.MAMBA)
    mamba_host_indices = mamba_host_pool.alloc(1)
    if mamba_host_indices is None:
        self._qwen38_host_capacity[0] += incremental_len
        self._qwen38_host_capacity[1] += 1
        return False

    entry = self.decode_host_mem_pool.get_entry(PoolName.MAMBA)
    try:
        # Going through L2TransferEngine is essential: Unified Mamba slots are
        # virtual IDs, translated under the move gate immediately before D2H.
        completion = self.cache_controller.l2_transfer_engine.submit_device_to_host(
            [
                L2Transfer(
                    host_pool=entry.host_pool,
                    device_pool=entry.device_pool,
                    host_indices=mamba_host_indices,
                    device_indices=mamba_device_indices,
                    layer_mapper=entry.layer_mapper,
                )
            ]
        )
        # Fence only the one request-wide Mamba snapshot.  Attention KV keeps
        # using the controller's asynchronous D2H path.
        completion.finish_event.synchronize()
        self.request_counter += 1
        ack_id = self.request_counter
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices,
            node_id=ack_id,
        )
    except BaseException:
        mamba_host_pool.free(mamba_host_indices)
        self._qwen38_host_capacity[0] += incremental_len
        self._qwen38_host_capacity[1] += 1
        raise
    if host_indices is None:
        mamba_host_pool.free(mamba_host_indices)
        self._qwen38_host_capacity[0] += incremental_len
        self._qwen38_host_capacity[1] += 1
        return False

    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=mamba_host_indices,
        device_indices=mamba_device_indices,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    self._qwen38_backup_busy[req] = True
    self._mark_offload_started(req)
    self.ongoing_offload[ack_id] = (
        req,
        host_indices,
        incremental_tokens,
        time.time(),
        [mamba_transfer],
    )
    state.inc_len += incremental_len
    return True


def _resolve_offload_transfers_v0523(self, virtual_indices):
    """Preserve the native Unified-SWA path outside the Qwen hybrid branch."""
    full_indices = virtual_indices
    if not isinstance(self.kv_cache, UnifiedSWAKVPool):
        return full_indices, []
    swa_indices = self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
        virtual_indices
    )
    live_swa_indices = swa_indices[swa_indices > 0].to(torch.int64)
    if live_swa_indices.numel() == 0:
        return full_indices, []
    return full_indices, [
        PoolTransfer(
            name=PoolName.SWA,
            device_indices=live_swa_indices,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]


def _check_offload_progress_v0523(self):
    if not getattr(self, "_qwen35_hybrid_decode", False):
        return _original_check_offload_progress(self)
    cc = self.cache_controller
    progress = torch.tensor(
        [
            len(cc.ack_write_queue),
            cc.ack_backup_queue.qsize(),
            self.decode_host_mem_pool.anchor_entry.host_pool.available_size(),
            self.decode_host_mem_pool.get_pool(PoolName.MAMBA).available_size(),
        ],
        dtype=torch.int64,
    )
    if self.tp_world_size > 1:
        torch.distributed.all_reduce(
            progress, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
    n_write, n_backup, kv_capacity, mamba_capacity = map(
        int, progress.tolist()
    )
    self._check_offload_progress(n_write)
    self._check_backup_progress(n_backup)
    self._qwen38_host_capacity[:] = [kv_capacity, mamba_capacity]


def _trigger_backup_v0523(
    self,
    req,
    host_indices,
    incremental_tokens,
    start_time,
    prior_hash,
    pool_transfers=(),
):
    page_hashes = self._compute_prefix_hash(req, incremental_tokens, prior_hash)
    for transfer in pool_transfers:
        if transfer.name == PoolName.MAMBA:
            transfer.keys = [page_hashes[-1]]
            transfer.hit_policy = PoolHitPolicy.TRAILING_PAGES
        else:
            page_count = transfer.device_indices.numel() // self.page_size
            transfer.keys = page_hashes[-page_count:]
    ack_id = self.cache_controller.write_storage(
        host_indices,
        incremental_tokens,
        hash_value=page_hashes,
        **({"extra_pools": pool_transfers} if pool_transfers else {}),
    )
    self.ongoing_backup[ack_id] = (
        req.rid,
        host_indices,
        start_time,
        pool_transfers,
    )
    if getattr(self, "_qwen35_hybrid_decode", False):
        self._qwen38_backup_owners[ack_id] = req
    return page_hashes[-1] if page_hashes else prior_hash


def _check_backup_progress_v0523(self, finish_count):
    for _ in range(finish_count):
        storage_operation = self.cache_controller.ack_backup_queue.get()
        ack_id = storage_operation.id
        req_id, host_indices, start_time, pool_transfers = self.ongoing_backup.pop(
            ack_id
        )
        owner = getattr(self, "_qwen38_backup_owners", {}).pop(ack_id, None)
        if owner is not None:
            self._qwen38_backup_busy.pop(owner, None)
        for transfer in pool_transfers:
            self.decode_host_mem_pool.get_pool(transfer.name).free(
                transfer.host_indices
            )
        self.decode_host_mem_pool.free(host_indices)
        logger.debug(
            "Finished backup request %s, free host memory, len:%d, "
            "cost time:%.2f seconds.",
            req_id,
            len(host_indices),
            time.time() - start_time,
        )


DecodeKVCacheOffloadManager.__init__ = _init_hybrid_decode_offload_v0523
DecodeKVCacheOffloadManager.offload_kv_cache = _offload_hybrid_mamba_v0523
DecodeKVCacheOffloadManager._resolve_offload_transfers = (
    _resolve_offload_transfers_v0523
)
DecodeKVCacheOffloadManager.check_offload_progress = _check_offload_progress_v0523
DecodeKVCacheOffloadManager._trigger_backup = _trigger_backup_v0523
DecodeKVCacheOffloadManager._check_backup_progress = _check_backup_progress_v0523

logger.warning(
    "Loaded run-local Qwen3.8 Mooncake compatibility for SGLang v0.5.23 "
    "using the native hybrid-Mamba host-pool builder"
)
