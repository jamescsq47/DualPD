"""Narrow Qwen3.5 decode-offload bridge for stock SGLang v0.5.21.

SGLang v0.5.21 already implements hybrid KV+Mamba HiCache/Mooncake on the
prefill side.  Its DecodeKVCacheOffloadManager, however, rejects the model's
HybridLinearKVPool before it can use the same native HostPoolGroup machinery.
This run-local shim wires that missing case without changing SGLang sources.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from http import HTTPStatus
from weakref import WeakKeyDictionary

import torch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation import utils as _decode_utils
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as _assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_kv_host_pool,
    build_pool_entry,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.runtime_context import get_memory, get_schedule, get_serving
from tp_workset_consensus import resolve_identity_workset, resolve_transfer_workset


logger = logging.getLogger(__name__)

_original_init = DecodeKVCacheOffloadManager.__init__
_original_offload = DecodeKVCacheOffloadManager.offload_kv_cache
_original_release = DecodeKVCacheOffloadManager._release_finished_req
_original_split_hicache_size = _assembler._split_hicache_size
_original_transfer_poll = DecodeTransferQueue._poll_with_metadata_gate
_original_transfer_pop = DecodeTransferQueue.pop_transferred
_original_handshake_update = DecodePreallocQueue._update_handshake_waiters


def _workset_signature(decode_req) -> int:
    """Return a deterministic positive int64 identity for one transfer room."""
    req = decode_req.req
    material = f"{req.rid}\0{req.bootstrap_room}".encode("utf-8", "surrogatepass")
    digest = hashlib.blake2b(
        material, digest_size=8, person=b"sgl-pd-ws-v1"
    ).digest()
    # Keep the value positive so zero remains the unoccupied-slot sentinel.
    return (int.from_bytes(digest, "little") & ((1 << 63) - 1)) or 1


def _poll_transfer_workset(self):
    """Poll P->D transfers with fixed-size, identity-checked TP consensus."""
    if not torch.distributed.is_initialized():
        return _original_transfer_poll(self)
    world_size = torch.distributed.get_world_size(self.gloo_group)
    if world_size == 1:
        return _original_transfer_poll(self)

    pollers = (
        [HiCacheRestoreGatedKVReceiver(dr) for dr in self.queue]
        if self.scheduler.enable_decode_hicache
        else [dr.kv_receiver for dr in self.queue]
    )
    polls = _decode_utils._poll_with_failure_injection(pollers)
    _decode_utils._apply_metadata_gate(polls, self.queue, self.metadata_buffers)

    # A restore failure is rank-local before this point in upstream v0.5.21.
    # Fold it into the collective poll so all ranks remove/retain together.
    for index, decode_req in enumerate(self.queue):
        if decode_req.hicache_restore_status == HiCacheRestoreResult.FAILED:
            polls[index] = int(KVPoll.Failed)

    capacity = int(self.metadata_buffers.bootstrap_room.shape[0])
    packed = torch.zeros((capacity, 3), dtype=torch.int64, device="cpu")
    local_entries = []
    for decode_req, poll in zip(self.queue, polls):
        metadata_index = int(decode_req.metadata_buffer_index)
        if not 0 <= metadata_index < capacity:
            raise RuntimeError(
                f"invalid P->D metadata slot {metadata_index}; capacity={capacity}"
            )
        signature = _workset_signature(decode_req)
        packed[metadata_index] = torch.tensor(
            (1, signature, int(poll)), dtype=torch.int64
        )
        local_entries.append((metadata_index, signature, int(poll)))

    rank_rows = [torch.empty_like(packed) for _ in range(world_size)]
    torch.distributed.all_gather(rank_rows, packed, group=self.gloo_group)
    gathered = torch.stack(rank_rows)
    resolved, mismatches = resolve_transfer_workset(
        gathered,
        local_entries,
        waiting_status=int(KVPoll.Bootstrapping),
    )

    now = time.monotonic()
    if mismatches:
        first = getattr(self, "_qwen35_workset_mismatch_since", None)
        if first is None:
            self._qwen35_workset_mismatch_since = now
            self._qwen35_workset_last_log = 0.0
            first = now
        if now - getattr(self, "_qwen35_workset_last_log", 0.0) >= 5.0:
            self._qwen35_workset_last_log = now
            logger.warning(
                "P->D TP workset is not aligned yet: rank=%s local=%d "
                "mismatched=%d age=%.1fs; retaining ownership and waiting",
                self.tp_rank,
                len(local_entries),
                len(mismatches),
                now - first,
            )
        timeout = float(
            os.getenv("SGLANG_QWEN35_TP_WORKSET_MISMATCH_TIMEOUT_S", "30")
        )
        if now - first >= timeout:
            raise RuntimeError(
                "P->D TP workset failed to converge within "
                f"{timeout:.1f}s; refusing a partial-rank KV transition"
            )
    else:
        self._qwen35_workset_mismatch_since = None
    return resolved


def _pop_transfer_with_empty_rank_participation(self, rids_to_check=None):
    """Keep an empty TP rank in the same fixed-shape workset collective."""
    if (
        self.queue
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size(self.gloo_group) == 1
    ):
        return _original_transfer_pop(self, rids_to_check)
    if self.enable_staging:
        return _original_transfer_pop(self, rids_to_check)
    _poll_transfer_workset(self)
    return []


def _update_handshake_workset(
    self,
    rids_to_check=None,
    pp_good_rids=None,
    pp_bad_rids=None,
):
    """Poll handshake state without requiring rank-local queue alignment.

    Stock v0.5.21 reduces a tensor whose length is ``len(self.queue)``.  Request
    delivery is asynchronous, so adjacent TP ranks can legitimately have N and
    N+1 entries and Gloo aborts before the queues converge.  Gather a fixed-size
    identity workset instead; missing peer entries remain Bootstrapping.
    """
    if self.pp_size > 1 or not torch.distributed.is_initialized():
        return _original_handshake_update(
            self, rids_to_check, pp_good_rids, pp_bad_rids
        )
    world_size = torch.distributed.get_world_size(self.gloo_group)
    if world_size == 1:
        return _original_handshake_update(
            self, rids_to_check, pp_good_rids, pp_bad_rids
        )

    capacity = int(self.metadata_buffers.bootstrap_room.shape[0])
    if len(self.queue) > capacity:
        raise RuntimeError(
            f"P->D handshake queue exceeds fixed TP workset capacity: "
            f"{len(self.queue)} > {capacity}"
        )

    local_entries = []
    packed = torch.zeros((capacity, 2), dtype=torch.int64, device="cpu")
    local_polls = _decode_utils._poll_with_failure_injection(
        [decode_req.kv_receiver for decode_req in self.queue]
    )
    for index, (decode_req, poll) in enumerate(zip(self.queue, local_polls)):
        signature = _workset_signature(decode_req)
        poll = int(poll)
        packed[index, 0] = signature
        packed[index, 1] = poll
        local_entries.append((signature, poll))

    rank_rows = [torch.empty_like(packed) for _ in range(world_size)]
    torch.distributed.all_gather(rank_rows, packed, group=self.gloo_group)
    polls, mismatches = resolve_identity_workset(
        torch.stack(rank_rows),
        local_entries,
        waiting_status=int(KVPoll.Bootstrapping),
    )

    now = time.monotonic()
    if mismatches and now - getattr(self, "_qwen35_handshake_last_log", 0.0) >= 5.0:
        self._qwen35_handshake_last_log = now
        logger.info(
            "P->D TP handshake workset converging: rank=%s local=%d missing=%d",
            self.tp_rank,
            len(local_entries),
            len(mismatches),
        )

    for decode_req, poll in zip(self.queue, polls):
        if poll == int(KVPoll.Bootstrapping):
            continue
        if rids_to_check is not None and decode_req.req.rid not in rids_to_check:
            continue
        if poll == int(KVPoll.WaitingForInput):
            decode_req.waiting_for_input = True
            decode_req.req.time_stats.set_bootstrap_done_time()
            continue
        if poll != int(KVPoll.Failed):
            raise ValueError(f"Unexpected poll case: {poll}")

        error_message = (
            f"Decode handshake failed for request rank={self.tp_rank} "
            f"req.rid={decode_req.req.rid} "
            f"req.bootstrap_room={decode_req.req.bootstrap_room}"
        )
        is_propagated = False
        try:
            decode_req.kv_receiver.failure_exception()
        except Exception as exc:
            error_message += f" with exception {exc}"
            is_propagated = getattr(exc, "is_from_another_rank", False)
        (logger.debug if is_propagated else logger.error)(error_message)
        prepare_abort(
            decode_req.req,
            error_message,
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        if self.scheduler.metrics_reporter.enable_metrics:
            self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()


def _split_qwen35_component_sizes(hicache_size, kv_pools):
    """Keep the established independent KV and Mamba per-rank capacities."""
    if len(kv_pools) == 2 and type(kv_pools[1]).__name__ == "MambaPool":
        mamba_size = float(
            os.getenv("SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB", hicache_size)
        )
        return float(hicache_size), mamba_size
    return _original_split_hicache_size(hicache_size, kv_pools)


def _layer_mapper(mapping: dict[int, int], upper_bound: int):
    def mapper(layer_id: int):
        if not 0 <= layer_id < upper_bound:
            return None
        return mapping.get(layer_id)

    return mapper


def _init_hybrid_decode_offload(
    self,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    tp_group,
    tree_cache,
):
    hybrid_kv = token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(hybrid_kv, HybridLinearKVPool):
        return _original_init(
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

    full_kv = hybrid_kv.full_kv_pool
    kv_host = build_kv_host_pool(
        kv_pool=full_kv,
        page_size=self.page_size,
        use_mla=hybrid_kv.use_mla,
        host_size=memory.hicache_size,
    )
    mamba_host_size = float(
        os.getenv("SGLANG_QWEN35_MAMBA_HICACHE_SIZE_GB", memory.hicache_size)
    )
    mamba_host = MambaPoolHost(
        req_to_token_pool.mamba_pool,
        memory.hicache_ratio,
        mamba_host_size,
        allocator_type="default",
        layout=memory.hicache_mem_layout,
    )

    full_mapping = dict(hybrid_kv.full_attention_layer_id_mapping)
    mamba_mapping = dict(req_to_token_pool.mamba_map)
    transfer_layer_id_max = max(full_mapping.keys() | mamba_mapping.keys()) + 1
    host_group = HostPoolGroup(
        [
            build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_host,
                device_pool=full_kv,
                layer_mapping=full_mapping,
                transfer_layer_id_max=transfer_layer_id_max,
                is_anchor=True,
            ),
            build_pool_entry(
                name=PoolName.MAMBA,
                host_pool=mamba_host,
                device_pool=req_to_token_pool.mamba_pool,
                layer_mapping=mamba_mapping,
                transfer_layer_id_max=transfer_layer_id_max,
            ),
        ]
    )
    extra_config = {}
    if memory.hicache_storage_backend_extra_config:
        extra_config = json.loads(memory.hicache_storage_backend_extra_config)
    controller = HybridCacheController(
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        mem_pool_host=host_group,
        page_size=self.page_size,
        tp_group=tp_group,
        load_cache_event=threading.Event(),
        write_policy=memory.hicache_write_policy,
        io_backend=memory.hicache_io_backend,
        storage_backend=memory.hicache_storage_backend,
        model_name=get_serving().served_model_name,
        storage_backend_extra_config=extra_config,
        transfer_layer_id_max=transfer_layer_id_max,
        host_memory_mode=memory.hicache_host_memory_mode,
    )
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
    logger.warning(
        "Enabled v0.5.21 Qwen3.5 hybrid decode offload bridge: "
        "KV=%.1f GiB Mamba=%.1f GiB per rank",
        memory.hicache_size,
        mamba_host_size,
    )


def _offload_at_stable_mamba_checkpoint(self, req):
    if not getattr(self, "_qwen35_hybrid_decode", False):
        return _original_offload(self, req)
    if req.kv.req_pool_idx in (None, -1) or not req.output_ids:
        return False
    checkpoint_len = req.kv.mamba_last_track_seqlen
    track_buffer = req.kv.mamba_ping_pong_track_buffer
    track_index = req.kv.mamba_last_track_idx
    if checkpoint_len is None or track_buffer is None or track_index is None:
        return False

    all_tokens_len = len(req.origin_input_ids) + len(req.output_ids) - 1
    prefill_len = self._prefill_offloaded_len(req)
    state = self.offloaded_state.get(req)
    inc_len = 0 if state is None else state.inc_len
    new_len = all_tokens_len - prefill_len - inc_len
    aligned = new_len // self.offload_stride * self.offload_stride
    if aligned <= 0 or prefill_len + inc_len + aligned != int(checkpoint_len):
        return False

    self._qwen35_active_mamba_indices = track_buffer[track_index].reshape(1)
    try:
        return _original_offload(self, req)
    finally:
        self._qwen35_active_mamba_indices = None


def _resolve_hybrid_transfers(self, device_indices):
    if not getattr(self, "_qwen35_hybrid_decode", False):
        return device_indices, []
    mamba_indices = self._qwen35_active_mamba_indices
    if mamba_indices is None:
        raise RuntimeError("Hybrid decode offload has no stable Mamba checkpoint")
    return device_indices, [
        PoolTransfer(
            name=PoolName.MAMBA,
            device_indices=mamba_indices,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]


def _trigger_hybrid_backup(
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
    return page_hashes[-1] if page_hashes else prior_hash


def _release_hybrid_finished_req(self, req):
    if (
        getattr(self, "_qwen35_hybrid_decode", False)
        and req.kv.req_pool_idx not in (None, -1)
        and req.kv.holds_mamba
    ):
        self.req_to_token_pool.free_mamba_cache(req)
    return _original_release(self, req)


DecodeKVCacheOffloadManager.__init__ = _init_hybrid_decode_offload
DecodeKVCacheOffloadManager.offload_kv_cache = _offload_at_stable_mamba_checkpoint
DecodeKVCacheOffloadManager._resolve_offload_transfers = _resolve_hybrid_transfers
DecodeKVCacheOffloadManager._trigger_backup = _trigger_hybrid_backup
DecodeKVCacheOffloadManager._release_finished_req = _release_hybrid_finished_req
_assembler._split_hicache_size = _split_qwen35_component_sizes
DecodeTransferQueue._poll_with_metadata_gate = _poll_transfer_workset
DecodeTransferQueue.pop_transferred = _pop_transfer_with_empty_rank_participation
DecodePreallocQueue._update_handshake_waiters = _update_handshake_workset

logger.warning(
    "Loaded run-local Qwen3.5 Mooncake compatibility for SGLang v0.5.21 "
    "with fixed-slot TP handshake/transfer workset consensus"
)
