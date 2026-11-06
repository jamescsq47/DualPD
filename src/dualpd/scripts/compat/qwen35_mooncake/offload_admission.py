"""Small, dependency-free admission gate for hybrid decode offload.

Mamba state is a request snapshot rather than token-indexed KV.  Letting a
request enqueue another snapshot while its previous snapshot is still being
written to storage consumes one large Host slot per decode stride and can
exhaust the pool long before Mooncake catches up.
"""

from __future__ import annotations

import time


def exact_mamba_snapshot_length(new_count: int, stride: int) -> int:
    """Return a snapshot length only at the state object's true boundary.

    KV can be sliced to a smaller aligned prefix, but a request-wide Mamba
    state always represents every token processed so far.  Flooring a delayed
    snapshot to an older boundary therefore associates new state with an old
    hash and corrupts a later restore.
    """
    new_count = int(new_count)
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    if new_count < stride or new_count % stride != 0:
        return 0
    return new_count


def stable_mamba_track_index(next_track_index: int, buffer_size: int) -> int:
    """Return the completed ping-pong slot, not the next writable slot."""
    next_track_index = int(next_track_index)
    buffer_size = int(buffer_size)
    if buffer_size == 1:
        if next_track_index != 0:
            raise ValueError("single Mamba track buffer requires index 0")
        return 0
    if buffer_size == 2:
        if next_track_index not in (0, 1):
            raise ValueError("ping-pong Mamba track index must be 0 or 1")
        return 1 - next_track_index
    raise ValueError("Mamba track buffer size must be 1 or 2")


class HybridOffloadAdmission:
    """Allow at most one D2H->storage snapshot per request at a time."""

    def __init__(self, log_interval_s: float = 5.0):
        self.log_interval_s = log_interval_s
        self._busy: set[str] = set()
        self._log_after: dict[str, float] = {}
        self._global_capacity_log_after = 0.0

    def can_submit(self, rid: str) -> bool:
        return rid not in self._busy

    def mark_submitted(self, rid: str) -> None:
        self._busy.add(rid)

    def mark_backup_finished(self, rid: str) -> None:
        self._busy.discard(rid)

    def should_log_capacity_failure(
        self, rid: str, now: float | None = None
    ) -> bool:
        """Return true at most once per log interval for this request."""
        current = time.monotonic() if now is None else now
        should_log = current >= self._log_after.get(rid, 0.0)
        if should_log:
            self._log_after[rid] = current + self.log_interval_s
        return should_log

    def should_log_global_capacity_failure(
        self, now: float | None = None
    ) -> bool:
        """Rate-limit a saturated-pool message across all requests.

        A per-request limiter still produces O(batch size) log traffic when a
        c128 decode batch reaches the Host staging limit.  Capacity is a pool
        property, so one message per interval is both more useful and much
        cheaper.
        """
        current = time.monotonic() if now is None else now
        should_log = current >= self._global_capacity_log_after
        if should_log:
            self._global_capacity_log_after = current + self.log_interval_s
        return should_log

    def forget(self, rid: str) -> None:
        self._busy.discard(rid)
        self._log_after.pop(rid, None)

    def is_busy(self, rid: str) -> bool:
        return rid in self._busy


class HybridCapacityBudget:
    """TP-consistent Host capacity snapshot for one scheduler iteration.

    All TP ranks refresh this budget from the same MIN-reduced capacities and
    visit requests in the same order.  Local reservations are therefore
    deterministic without a collective for every request.
    """

    def __init__(self) -> None:
        self.kv_slots = 0
        self.mamba_slots = 0

    def reset(self, kv_slots: int, mamba_slots: int) -> None:
        self.kv_slots = max(0, int(kv_slots))
        self.mamba_slots = max(0, int(mamba_slots))

    def reserve(self, kv_slots: int, mamba_slots: int = 1) -> bool:
        kv_slots = int(kv_slots)
        mamba_slots = int(mamba_slots)
        if kv_slots > self.kv_slots or mamba_slots > self.mamba_slots:
            return False
        self.kv_slots -= kv_slots
        self.mamba_slots -= mamba_slots
        return True


def needs_internal_prefix_backup(
    *, backuped: bool, child_count: int, write_policy: str
) -> bool:
    """Return whether eviction must preserve an internal radix prefix.

    HiMambaRadixCache treats a node whose children are all Host-resident as a
    device leaf.  Such a node can still have structural children and cannot be
    deleted by the ordinary write-through leaf path; it first needs a Host
    backup so the prefix tree remains connected.
    """
    return not backuped and child_count > 0 and write_policy == "write_through"


def host_capacity_shortfall(*, available: int, needed: int) -> int:
    """Return the exact number of Host slots that must be reclaimed.

    A full Host tier is normal cache pressure rather than a fatal device OOM.
    The caller TP-reduces this local shortfall before evicting old Host entries.
    """
    return max(0, int(needed) - int(available))


def restore_sources_available(
    *, kv_sources_present, mamba_required: bool, mamba_present: bool
) -> bool:
    """Return whether a queued Host hit still has every physical source.

    A request can retain a radix ``last_node`` after ordinary LRU pressure has
    evicted that Host entry.  Such a stale hit is a cache miss and must fall
    back to recomputation instead of dereferencing a released buffer.
    """
    return all(bool(value) for value in kv_sources_present) and (
        not mamba_required or bool(mamba_present)
    )


def is_structurally_attached(node, get_child_key_fn) -> bool:
    """Return whether ``node`` is still the exact child recorded by its parent.

    HiCache eviction candidates are kept in auxiliary sets/LRU lists.  Tree
    splits and asynchronous storage completion can leave a retired object in
    one of those indexes after its parent slot has been replaced.  Such an
    object must be removed from the index, but its buffers must not be freed a
    second time.
    """
    parent = getattr(node, "parent", None)
    key = getattr(node, "key", None)
    if parent is None or key is None:
        return False
    return parent.children.get(get_child_key_fn(key)) is node
