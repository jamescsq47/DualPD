"""Process-local serialization for Mooncake store calls.

The Mooncake Python binding used by the Qwen3.5 baseline can abort the whole
process when one store instance executes a zero-copy put and get concurrently.
Keep the lock on the store object so unrelated worker processes remain fully
parallel while every native operation on one C++ store is mutually exclusive.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar


T = TypeVar("T")
_LOCK_ATTRIBUTE = "_qwen35_mooncake_native_io_lock"
_LOCK_CREATION_GUARD = threading.Lock()


def store_io_lock(store: Any) -> threading.RLock:
    lock = getattr(store, _LOCK_ATTRIBUTE, None)
    if lock is not None:
        return lock
    with _LOCK_CREATION_GUARD:
        lock = getattr(store, _LOCK_ATTRIBUTE, None)
        if lock is None:
            lock = threading.RLock()
            setattr(store, _LOCK_ATTRIBUTE, lock)
    return lock


def serialized_store_call(
    store: Any, operation: Callable[..., T], *args: Any, **kwargs: Any
) -> T:
    with store_io_lock(store):
        return operation(store, *args, **kwargs)

