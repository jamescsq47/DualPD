from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/pd/scripts/compat/qwen35_mooncake/io_serialization.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_mooncake_io_serialization", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def object_with_attributes() -> object:
    class Store:
        pass

    return Store()


def test_store_calls_are_serialized_per_instance() -> None:
    store = object_with_attributes()
    barrier = threading.Barrier(9)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def operation(_store: object) -> None:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with state_lock:
            active -= 1

    def worker() -> None:
        barrier.wait()
        MODULE.serialized_store_call(store, operation)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert max_active == 1


def test_different_store_instances_remain_parallel() -> None:
    stores = [object_with_attributes(), object_with_attributes()]
    start = threading.Barrier(3)
    entered = threading.Barrier(2)

    def operation(_store: object) -> None:
        entered.wait(timeout=1)

    def worker(store: object) -> None:
        start.wait()
        MODULE.serialized_store_call(store, operation)

    threads = [threading.Thread(target=worker, args=(store,)) for store in stores]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
