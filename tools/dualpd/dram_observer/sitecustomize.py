"""Opt-in, run-local Host-memory observation. No cache policy is changed.

The import hook registers constructed pools. A daemon reads counters every two
seconds; it never waits for an allocator lock or calls a lifecycle method.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time
import weakref


TARGETS = {
    "sglang.srt.mem_cache.memory_pool_host": ("HostKVCache",),
    "sglang.srt.disaggregation.agentic_host_staging": ("SharedHostSnapshotArena",),
    "sglang.srt.disaggregation.p2d_host_staging": ("_RegisteredP2DHostArena",),
}
_pools = {}
_thread_pid = None


def argument(name, default=None):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def pool_sample(obj, metadata):
    """Read Python counters only; do not acquire locks or touch tensor data."""
    if metadata["kind"] == "hicache":
        total_tokens = int(obj.size)
        free_tokens = len(obj.free_slots)
        per_token = int(obj.size_per_token)
        used = (total_tokens - free_tokens) * per_token
        capacity = total_tokens * per_token
        extra = dict(used_tokens=total_tokens - free_tokens,
                     capacity_tokens=total_tokens, bytes_per_token=per_token)
    else:
        used, capacity = int(obj.used_bytes), int(obj.capacity_bytes)
        extra = dict(committed_bytes=getattr(obj, "committed_bytes", None),
                     closed=bool(getattr(obj, "_closed", False)))
    if not 0 <= used <= capacity:
        raise ValueError(f"inconsistent counter sample: used={used}, capacity={capacity}")
    return {**metadata, **extra, "used_bytes": used, "capacity_bytes": capacity}


def snapshot():
    records, errors = [], []
    for key, (ref, metadata) in tuple(_pools.items()):
        obj = ref()
        if obj is None:
            _pools.pop(key, None)
            continue
        try:
            records.append(pool_sample(obj, metadata))
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    return dict(ts=time.time(), monotonic=time.monotonic(), pid=os.getpid(),
                role=argument("--disaggregation-mode"), port=argument("--port"),
                engine=os.environ.get("SGLANG_AGENTIC_KV_ENGINE_ID"),
                pools=records, errors=errors)


def writer():
    directory = Path(os.environ["PD_DRAM_TRACE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    latest = directory / f"{pid}.latest.json"
    temporary = directory / f"{pid}.latest.tmp"
    with (directory / f"{pid}.jsonl").open("a", buffering=1) as stream:
        while os.getpid() == pid:
            try:
                payload = json.dumps(snapshot(), separators=(",", ":"))
                stream.write(payload + "\n")
                temporary.write_text(payload + "\n")
                os.replace(temporary, latest)
            except Exception as exc:
                # Observation failure must not affect allocator ownership.
                print(f"DRAM_OBSERVER_ERROR pid={pid}: {exc}", file=sys.stderr)
            time.sleep(float(os.environ.get("PD_DRAM_SAMPLE_SECONDS", "2")))


def register(obj, cls_name, args, kwargs):
    global _thread_pid
    kind = "hicache" if cls_name == "HostKVCache" else "arena"
    directory = str(kwargs.get("directory", args[0] if args and kind == "arena" else ""))
    direction = None if kind == "hicache" else ("p2d" if "p2d" in directory else "d2p")
    metadata = dict(kind=kind, direction=direction, class_name=cls_name,
                    object_id=id(obj), backing=getattr(obj, "path", None))
    _pools[id(obj)] = (weakref.ref(obj), metadata)
    if _thread_pid != os.getpid():
        _thread_pid = os.getpid()
        threading.Thread(target=writer, name="dram-observer", daemon=True).start()


def instrument(module):
    for name in TARGETS[module.__name__]:
        cls = getattr(module, name, None)
        if cls is None or getattr(cls.__init__, "_dram_observed", False):
            continue
        original = cls.__init__

        def make_wrapper(original, name):
            @functools.wraps(original)
            def init(self, *args, **kwargs):
                original(self, *args, **kwargs)
                if argument("--disaggregation-mode") in {"prefill", "decode"}:
                    try:
                        register(self, name, args, kwargs)
                    except Exception as exc:
                        # An observer failure must not turn successful pool
                        # construction into a lifecycle/ownership failure.
                        print(f"DRAM_OBSERVER_REGISTER_ERROR: {exc}", file=sys.stderr)
            init._dram_observed = True
            return init

        cls.__init__ = make_wrapper(original, name)


class ObservingLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        instrument(module)


class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = ObservingLoader(spec.loader)
        return spec


if os.environ.get("PD_DRAM_TRACE_DIR"):
    sys.meta_path.insert(0, ObserverFinder())
