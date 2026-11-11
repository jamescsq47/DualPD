from __future__ import annotations

from contextlib import nullcontext


class _NoopSpan:
    def set(self, key, value):
        return self

    def update(self, attrs):
        return self


def span(args, target, name: str, attrs: dict | None = None):
    if not getattr(args, "use_slime_dashboard", False):
        return nullcontext(_NoopSpan())
    from dualpd.runtime.observation.trace_utils import trace_span

    return trace_span(target, name, attrs=attrs)


def event(args, target, name: str, attrs: dict | None = None) -> None:
    if not getattr(args, "use_slime_dashboard", False):
        return
    from dualpd.runtime.observation.trace_utils import trace_event

    trace_event(target, name, attrs=attrs)

