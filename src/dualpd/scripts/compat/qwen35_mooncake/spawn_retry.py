"""Retry transient ``posix_spawn`` resource failures.

At high SWE-bench concurrency a node can briefly have thousands of verifier
threads while Triton lazily invokes ``ptxas``.  Linux may then return EAGAIN
from ``posix_spawn`` even though neither the Slurm cgroup nor RLIMIT_NPROC is
exhausted.  Retrying only that errno keeps the compiler launch fail-closed for
every permanent error and does not change successful spawn behavior.
"""

from __future__ import annotations

import errno
import subprocess
import time
from typing import Any


_PATCH_MARKER = "_dualpd_eagain_retry_installed"


def install_posix_spawn_eagain_retry(
    *,
    popen_cls: type = subprocess.Popen,
    max_attempts: int = 40,
    initial_delay_seconds: float = 0.025,
    max_delay_seconds: float = 0.25,
) -> bool:
    """Install a bounded EAGAIN-only retry around ``Popen._posix_spawn``.

    Returns ``True`` when this call installs the wrapper and ``False`` when a
    wrapper is already installed.  The worst-case added wait is under ten
    seconds with the defaults; exhaustion re-raises the original exception.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if initial_delay_seconds < 0 or max_delay_seconds < 0:
        raise ValueError("retry delays must be non-negative")

    original = popen_cls._posix_spawn
    if getattr(original, _PATCH_MARKER, False):
        return False

    def _posix_spawn_with_eagain_retry(self: Any, *args: Any, **kwargs: Any):
        delay = initial_delay_seconds
        for attempt in range(max_attempts):
            try:
                return original(self, *args, **kwargs)
            except BlockingIOError as error:
                if error.errno != errno.EAGAIN or attempt + 1 >= max_attempts:
                    raise
                time.sleep(delay)
                delay = min(max_delay_seconds, max(delay * 2, initial_delay_seconds))

        raise AssertionError("unreachable posix_spawn retry state")

    setattr(_posix_spawn_with_eagain_retry, _PATCH_MARKER, True)
    setattr(_posix_spawn_with_eagain_retry, "_dualpd_original", original)
    popen_cls._posix_spawn = _posix_spawn_with_eagain_retry
    return True

