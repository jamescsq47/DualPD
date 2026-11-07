from __future__ import annotations

import errno
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/compat/qwen35_mooncake/spawn_retry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_mooncake_spawn_retry", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fake_popen_class(outcomes):
    class FakePopen:
        calls = 0

        def _posix_spawn(self, *args, **kwargs):
            del args, kwargs
            type(self).calls += 1
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    return FakePopen


def test_retries_eagain_then_returns_success(monkeypatch) -> None:
    popen = fake_popen_class(
        [
            BlockingIOError(errno.EAGAIN, "temporary"),
            BlockingIOError(errno.EAGAIN, "temporary"),
            "started",
        ]
    )
    sleeps = []
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)

    assert MODULE.install_posix_spawn_eagain_retry(
        popen_cls=popen,
        max_attempts=4,
        initial_delay_seconds=0.01,
        max_delay_seconds=0.02,
    )
    assert popen()._posix_spawn() == "started"
    assert popen.calls == 3
    assert sleeps == [0.01, 0.02]


def test_non_eagain_failure_is_not_retried(monkeypatch) -> None:
    popen = fake_popen_class([FileNotFoundError(errno.ENOENT, "missing")])
    sleeps = []
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)
    MODULE.install_posix_spawn_eagain_retry(popen_cls=popen)

    with pytest.raises(FileNotFoundError):
        popen()._posix_spawn()
    assert popen.calls == 1
    assert sleeps == []


def test_retry_exhaustion_reraises_eagain(monkeypatch) -> None:
    popen = fake_popen_class(
        [BlockingIOError(errno.EAGAIN, "temporary") for _ in range(3)]
    )
    sleeps = []
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)
    MODULE.install_posix_spawn_eagain_retry(
        popen_cls=popen,
        max_attempts=3,
        initial_delay_seconds=0,
        max_delay_seconds=0,
    )

    with pytest.raises(BlockingIOError) as failure:
        popen()._posix_spawn()
    assert failure.value.errno == errno.EAGAIN
    assert popen.calls == 3
    assert sleeps == [0, 0]


def test_install_is_idempotent() -> None:
    popen = fake_popen_class(["started"])
    assert MODULE.install_posix_spawn_eagain_retry(popen_cls=popen)
    assert not MODULE.install_posix_spawn_eagain_retry(popen_cls=popen)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_delay_seconds": -0.1},
        {"max_delay_seconds": -0.1},
    ],
)
def test_invalid_retry_configuration_is_rejected(kwargs) -> None:
    popen = fake_popen_class(["unused"])
    with pytest.raises(ValueError):
        MODULE.install_posix_spawn_eagain_retry(popen_cls=popen, **kwargs)
