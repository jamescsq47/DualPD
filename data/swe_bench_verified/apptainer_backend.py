"""Apptainer sandbox transport for SWE-bench task images.

Apptainer instances are intentionally not used here.  Some clusters permit a
plain ``apptainer exec`` but reject the persistent instance mount namespace.
The immutable image is therefore entered by a task-owned command broker, while
the mutable repository, home directory, and /tmp are private host directories
bound into that process.  Commands requiring a newly uploaded root-level bind
fall back to an isolated one-shot exec.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import json
import os
import shutil
import signal
import tempfile
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


_TASK_MARKER = ".pd-swe-apptainer-task"
_START_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_COPY_TESTBED = (
    "set -e; "
    "test -d /testbed; test -e /testbed/.git; "
    # Do not preserve image ownership: the bound workspace must remain owned
    # by the Slurm user even when fakeroot is unavailable.
    "cp -R -P /testbed/. /pd-swe-state/testbed/; "
    "chmod -R u+rwX /pd-swe-state/testbed"
)

# Keep output pipes away from descendants and normalize an actual deadline to
# exit 124.  The new PID namespace is destroyed when this one-shot exec exits,
# so there is no persistent container in which escaped processes can survive.
_TIMEOUT_RUNNER = """
import os, signal, subprocess, sys, tempfile
with tempfile.TemporaryFile() as output:
    p = subprocess.Popen(sys.argv[2:], start_new_session=True,
                         stdout=output, stderr=subprocess.STDOUT)
    try:
        code = p.wait(timeout=float(sys.argv[1]))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        code = 124
    size = os.fstat(output.fileno()).st_size
    for offset in range(0, size, 1024 * 1024):
        sys.stdout.buffer.write(os.pread(output.fileno(), min(size - offset, 1024 * 1024), offset))
sys.exit(code if code >= 0 else 128 - code)
"""

# A long-lived plain ``apptainer exec`` avoids rebuilding the mount namespace
# for every short agent command.  This is not an Apptainer instance and needs
# no instance daemon or privileged operation.  Output is buffered in a file so
# descendants cannot retain the protocol pipe after their shell exits.
_COMMAND_BROKER = r"""
import base64, errno, fcntl, json, os, signal, subprocess, sys, tempfile, time, traceback

def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

send({"ready": True})
for line in sys.stdin:
    try:
        request = json.loads(line)
        started = time.monotonic()
        environment = os.environ.copy()
        environment.update(request.get("environment") or {})
        with tempfile.TemporaryFile() as output:
            # A HiCache worker can temporarily reserve enough virtual memory
            # that a simultaneous burst of verifier forks returns EAGAIN even
            # though the node remains healthy.  Popen has not created a child
            # when it raises BlockingIOError, so retrying this exact operation
            # is side-effect free.  Keep the retry bounded so a real resource
            # leak still surfaces as an infrastructure failure.  At c256 the
            # verifier burst can legitimately keep the per-user process table
            # saturated for longer than 20 seconds, so allow that transient
            # pressure to drain instead of turning completed episodes into
            # environment errors.
            spawn_deadline = time.monotonic() + 120.0
            spawn_retries = 0
            while True:
                try:
                    # All task brokers for one workload share this lock.  Only
                    # serialize the fork/exec transition (not command
                    # execution): a c256 verifier burst otherwise asks the
                    # kernel for 16 process trees at once while TP workers may
                    # also be loading a DeepGEMM JIT image.  Spreading that
                    # short transition avoids transient EAGAIN without
                    # changing verifier or model concurrency.
                    with open("/pd-swe-spawn.lock", "a") as spawn_lock:
                        fcntl.flock(spawn_lock, fcntl.LOCK_EX)
                        process = subprocess.Popen(
                            request["argv"],
                            env=environment,
                            start_new_session=True,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                        )
                        # Give the fakeroot wrapper enough time to establish
                        # its helper before allowing the next process tree.
                        time.sleep(0.05)
                    break
                except BlockingIOError as error:
                    if error.errno != errno.EAGAIN or time.monotonic() >= spawn_deadline:
                        raise
                    spawn_retries += 1
                    time.sleep(min(0.05 * spawn_retries, 0.5))
            try:
                returncode = process.wait(timeout=float(request["timeout"]))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                returncode = int(request["timeout_exit_code"])
            size = os.fstat(output.fileno()).st_size
            contents = bytearray()
            for offset in range(0, size, 1024 * 1024):
                contents.extend(os.pread(output.fileno(), min(size - offset, 1024 * 1024), offset))
        send({
            "returncode": returncode if returncode >= 0 else 128 - returncode,
            "output_b64": base64.b64encode(contents).decode("ascii"),
            "child_seconds": time.monotonic() - started,
            "spawn_retries": spawn_retries,
        })
    except BaseException as error:
        send({"broker_error": repr(error), "traceback": traceback.format_exc()})
"""


def _transient_spawn_failure(output: str) -> bool:
    """Identify Apptainer/fakeroot failures before a sandbox starts."""

    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "resource temporarily unavailable",
            "cannot fork",
            "can't start new thread",
            "cannot start thread",
            "no ld_preload in fakeroot environment",
            "failed to spawn stage 1",
            'container creation failed: "fuseapps": no such image driver',
            "--fakeroot used without sandbox image or user namespaces",
        )
    )


def _bind_value(source: Path, destination: str, mode: str = "rw") -> str:
    """Return a bind spec only when neither side is parser-ambiguous."""

    source_text = str(source)
    if any(character in source_text for character in (",", ":")):
        raise ValueError(f"Apptainer bind source contains ',' or ':': {source_text!r}")
    if any(character in destination for character in (",", ":")):
        raise ValueError(
            f"Apptainer bind destination contains ',' or ':': {destination!r}"
        )
    return f"{source_text}:{destination}:{mode}"


def _safe_container_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.is_absolute() or value == "/" or ".." in path.parts:
        raise ValueError(f"upload path must be a normalized absolute file path: {value!r}")
    return path


def _start_semaphore(limit: int) -> asyncio.Semaphore:
    if limit < 1:
        raise ValueError("apptainer_start_max_concurrent must be positive")
    loop = asyncio.get_running_loop()
    by_limit = _START_SEMAPHORES.setdefault(loop, {})
    semaphore = by_limit.get(limit)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        by_limit[limit] = semaphore
    return semaphore


def _apptainer_host_environment() -> dict[str, str]:
    """Keep short-lived Apptainer control processes from exhausting host tasks.

    Apptainer is a Go binary.  On the H100 nodes it sees the full CPU set and,
    unless constrained, creates a GC worker pool proportional to that set even
    for metadata-only commands such as ``sif info``.  Hundreds of concurrent
    SWE sandboxes can therefore consume the per-user task allowance in a short
    burst before the actual two-CPU verifier command starts.  This environment
    applies only to the host-side Apptainer wrapper; commands inside the
    sandbox retain ``apptainer_cpu`` and their existing BLAS/OpenMP settings.
    """

    environment = os.environ.copy()
    environment["GOMAXPROCS"] = "1"
    return environment


@dataclass
class ApptainerTask:
    """One isolated SWE task backed by an immutable SIF and bound state."""

    image: str
    command_timeout: float
    start_timeout: float
    network: str
    apptainer: str
    work_root: str
    instance_id: str
    fakeroot: bool = True
    nv: bool = False
    cpu: int | None = None
    memory_gb: int | None = None
    normalize_tool_timeout: bool = True
    keep_workspace: bool = False
    start_max_concurrent: int = 8
    broker_start_attempts: int = 40
    persistent_exec: bool = True
    disable_proc: bool = False
    single_thread_squashfuse: bool = False
    state_mode: str = "host_bind"
    overlay_size_mb: int = 4096
    task_dir: Path | None = None
    testbed_dir: Path | None = None
    home_dir: Path | None = None
    tmp_dir: Path | None = None
    overlay_path: Path | None = None
    rootfs_mount_dir: Path | None = None
    _rootfs_mount_process: asyncio.subprocess.Process | None = field(
        default=None, init=False, repr=False
    )
    _marker_token: str | None = None
    _upload_binds: dict[str, Path] = field(default_factory=dict)
    _broker: asyncio.subprocess.Process | None = field(default=None, init=False, repr=False)
    _broker_lock: asyncio.Lock | None = field(default=None, init=False, repr=False)
    _broker_stderr_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _broker_stderr_tail: bytearray = field(default_factory=bytearray, init=False, repr=False)
    metrics: dict[str, object] = field(
        default_factory=lambda: {
            "backend": "apptainer",
            "exec_calls": [],
            "uploads": [],
            "persistent_instance": False,
            "state_mode": "host_bind",
        }
    )

    async def _run_host(self, *args: str, timeout: float) -> tuple[int, str]:
        spawn_deadline = time.monotonic() + min(20.0, max(1.0, timeout))
        retries = 0
        while True:
            try:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=_apptainer_host_environment(),
                )
                break
            except (BlockingIOError, RuntimeError) as error:
                transient = (
                    isinstance(error, BlockingIOError) and error.errno == errno.EAGAIN
                ) or "can't start new thread" in str(error).lower()
                if not transient or time.monotonic() >= spawn_deadline:
                    raise
                retries += 1
                await asyncio.sleep(min(0.05 * retries, 0.5))
        if retries:
            self.metrics["host_spawn_retries"] = int(
                self.metrics.get("host_spawn_retries", 0)
            ) + retries
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except (TimeoutError, asyncio.CancelledError):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            # Apptainer may leave a squashfuse helper outside the killed process
            # group.  Such a helper can retain the stdout pipe even after the
            # actual ``apptainer exec`` process exits.  A second unbounded
            # communicate() would then turn the command timeout into a permanent
            # rollout hang.  Close our pipe transport and only reap the direct
            # child for a bounded interval; the component supervisor owns any
            # escaped helper cleanup.
            stdout = process.stdout
            transport = getattr(stdout, "_transport", None)
            if transport is not None:
                transport.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=10)
            raise
        return process.returncode or 0, output.decode("utf-8", errors="replace")

    async def _drain_broker_stderr(self, stream: asyncio.StreamReader) -> None:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            self._broker_stderr_tail.extend(chunk)
            if len(self._broker_stderr_tail) > 65536:
                del self._broker_stderr_tail[:-65536]

    def _broker_error_tail(self) -> str:
        return bytes(self._broker_stderr_tail).decode("utf-8", errors="replace")

    async def _start_broker(self) -> None:
        if not self.persistent_exec:
            return
        if self.broker_start_attempts < 1:
            raise ValueError("broker_start_attempts must be positive")
        started = time.monotonic()
        errors: list[str] = []
        for attempt in range(1, self.broker_start_attempts + 1):
            self._broker_stderr_tail.clear()
            try:
                process = await asyncio.create_subprocess_exec(
                    *self._runtime_prefix(bind_testbed=True),
                    self._runtime_image(),
                    "python3",
                    "-u",
                    "-c",
                    _COMMAND_BROKER,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                    limit=128 * 1024 * 1024,
                    env=_apptainer_host_environment(),
                )
            except (BlockingIOError, RuntimeError) as error:
                transient = (
                    isinstance(error, BlockingIOError) and error.errno == errno.EAGAIN
                ) or "can't start new thread" in str(error).lower()
                if not transient:
                    raise
                errors.append(f"attempt {attempt}: {error!r}")
                if attempt == self.broker_start_attempts:
                    raise RuntimeError(
                        "Apptainer command broker process could not be spawned after "
                        f"{self.broker_start_attempts} attempts: " + " | ".join(errors)
                    ) from error
                await asyncio.sleep(min(0.25 * attempt, 2.0))
                continue
            self._broker = process
            self._broker_lock = asyncio.Lock()
            assert process.stderr is not None and process.stdout is not None
            self._broker_stderr_task = asyncio.create_task(
                self._drain_broker_stderr(process.stderr)
            )
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(), self.start_timeout
                )
                ready = json.loads(line) if line else None
                if ready != {"ready": True}:
                    raise RuntimeError(
                        f"invalid command-broker greeting: {ready!r}"
                    )
            except asyncio.CancelledError:
                await self._stop_broker(force=True)
                raise
            except Exception as error:
                errors.append(
                    f"attempt {attempt}: {error!r}; "
                    f"stderr={self._broker_error_tail()!r}"
                )
                await self._stop_broker(force=True)
                if attempt == self.broker_start_attempts:
                    raise RuntimeError(
                        "Apptainer command broker failed to start after "
                        f"{self.broker_start_attempts} attempts: "
                        + " | ".join(errors)
                    ) from error
                # At high verifier concurrency, process or mount-namespace
                # creation can fail transiently. A bounded backoff avoids
                # turning that host-side burst into a failed SWE episode.
                await asyncio.sleep(min(0.25 * attempt, 2.0))
            else:
                self.metrics["command_broker_start_attempts"] = attempt
                self.metrics["command_broker_start_retries"] = attempt - 1
                self.metrics["command_broker_start_seconds"] = (
                    time.monotonic() - started
                )
                return

    async def _stop_broker(self, *, force: bool = False) -> None:
        process, self._broker = self._broker, None
        stderr_task, self._broker_stderr_task = self._broker_stderr_task, None
        self._broker_lock = None
        if process is None:
            if stderr_task is not None:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
            return
        if process.returncode is None and not force and process.stdin is not None:
            process.stdin.close()
            with contextlib.suppress(Exception):
                await process.stdin.wait_closed()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), 10)
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(process.wait()), 10)
        # Apptainer can re-parent a squashfuse helper outside the broker's
        # process group.  The helper inherits the broker's stdout/stderr and
        # can therefore keep StreamReader EOF pending after the direct child
        # has exited.  Closing our pipe transports and cancelling the drain is
        # safe once the broker is no longer usable, and keeps a command timeout
        # from becoming an unbounded episode hang.
        for stream in (process.stdout, process.stderr):
            transport = getattr(stream, "_transport", None)
            if transport is not None:
                transport.close()
        if stderr_task is not None:
            if not stderr_task.done():
                stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(stderr_task, 5)

    async def _execute_broker(
        self,
        *,
        command: str,
        seconds: int,
        phase: str,
        environment: dict[str, str],
        interpreter: tuple[str, ...],
    ) -> tuple[int, str, float]:
        process = self._broker
        lock = self._broker_lock
        if process is None or lock is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Apptainer command broker is unavailable")
        timeout_exit_code = 124 if self.normalize_tool_timeout and phase == "agent_tool" else 137
        request = json.dumps(
            {
                "argv": [*interpreter, command],
                "environment": environment,
                "timeout": seconds,
                "timeout_exit_code": timeout_exit_code,
            },
            separators=(",", ":"),
        ).encode() + b"\n"
        async with lock:
            try:
                process.stdin.write(request)
                await process.stdin.drain()
                line = await asyncio.wait_for(process.stdout.readline(), seconds + 30)
            except TimeoutError:
                # The broker normally turns its child deadline into 124/137.
                # Keep the outer deadline as a second, bounded line of defence:
                # a wedged broker must still look like the command timeout it
                # was enforcing, rather than failing the entire SWE episode as
                # an infrastructure error.
                await self._stop_broker(force=True)
                return (
                    timeout_exit_code,
                    "[apptainer] command broker exceeded its timeout grace period",
                    float(seconds + 30),
                )
            except (asyncio.CancelledError, BrokenPipeError):
                await self._stop_broker(force=True)
                raise
        if not line:
            returncode = process.returncode
            raise RuntimeError(
                "Apptainer command broker exited unexpectedly "
                f"(code={returncode}, stderr={self._broker_error_tail()!r})"
            )
        response = json.loads(line)
        if "broker_error" in response:
            raise RuntimeError(
                f"Apptainer command broker failed: {response['broker_error']}\n"
                f"{response.get('traceback', '')}"
            )
        output = base64.b64decode(response["output_b64"]).decode(
            "utf-8", errors="replace"
        )
        return int(response["returncode"]), output, float(response["child_seconds"])

    def _require_started(self) -> tuple[Path, Path, Path, Path]:
        if not all((self.task_dir, self.testbed_dir, self.home_dir, self.tmp_dir)):
            raise RuntimeError("Apptainer SWE-bench sandbox is not running")
        return self.task_dir, self.testbed_dir, self.home_dir, self.tmp_dir

    def _runtime_image(self) -> str:
        return str(self.rootfs_mount_dir or self.image)

    async def _start_single_thread_rootfs(self, image: Path) -> None:
        """Mount a SIF rootfs with one FUSE thread instead of the default pool."""

        if not self.single_thread_squashfuse:
            return
        if not self.disable_proc:
            raise ValueError(
                "apptainer_single_thread_squashfuse requires "
                "apptainer_disable_proc=true on this rootless site"
            )
        if self.state_mode != "host_bind":
            raise ValueError(
                "apptainer_single_thread_squashfuse currently supports only "
                "apptainer_state_mode=host_bind"
            )
        assert self.task_dir is not None
        code, info = await self._run_host(
            self.apptainer,
            "sif",
            "info",
            "1",
            str(image),
            timeout=min(self.start_timeout, 60),
        )
        if code:
            raise RuntimeError(f"failed to inspect SIF rootfs offset: {info.strip()}")
        offset = None
        for line in info.splitlines():
            if line.strip().startswith("Offset:"):
                offset = int(line.split(":", 1)[1].strip())
                break
        if offset is None or "Filesystem Type:  Squashfs" not in info:
            raise RuntimeError(f"SIF object 1 is not a Squashfs rootfs: {image}")

        apptainer_path = Path(self.apptainer).resolve()
        install_root = apptainer_path.parent.parent
        squashfuse_candidates = (
            install_root / "libexec" / "apptainer" / "bin" / "squashfuse_ll",
            install_root
            / "x86_64"
            / "libexec"
            / "apptainer"
            / "bin"
            / "squashfuse_ll",
        )
        squashfuse = next(
            (candidate for candidate in squashfuse_candidates if candidate.is_file()),
            None,
        )
        if squashfuse is None:
            raise RuntimeError(
                "single-thread squashfuse helper is unavailable; checked: "
                + ", ".join(str(candidate) for candidate in squashfuse_candidates)
            )
        mount_dir = self.task_dir / "rootfs"
        mount_dir.mkdir(mode=0o700)
        process = await asyncio.create_subprocess_exec(
            str(squashfuse),
            "-s",
            "-f",
            "-o",
            f"uid={os.getuid()},gid={os.getgid()},offset={offset}",
            str(image),
            str(mount_dir),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.start_timeout
        while not (mount_dir / "bin" / "sh").exists():
            if process.returncode is not None:
                raise RuntimeError(
                    f"single-thread squashfuse exited before mount was ready: "
                    f"code={process.returncode}, image={image}"
                )
            if time.monotonic() >= deadline:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
                raise TimeoutError(f"timed out mounting SIF rootfs: {image}")
            await asyncio.sleep(0.05)
        self.rootfs_mount_dir = mount_dir
        self._rootfs_mount_process = process
        self.metrics["rootfs_transport"] = "single_thread_squashfuse"

    async def _stop_single_thread_rootfs(self) -> None:
        mount_dir, self.rootfs_mount_dir = self.rootfs_mount_dir, None
        process, self._rootfs_mount_process = self._rootfs_mount_process, None
        if mount_dir is not None:
            fusermount = shutil.which("fusermount3") or shutil.which("fusermount")
            if fusermount is not None:
                with contextlib.suppress(Exception):
                    await self._run_host(fusermount, "-u", str(mount_dir), timeout=30)
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), 10)
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), 5)

    def _runtime_prefix(self, *, bind_testbed: bool) -> list[str]:
        task_dir, testbed_dir, home_dir, tmp_dir = self._require_started()
        container_home = "/root" if self.fakeroot else str(Path.home())
        command = [
            self.apptainer,
            "--silent",
            "exec",
            "--cleanenv",
            "--containall",
            "--no-eval",
            # Limit the persistent Go appinit wrapper itself.  execute() adds
            # the task's configured value again for one-shot children, while
            # the command broker explicitly overrides it per child request.
            "--env",
            "GOMAXPROCS=1",
        ]
        if self.fakeroot:
            command.append("--fakeroot")
        if self.state_mode == "overlay":
            if self.overlay_path is None:
                raise RuntimeError("Apptainer overlay has not been created")
            command.extend(("--overlay", str(self.overlay_path)))
        if self.nv:
            command.append("--nv")
        if self.disable_proc:
            command.extend(("--no-mount", "proc"))
        if self.network == "none":
            command.extend(("--net", "--network", "none"))
        elif self.network not in {"", "host"}:
            raise ValueError(
                "Apptainer backend currently supports only container_network "
                f"'none' or 'host', got {self.network!r}"
            )
        home_bind = _bind_value(home_dir, container_home).rsplit(":", 1)[0]
        command.extend(("--home", home_bind))
        command.extend(("--bind", _bind_value(tmp_dir, "/tmp")))
        spawn_lock = Path(self.work_root).expanduser() / ".spawn.lock"
        spawn_lock.touch(exist_ok=True)
        command.extend(("--bind", _bind_value(spawn_lock, "/pd-swe-spawn.lock")))
        if bind_testbed and self.state_mode == "host_bind":
            command.extend(("--bind", _bind_value(testbed_dir, "/testbed")))
        if bind_testbed:
            for destination, source in sorted(self._upload_binds.items()):
                command.extend(("--bind", _bind_value(source, destination, "ro")))
            command.extend(("--pwd", "/testbed"))
        elif self.state_mode == "host_bind":
            command.extend(("--bind", _bind_value(task_dir, "/pd-swe-state")))
            command.extend(("--pwd", "/"))
        else:
            command.extend(("--pwd", "/"))
        return command

    async def start(self) -> None:
        apptainer = Path(self.apptainer)
        image = Path(self.image)
        if not apptainer.is_file() or not os.access(apptainer, os.X_OK):
            raise RuntimeError(f"Apptainer executable is unavailable: {apptainer}")
        if not image.is_file():
            raise RuntimeError(f"SWE-bench SIF is unavailable: {image}")
        if self.cpu is not None and self.cpu <= 0:
            raise ValueError("apptainer_cpu must be positive")
        if self.memory_gb is not None and self.memory_gb <= 0:
            raise ValueError("apptainer_memory_gb must be positive")
        if self.state_mode not in {"host_bind", "overlay"}:
            raise ValueError("apptainer_state_mode must be 'host_bind' or 'overlay'")
        if self.overlay_size_mb <= 0:
            raise ValueError("apptainer_overlay_size_mb must be positive")
        if self.state_mode == "overlay" and not self.fakeroot:
            raise ValueError("apptainer_state_mode=overlay requires apptainer_fakeroot=true")

        root = Path(self.work_root).expanduser()
        if not root.is_absolute():
            raise ValueError("apptainer_work_root must be an absolute path")
        root.mkdir(parents=True, exist_ok=True)
        self.task_dir = Path(tempfile.mkdtemp(prefix="pd-swe-", dir=root))
        self.testbed_dir = self.task_dir / "testbed"
        self.home_dir = self.task_dir / "home"
        self.tmp_dir = self.task_dir / "tmp"
        self.overlay_path = (
            self.task_dir / "task-overlay.img"
            if self.state_mode == "overlay"
            else None
        )
        for directory in (self.testbed_dir, self.home_dir, self.tmp_dir):
            directory.mkdir(mode=0o700)
        self._marker_token = uuid.uuid4().hex
        (self.task_dir / _TASK_MARKER).write_text(self._marker_token + "\n")
        await self._start_single_thread_rootfs(image)

        self.metrics.update(
            image=str(image),
            workspace=str(self.task_dir),
            network=self.network,
            fakeroot=self.fakeroot,
            nv=self.nv,
            cpu_threads=self.cpu,
            # Rootless Apptainer cannot enforce cgroups on this site.  Keep the
            # requested value visible instead of pretending it is enforced.
            memory_limit_gb=self.memory_gb,
            memory_limit_enforced=False,
            persistent_exec=self.persistent_exec,
            proc_mounted=not self.disable_proc,
            single_thread_squashfuse=self.single_thread_squashfuse,
            state_mode=self.state_mode,
            overlay_size_mb=self.overlay_size_mb if self.state_mode == "overlay" else None,
        )
        queued = time.monotonic()
        async with _start_semaphore(self.start_max_concurrent):
            started = time.monotonic()
            self.metrics["start_queue_seconds"] = started - queued
            start_attempts = 12
            for attempt in range(1, start_attempts + 1):
                if self.state_mode == "overlay":
                    assert self.overlay_path is not None
                    returncode, output = await self._run_host(
                        self.apptainer,
                        "overlay",
                        "create",
                        "--fakeroot",
                        "--size",
                        str(self.overlay_size_mb),
                        "--sparse",
                        str(self.overlay_path),
                        timeout=self.start_timeout,
                    )
                    if not returncode:
                        returncode, output = await self._run_host(
                            *self._runtime_prefix(bind_testbed=True),
                            self._runtime_image(),
                            "bash",
                            "-lc",
                            "test -d /testbed; test -e /testbed/.git; test -w /testbed",
                            timeout=self.start_timeout,
                        )
                else:
                    returncode, output = await self._run_host(
                        *self._runtime_prefix(bind_testbed=False),
                        self._runtime_image(),
                        "bash",
                        "-lc",
                        _COPY_TESTBED,
                        timeout=self.start_timeout,
                    )
                if not returncode:
                    break
                if attempt == start_attempts or not _transient_spawn_failure(output):
                    break
                if self.state_mode == "host_bind":
                    assert self.testbed_dir is not None
                    shutil.rmtree(self.testbed_dir)
                    self.testbed_dir.mkdir(mode=0o700)
                elif self.overlay_path is not None:
                    self.overlay_path.unlink(missing_ok=True)
                await asyncio.sleep(min(0.25 * attempt, 2.0))
            self.metrics["container_start_attempts"] = attempt
            self.metrics["container_start_retries"] = attempt - 1
            self.metrics["container_start_seconds"] = time.monotonic() - started
            if returncode:
                action = "create writable overlay" if self.state_mode == "overlay" else "materialize /testbed"
                raise RuntimeError(f"failed to {action} from {self.image}: {output.strip()}")
            if self.state_mode == "host_bind" and not (self.testbed_dir / ".git").exists():
                raise RuntimeError(f"SIF has no usable /testbed Git repository: {self.image}")
            await self._start_broker()
            setup_code, setup_output = await self.execute(
                "git config --global --add safe.directory /testbed && "
                "git config --global core.preloadIndex false",
                timeout=30,
                phase="sandbox_setup",
            )
            if setup_code:
                raise RuntimeError(f"failed to configure isolated Git home: {setup_output}")

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        phase: str = "agent_tool",
        environment: dict[str, str] | None = None,
        interpreter: tuple[str, ...] = ("bash", "-lc"),
    ) -> tuple[int, str]:
        self._require_started()
        seconds = max(1, int(timeout or self.command_timeout))
        runtime = self._runtime_prefix(bind_testbed=True)
        thread_count = str(self.cpu) if self.cpu is not None else None
        runtime_environment = dict(environment or {})
        if thread_count is not None:
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "GOMAXPROCS",
            ):
                runtime_environment.setdefault(key, thread_count)
        runtime_environment.setdefault("PYTHONUNBUFFERED", "1")
        for key, value in sorted(runtime_environment.items()):
            if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
                raise ValueError(f"invalid environment variable name: {key!r}")
        started = time.monotonic()
        broker_child_seconds = None
        if self._broker is not None and not self._upload_binds:
            returncode, output, broker_child_seconds = await self._execute_broker(
                command=command,
                seconds=seconds,
                phase=phase,
                environment=runtime_environment,
                interpreter=interpreter,
            )
            transport = "persistent_exec"
        else:
            for key, value in sorted(runtime_environment.items()):
                runtime.extend(("--env", f"{key}={value}"))
            if self.normalize_tool_timeout and phase == "agent_tool":
                runner = ("python3", "-c", _TIMEOUT_RUNNER, str(seconds))
            else:
                runner = ("timeout", "-s", "KILL", f"{seconds}s")
            returncode, output = await self._run_host(
                *runtime,
                self._runtime_image(),
                *runner,
                *interpreter,
                command,
                timeout=seconds + 30,
            )
            transport = "one_shot"
        exec_calls = self.metrics["exec_calls"]
        assert isinstance(exec_calls, list)
        exec_calls.append(
            {
                "phase": phase,
                "duration_seconds": time.monotonic() - started,
                "timeout_seconds": seconds,
                "exit_code": returncode,
                "command_chars": len(command),
                "output_chars": len(output),
                "transport": transport,
                "child_seconds": broker_child_seconds,
            }
        )
        return returncode, output

    async def upload_bytes(self, contents: bytes, path: str) -> None:
        task_dir, testbed_dir, _home_dir, tmp_dir = self._require_started()
        destination = _safe_container_path(path)
        started = time.monotonic()
        parts = destination.parts
        if self.state_mode == "overlay" and not (
            len(parts) >= 2 and parts[1] == "tmp"
        ):
            encoded = base64.b64encode(contents).decode("ascii")
            destination_json = json.dumps(str(destination))
            code, output = await self.execute(
                "import base64,pathlib; "
                f"p=pathlib.Path({destination_json}); p.parent.mkdir(parents=True, exist_ok=True); "
                f"p.write_bytes(base64.b64decode({encoded!r}))",
                timeout=self.start_timeout,
                phase="sandbox_upload",
                interpreter=("python3", "-c"),
            )
            if code:
                raise RuntimeError(f"failed to upload {destination} into overlay: {output}")
            host_path = None
        elif len(parts) >= 2 and parts[1] == "testbed":
            host_path = testbed_dir.joinpath(*parts[2:])
        elif len(parts) >= 2 and parts[1] == "tmp":
            host_path = tmp_dir.joinpath(*parts[2:])
        else:
            upload_dir = task_dir / "uploads"
            upload_dir.mkdir(mode=0o700, exist_ok=True)
            host_path = upload_dir / uuid.uuid4().hex
            self._upload_binds[str(destination)] = host_path
        if host_path is not None:
            host_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = host_path.with_name(host_path.name + ".tmp-" + uuid.uuid4().hex)
            try:
                temporary.write_bytes(contents)
                temporary.chmod(0o644)
                os.replace(temporary, host_path)
            finally:
                temporary.unlink(missing_ok=True)
        uploads = self.metrics["uploads"]
        assert isinstance(uploads, list)
        uploads.append(
            {
                "path": str(destination),
                "bytes": len(contents),
                "duration_seconds": time.monotonic() - started,
            }
        )

    async def close(self) -> None:
        await asyncio.shield(self._stop_broker())
        await asyncio.shield(self._stop_single_thread_rootfs())
        task_dir, self.task_dir = self.task_dir, None
        self.testbed_dir = self.home_dir = self.tmp_dir = None
        self.overlay_path = None
        self._upload_binds.clear()
        if task_dir is None:
            return
        started = time.monotonic()
        if self.keep_workspace:
            self.metrics["workspace_retained"] = str(task_dir)
            return
        root = Path(self.work_root).expanduser().resolve()
        resolved = task_dir.resolve()
        marker = resolved / _TASK_MARKER
        expected = self._marker_token
        self._marker_token = None
        try:
            is_owned_child = resolved.parent == root and resolved.name.startswith("pd-swe-")
            marker_matches = expected is not None and marker.read_text().strip() == expected
        except (OSError, ValueError):
            is_owned_child = marker_matches = False
        if not (is_owned_child and marker_matches):
            raise RuntimeError(f"refusing to remove unowned Apptainer workspace: {resolved}")
        shutil.rmtree(resolved)
        self.metrics["container_close_seconds"] = time.monotonic() - started
