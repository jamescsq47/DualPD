"""Multi-turn SWE-bench inference harness backed by isolated Docker tasks."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .apptainer_backend import ApptainerTask



# Run inside the task container. Unlike `timeout --signal=KILL`, the supervisor
# survives killing the command group and returns the conventional timeout124.
# A command killed independently still returns137, not a false timeout.
_DOCKER_TIMEOUT_RUNNER = """
import os, signal, subprocess, sys, tempfile
# Descendants must not inherit the docker exec attachment pipe. A background
# process may outlive its shell, but cannot delay the result by holding EOF open.
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

# Only for this adapter's single-task Docker PID namespace, never the host.
# Quiescence, not merely signal delivery, is required before patch capture.
_DOCKER_TIMEOUT_QUIESCE = """
import os, signal, sys, time
deadline = time.monotonic() + 5
while True:
    active = []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) in (1, os.getpid()):
            continue
        try:
            with open('/proc/' + name + '/stat') as f:
                state = f.read().rsplit(')', 1)[1].split()[0]
            if state not in ('Z', 'X'):
                active.append(int(name))
        except FileNotFoundError:
            pass
    if not active:
        sys.exit(0)
    if time.monotonic() >= deadline:
        sys.exit(1)
    for pid in active:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(.02)
"""


def _truncate_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    keep = max(1, (limit - 96) // 2)
    removed = len(value) - 2 * keep
    return f"{value[:keep]}\n...[truncated {removed} characters]...\n{value[-keep:]}"


def _image_name(metadata: dict[str, Any], options: dict[str, Any]) -> str:
    if metadata.get("image_name"):
        return str(metadata["image_name"])
    instance = str(metadata["instance_id"]).lower().replace("__", "_1776_")
    template = str(
        options.get(
            "image_template",
            "docker.io/swebench/sweb.eval.x86_64.{instance}:latest",
        )
    )
    return template.format(instance=instance, instance_id=metadata["instance_id"])


@dataclass
class DockerTask:
    image: str
    command_timeout: float
    start_timeout: float
    network: str
    container_id: str | None = None
    metrics: dict[str, Any] = field(
        default_factory=lambda: {
            "backend": "docker",
            "exec_calls": [],
            "uploads": [],
        }
    )
    # Opt-in transport parity with Daytona. Preserve legacy image defaults
    # for other datasets/launchers that share this Docker adapter.
    user: str | None = None
    cpu: int | None = None
    memory_gb: int | None = None
    normalize_tool_timeout: bool = False

    async def _run_host(self, *args: str, timeout: float) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise
        return process.returncode or 0, output.decode("utf-8", errors="replace")

    async def start(self) -> None:
        inspect_started = time.monotonic()
        inspect_code, _ = await self._run_host(
            "docker", "image", "inspect", self.image, timeout=60
        )
        self.metrics["image_inspect_seconds"] = time.monotonic() - inspect_started
        self.metrics["image_present_before_start"] = inspect_code == 0
        name = f"pd-swe-{uuid.uuid4().hex[:12]}"
        command = [
            "docker", "run", "-d", "--rm", "--name", name,
        ]
        run_id = os.environ.get("PD_SWE_RUN_ID", "").strip()
        if run_id:
            command.extend(["--label", f"pd.swe.run_id={run_id}"])
        if self.user is not None:
            command.extend(["--user", self.user])
        if self.cpu is not None:
            if self.cpu <= 0:
                raise ValueError("docker_cpu must be positive")
            command.extend(["--cpus", str(self.cpu)])
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                command.extend(["-e", f"{key}={self.cpu}"])
            command.extend(["-e", "PYTHONUNBUFFERED=1"])
        if self.memory_gb is not None:
            if self.memory_gb <= 0:
                raise ValueError("docker_memory_gb must be positive")
            command.extend(["--memory", f"{self.memory_gb}g"])
        self.metrics.update(container_user=self.user, cpu_quota=self.cpu,
                            memory_limit_gb=self.memory_gb)
        command.extend([
            "--entrypoint", "sleep", "--network", self.network,
            "-w", "/testbed", self.image, "86400",
        ])
        start_started = time.monotonic()
        returncode, output = await self._run_host(*command, timeout=self.start_timeout)
        self.metrics["container_start_seconds"] = time.monotonic() - start_started
        if returncode:
            raise RuntimeError(f"docker run failed for {self.image}: {output.strip()}")
        self.container_id = output.strip()
        if not self.container_id:
            raise RuntimeError(f"docker run returned no container id for {self.image}")
        # Official images can use a non-root repository owner. Mark only the
        # isolated /testbed checkout safe, without modifying the host config.
        await self.execute(
            "git config --global --add safe.directory /testbed",
            timeout=30,
            phase="sandbox_setup",
        )

    async def execute(
        self,
        command: str,
        *,
        timeout: float | None = None,
        phase: str = "agent_tool",
        environment: dict[str, str] | None = None,
        interpreter: tuple[str, ...] = ("bash", "-lc"),
    ) -> tuple[int, str]:
        if not self.container_id:
            raise RuntimeError("SWE-bench container is not running")
        seconds = max(1, int(timeout or self.command_timeout))
        started = time.monotonic()
        docker_environment: list[str] = []
        for key, value in (environment or {}).items():
            docker_environment.extend(("-e", f"{key}={value}"))
        if self.normalize_tool_timeout and phase == "agent_tool":
            runner = ["python3", "-c", _DOCKER_TIMEOUT_RUNNER, str(seconds)]
        else:
            runner = ["timeout", "--signal=KILL", f"{seconds}s"]
        returncode, output = await self._run_host(
            "docker",
            "exec",
            "-w",
            "/testbed",
            *docker_environment,
            self.container_id,
            *runner,
            *interpreter,
            command,
            timeout=seconds + 10,
        )
        timeout_cleanup_seconds = None
        if self.normalize_tool_timeout and phase == "agent_tool" and returncode == 124:
            cleanup_started = time.monotonic()
            cleanup_code, cleanup_output = await self._run_host(
                "docker", "exec", "--user", "root", self.container_id,
                "python3", "-c", _DOCKER_TIMEOUT_QUIESCE, timeout=10,
            )
            if cleanup_code:
                raise RuntimeError(f"tool timeout cleanup did not quiesce: {cleanup_output}")
            timeout_cleanup_seconds = time.monotonic() - cleanup_started
        self.metrics["exec_calls"].append(
            {
                "phase": phase,
                "duration_seconds": time.monotonic() - started,
                "timeout_seconds": seconds,
                "exit_code": returncode,
                "command_chars": len(command),
                "output_chars": len(output),
                **({"timeout_cleanup_seconds": timeout_cleanup_seconds}
                   if timeout_cleanup_seconds is not None else {}),
            }
        )
        return returncode, output

    async def upload_bytes(self, contents: bytes, path: str) -> None:
        if not self.container_id:
            raise RuntimeError("SWE-bench container is not running")
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(prefix="pd-swe-upload-", delete=False) as handle:
                handle.write(contents)
                temporary_path = handle.name
            # ``docker cp`` preserves the source mode while creating the file
            # as root inside the container.  Current OpenHands source-minimal
            # images run as the unprivileged ``openhands`` user, so the
            # NamedTemporaryFile default (0600) would make verifier scripts and
            # patches unreadable after the copy.
            os.chmod(temporary_path, 0o644)
            started = time.monotonic()
            returncode, output = await self._run_host(
                "docker",
                "cp",
                temporary_path,
                f"{self.container_id}:{path}",
                timeout=self.start_timeout,
            )
            if returncode:
                raise RuntimeError(
                    f"docker cp failed for {self.container_id}:{path}: {output.strip()}"
                )
            self.metrics["uploads"].append(
                {
                    "path": path,
                    "bytes": len(contents),
                    "duration_seconds": time.monotonic() - started,
                }
            )
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

    async def close(self) -> None:
        container_id, self.container_id = self.container_id, None
        if not container_id:
            return
        started = time.monotonic()
        with contextlib.suppress(Exception):
            await asyncio.shield(
                self._run_host("docker", "rm", "-f", container_id, timeout=60)
            )
        self.metrics["container_close_seconds"] = time.monotonic() - started




def _create_task(
    metadata: dict[str, Any], options: dict[str, Any]
) -> DockerTask | ApptainerTask:
    backend = str(options.get("sandbox_backend", "docker")).strip().lower()
    common = {
        "image": _image_name(metadata, options),
        "command_timeout": float(options.get("command_timeout_seconds", 300)),
        "start_timeout": float(options.get("container_start_timeout_seconds", 600)),
    }
    if backend == "docker":
        return DockerTask(
            **common,
            network=str(options.get("container_network", "none")),
            user=str(options["docker_user"]) if options.get("docker_user") is not None else None,
            cpu=int(options["docker_cpu"]) if options.get("docker_cpu") is not None else None,
            memory_gb=int(options["docker_memory_gb"]) if options.get("docker_memory_gb") is not None else None,
            normalize_tool_timeout=bool(options.get("docker_normalize_tool_timeout", False)),
        )
    if backend == "apptainer":
        instance = str(metadata["instance_id"]).lower().replace("__", "_1776_")
        image_template = str(options.get("apptainer_image_template", "")).strip()
        image_dir = str(options.get("apptainer_image_dir", "")).strip()
        if image_template:
            image = os.path.expanduser(os.path.expandvars(image_template)).format(
                instance=instance,
                instance_id=str(metadata["instance_id"]),
            )
        elif image_dir:
            image = str(
                Path(os.path.expanduser(os.path.expandvars(image_dir)))
                / f"{instance}.sif"
            )
        else:
            raise ValueError(
                "sandbox_backend=apptainer requires apptainer_image_dir or "
                "apptainer_image_template"
            )
        work_root = str(options.get("apptainer_work_root", "")).strip()
        if not work_root:
            work_root = os.environ.get("PD_SWE_APPTAINER_WORK_ROOT", "").strip()
        if not work_root:
            raise ValueError(
                "sandbox_backend=apptainer requires apptainer_work_root or "
                "PD_SWE_APPTAINER_WORK_ROOT"
            )
        work_root = os.path.expanduser(os.path.expandvars(work_root))
        return ApptainerTask(
            command_timeout=common["command_timeout"],
            start_timeout=common["start_timeout"],
            image=image,
            network=str(options.get("container_network", "none")),
            apptainer=str(
                options.get(
                    "apptainer_binary",
                    "apptainer",
                )
            ),
            work_root=work_root,
            instance_id=str(metadata["instance_id"]),
            fakeroot=bool(options.get("apptainer_fakeroot", True)),
            nv=bool(options.get("apptainer_nv", False)),
            cpu=int(options["apptainer_cpu"])
            if options.get("apptainer_cpu") is not None
            else None,
            memory_gb=int(options["apptainer_memory_gb"])
            if options.get("apptainer_memory_gb") is not None
            else None,
            normalize_tool_timeout=bool(
                options.get("apptainer_normalize_tool_timeout", True)
            ),
            keep_workspace=bool(options.get("apptainer_keep_workspace", False)),
            start_max_concurrent=int(
                options.get("apptainer_start_max_concurrent", 8)
            ),
            persistent_exec=bool(options.get("apptainer_persistent_exec", True)),
            disable_proc=bool(options.get("apptainer_disable_proc", False)),
            single_thread_squashfuse=bool(
                options.get("apptainer_single_thread_squashfuse", False)
            ),
            state_mode=str(options.get("apptainer_state_mode", "host_bind")),
            overlay_size_mb=int(options.get("apptainer_overlay_size_mb", 4096)),
        )
    raise ValueError(f"unsupported SWE-bench sandbox_backend {backend!r}")




def _verifier_semaphore(args: Any, limit: int) -> asyncio.Semaphore:
    if limit < 1:
        raise ValueError("verifier_max_concurrent must be positive")
    semaphores = getattr(args, "_swe_bench_verifier_semaphores", None)
    if semaphores is None:
        semaphores = {}
        setattr(args, "_swe_bench_verifier_semaphores", semaphores)
    semaphore = semaphores.get(limit)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        semaphores[limit] = semaphore
    return semaphore
