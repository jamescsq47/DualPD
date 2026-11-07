"""Reproducible CPU extraction regression gate (does not start GPU services).

The full historical test inventory also contains optional third-party probes,
machine-specific baseline AST checks and GPU imports. These are not silently
treated as CPU passes. See docs/EXTRACTION.md.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parent / "sglang_pd"
FRONTEND = [
    "tests/test_standalone.py",
    "tests/test_agentic_kv_request.py",
    "tests/test_late_binding_router.py",
    "tests/test_workload_plugins.py",
    "tests/test_swe_bench_openenv_harness.py",
    "tests/test_swe_bench_harness.py",
    "tests/test_model_http_transport.py",
    "scripts/cluster/test_multinode.py",
    "scripts/cluster/test_global_pd_router.py",
    "scripts/cluster/test_process_supervisor.py",
    "scripts/cluster/test_qwen35_slurm.py",
    "scripts/cluster/test_qwen35_native_pd_slurm.py",
    "scripts/cluster/test_qwen35_colocated_slurm.py",
]
ENGINE_TESTS = [
    "test_agentic_group_protocol.py", "test_agentic_group_transfer.py",
    "test_agentic_multinode.py", "test_agentic_kv_lifecycle.py",
    "test_agentic_transfer_queues.py", "test_agentic_host_transfer.py",
    "test_agentic_multinode_runtime.py", "test_agentic_remote_host.py",
    "test_agentic_memory_authority.py",
]


def main():
    output = ROOT / "tests/reports"
    output.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(map(str, [
        ENGINE / "python", ROOT, ROOT / "scripts/cluster",
    ]))
    groups = [
        ("frontend", ROOT, FRONTEND),
        ("engine", ENGINE, ["python/sglang/srt/disaggregation/dualpd/" + p for p in ENGINE_TESTS]),
    ]
    rows = []
    for name, directory, tests in groups:
        command = [sys.executable, "-m", "pytest", "-q", "--tb=short",
                   f"--junitxml={output / (name + '.xml')}", *tests]
        start = time.time()
        result = subprocess.run(command, cwd=directory, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / (name + ".log")).write_text(result.stdout)
        print(result.stdout, end="")
        rows.append(dict(group=name, command=command, cwd=str(directory),
                         returncode=result.returncode, elapsed_seconds=time.time()-start))
    (output / "cpu-results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if any(row["returncode"] for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
