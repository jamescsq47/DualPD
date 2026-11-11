"""SWE-bench adapter using the OpenEnv-style episode contract from Miles PR #51."""

from dualpd.runtime.workloads.api import HarnessSpec
from dualpd.data.swe_bench_verified.loader import load_samples

from .harness import generate


HARNESS = HarnessSpec(
    name="swe_bench_verified",
    load_samples=load_samples,
    generate=generate,
    default_max_response_tokens=81920,
    tools=("shell",),
    metadata={
        "agent": "Miles PR #51 OpenEnv-style SWE-bench shell loop",
        "environment": "per-trajectory Docker or Apptainer sandbox",
        "verifier": "official SWE-bench verifier injected after agent termination",
    },
)

__all__ = ["HARNESS"]
