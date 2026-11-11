"""BrowseComp dataset adapter and search-agent harness."""

from dualpd.runtime.workloads.api import HarnessSpec

from .harness import generate
from .loader import load_samples


HARNESS = HarnessSpec(
    name="browsecomp",
    load_samples=load_samples,
    generate=generate,
    default_max_response_tokens=36864,
    tools=("search", "open_page", "finish"),
    required_services=("browsecomp_search",),
)

__all__ = ["HARNESS"]
