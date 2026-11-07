"""GAIA text-only web-and-code agent for serving experiments."""

from dualpd.data.api import HarnessSpec

from .loader import load_samples


async def generate(args, sample, sampling_params):
    from .harness import generate as implementation

    return await implementation(args, sample, sampling_params)


HARNESS = HarnessSpec(
    name="gaia_text",
    load_samples=load_samples,
    generate=generate,
    default_max_response_tokens=32768,
    tools=("search_engine_query", "fetch_url_content", "code_interpreter"),
    metadata={
        "serving_modes": ["colocated", "pd"],
        "benchmark": "OpenResearcher/web-bench:gaia_text",
    },
)
