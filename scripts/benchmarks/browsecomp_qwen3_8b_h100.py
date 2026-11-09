"""Frozen Qwen3-8B TP1 4P4D/c256 BrowseComp regression recipe.

This maintained entry does not import historical matrix/autopilot controllers.
It preserves the accepted deployment inputs, including the analysis Python and
all recorded environment keys; it does not configure other historical cases.
"""
import os
from pathlib import Path

ANALYSIS_ENV = Path('/orcd/compute/songhan/001/home/shangy/miniconda3/envs/pd')
DATA = Path('/orcd/scratch/orcd/003/shangy/dualpd') / 'datasets'
MODEL = Path('/orcd/scratch/orcd/003/shangy/dualpd/models/Qwen3-8B')


def clean_environment() -> dict[str, str]:
    keep = {
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "HF_TOKEN",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.setdefault("HOME", "/home/shangy")
    env.setdefault("LANG", "C.UTF-8")
    env["HF_HOME"] = str(DATA / "huggingface-cache")
    return env


def environment(root: Path, engine: Path, python: Path, directory: Path):
    env = clean_environment()
    env.update(
        MODEL_PATH=str(MODEL),
        PD_DATA_ROOT=str(DATA),
        MATH_DATA=str(DATA / 'dapo-math-17k/dapo-math-17k.jsonl'),
        QA_DATA=str(DATA / 'browsecomp/bc_train.jsonl'),
        WORKLOAD_CONFIG='',
        RUN_DIR=str(directory),
        TEMPERATURE='0',
        TOP_P='1',
        TOP_K='-1',
        REQUESTS='680',
        MAX_INFLIGHT=str(256),
        MATH_RATIO='0',
        SCHEDULE_FILE=str(root / 'configs/workloads/fixed_browsecomp_source_order_n680.json'),
        PRESERVE_SOURCE_ORDER='true',
        DISPATCH_POLICY='fixed',
        SEED='2026',
        WARMUP_SECONDS='300',
        MAX_WARMUP_SECONDS='420',
        MEASURE_SECONDS='1200',
        CLOSED_LOOP='1',
        WARMUP_REQUESTS='0',
        MODEL_CONTEXT_LENGTH='40960',
        MODEL_MAX_RESPONSE_LENGTH='36864',
        SEARCH_GPU='7',
        SEARCH_START_AFTER_MODELS='true',
        SEARCH_SERVER_EMBEDDING_CACHE=os.environ.get(
            'SEARCH_SERVER_EMBEDDING_CACHE',
            str(next(
                (DATA / 'huggingface-cache/hub/datasets--miaolu3--browsecomp-plus/snapshots').glob(
                    '*/corpus_embeddings.pkl'
                ),
                root / 'data/browsecomp/artifacts/search/corpus_embeddings.pkl',
            )),
        ),
        PD_INFERENCE_RETURN_LOGPROB='false',
        MEM_FRACTION_STATIC='0.80',
        MAX_EXISTING_GPU_MEMORY_MB='1024',
        PD_IDLE_GPU_MAX_MIB='1024',
        SLIME_HTTP_READ_TIMEOUT_SECONDS='3600',
        POST_ANALYZER='pd_offload',
        SEARCH_PORT='8750',
        PD_ENV_BIN=str(python.parent),
        PATH=f"{python.parent}:{ANALYSIS_ENV / 'bin'}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        PYTHONPATH=f"{engine / 'python'}:{root}",
        LD_LIBRARY_PATH=(
            f"{ANALYSIS_ENV / 'lib/python3.12/site-packages/nvidia/cuda_runtime/lib'}"
            f":{ANALYSIS_ENV / 'lib'}"
        ),
        CPATH='/usr/include/x86_64-linux-gnu:/usr/include',
        LIBRARY_PATH='/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu',
        PREFILL_GPUS='0 2 4 6',
        DECODE_GPUS='1 3 5 7',
        PREFILL_TP_SIZE='1',
        DECODE_TP_SIZE='1',
        PREFILL_PORTS='27300 27320 27400 27420',
        BOOTSTRAP_PORTS='28300 28320 28400 28420',
        DECODE_PORTS='27301 27321 27401 27421',
        PREFILL_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.80',
        DECODE_MEM_FRACTION_STATICS='0.80 0.80 0.80 0.60',
        ROUTER_PORT='27510',
        ROUTER_PROMETHEUS_PORT='27520',
        AGENTIC_DIRECT_BASE_PORT='61000',
        AGENTIC_DIRECT_PORTS='61000 61001 61002 61003',
        FAST_TOOL_THRESHOLD_SECONDS='1',
        DIRECT_WAIT_SECONDS='1',
        D_TARGET_KV_FRACTION='1.0',
        P2D_SPILL_DELAY_SECONDS='0.5',
        P2D_HOST_STAGING='true',
        D2P_HOST_ARENA_GIB_PER_P='128',
        P2D_HOST_ARENA_GIB_PER_P='64',
        MAX_PREFILL_INFLIGHT='12',
        P_TO_D_CONSUMERS='24',
        SGLANG_AGENTIC_KV_CUSTOM_STORAGE_ONLY='true',
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_RECOMPUTE='true',
        SGLANG_AGENTIC_KV_FAST_DIRECT_FAILURE_RECOMPUTE='false',
        SGLANG_AGENTIC_KV_FORCE_SLOW_PATH='false',
        SGLANG_AGENTIC_KV_DISABLE_D2P_REUSE='false',
        SGLANG_PD_ABLATION_P2D_PREBIND='false',
        SGLANG_PD_ABLATION_RANDOM_ROUTING='false',
        SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_BACKEND='memfd',
        SGLANG_AGENTIC_KV_SHARED_HOST_ARENA_GIB='128',
        SGLANG_AGENTIC_KV_P2D_HOST_ARENA_BACKEND='memfd',
        SGLANG_AGENTIC_KV_P2D_SHARED_HOST_ARENA_GIB='64',
        SGLANG_AGENTIC_KV_REGISTERED_EXTENT_DMA='1',
        SGLANG_AGENTIC_KV_REGISTER_EAGER_ARENA='1',
        SGLANG_AGENTIC_KV_REGISTER_STARTUP_BARRIER='1',
        SGLANG_AGENTIC_KV_REGISTER_PREWARM_TIMEOUT_SECONDS='1800',
        SGLANG_AGENTIC_KV_REGISTER_WINDOW_GIB='8',
        SGLANG_AGENTIC_KV_REGISTER_CACHE_GIB='768',
        SGLANG_AGENTIC_KV_D2H_INFLIGHT='2',
        P_H2D_MAX_INFLIGHT='2',
        SGLANG_AGENTIC_KV_P_H2D_MAX_INFLIGHT='2',
        SGLANG_AGENTIC_KV_P2D_D2H_WORKERS='2',
        SGLANG_AGENTIC_KV_P2D_H2D_WORKERS='2',
        SGLANG_AGENTIC_KV_D2H_CHUNK_TOKENS='4096',
        SGLANG_AGENTIC_KV_P_H2D_CHUNK_TOKENS='4096',
        SGLANG_AGENTIC_KV_P2D_D2H_CHUNK_TOKENS='4096',
        SGLANG_AGENTIC_KV_P2D_H2D_CHUNK_TOKENS='4096',
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_HIGH='32',
        SGLANG_AGENTIC_KV_SLOW_CONGESTION_LOW='8',
        DUALPD_ENGINE_ROOT=str(engine),
        MAX_CONTEXT_LENGTH='40960',
        MAX_RESPONSE_LENGTH='36864',
        TERM='dumb',
    )
    env.pop("HF_TOKEN", None)  # Cached inputs only; never persist a token.
    return env, ["bash", str(root / "scripts/new_method/run_4p4d_numa_case.sh")]
