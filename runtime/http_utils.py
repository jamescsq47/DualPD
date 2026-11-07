"""Original local HTTP retry and response lifecycle; no training dispatch."""
import asyncio
import json
import logging
import httpx
logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None
_client_concurrency = 0
_distributed_post_enabled = False


def init_http_client(args):
    global _http_client, _client_concurrency
    if getattr(args, "use_distributed_post", False):
        raise ValueError("DualPD inference uses local HTTP clients, not Ray dispatch")
    if not args.rollout_num_gpus:
        return
    _client_concurrency = (
        args.sglang_server_concurrency * args.rollout_num_gpus
        // args.rollout_num_gpus_per_engine
    )
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=_client_concurrency,
                max_keepalive_connections=_client_concurrency,
                keepalive_expiry=30,
            ),
            timeout=httpx.Timeout(
                connect=10, read=getattr(args, "sglang_router_request_timeout_secs", 120),
                write=10, pool=5,
            ),
            trust_env=False,
        )


async def post(url, payload, max_retries=60, headers=None):
    return await _post(_http_client, url, payload, max_retries, headers=headers)

async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output


async def get(url, max_retries=5):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await _http_client.get(url)
            response.raise_for_status()
            content = await response.aread()
            output = json.loads(content)
        except Exception as e:
            retry_count += 1
            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None
            logger.info(
                f"GET {url} failed: {e}, retrying ({retry_count}/{max_retries}, response={response_text})"
            )
            if retry_count >= max_retries:
                raise
            await asyncio.sleep(2)
            continue
        finally:
            if response is not None:
                await response.aclose()
        return output

