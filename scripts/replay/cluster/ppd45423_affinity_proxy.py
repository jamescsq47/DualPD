#!/usr/bin/env python3
"""Conversation-affine HTTP proxy for SGLang decode-local prefill experiments.

The first prompt of a trajectory is sent through a dedicated P->D router.
Longer prompts that retain that exact first-prompt token prefix are sent
directly to the same decode endpoint with ``disagg_local_prefill=true``.

This is intentionally an experiment-side proxy.  It does not modify SGLang or
the shared BrowseComp harness, and it records every routing decision so an
invalid affinity match cannot silently enter a benchmark result.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


@dataclass(frozen=True)
class Route:
    endpoint: str
    pair: int
    local_prefill: bool
    session_key: str


@dataclass(frozen=True)
class Session:
    initial_ids: tuple[int, ...]
    pair: int
    key: str


class TrieNode:
    __slots__ = ("children", "session")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.session: Session | None = None


def _key(token_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()[:20]


class AffinityProxy:
    def __init__(
        self,
        pd_endpoints: list[str],
        decode_endpoints: list[str],
        event_log: Path,
        mode: str,
    ) -> None:
        if not pd_endpoints or len(pd_endpoints) != len(decode_endpoints):
            raise ValueError("PD and decode endpoints must have the same non-zero length")
        if mode not in {"local", "ordinary"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.pd_endpoints = [value.rstrip("/") for value in pd_endpoints]
        self.decode_endpoints = [value.rstrip("/") for value in decode_endpoints]
        self.event_log = event_log
        self.mode = mode
        self.trie = TrieNode()
        self.session_count = 0
        self.exact_routes: dict[tuple[int, ...], Route] = {}
        self.route_lock = asyncio.Lock()
        self.log_lock = asyncio.Lock()
        self.client: ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        self.client = ClientSession(
            # c576 needs at least one inbound and one upstream socket per
            # in-flight request.  Bound the pool, but keep it above the formal
            # concurrency so requests are not serialized in the proxy.
            connector=TCPConnector(
                limit=2048, keepalive_timeout=30, enable_cleanup_closed=True
            ),
            timeout=ClientTimeout(total=None, connect=10, sock_read=3600),
        )

    async def stop(self, _app: web.Application) -> None:
        if self.client is not None:
            await self.client.close()

    async def choose_route(self, token_ids: tuple[int, ...]) -> Route:
        async with self.route_lock:
            previous = self.exact_routes.get(token_ids)
            if previous is not None:
                return previous

            parent: Session | None = None
            if self.mode == "local":
                node = self.trie
                for index, token in enumerate(token_ids):
                    node = node.children.get(token)
                    if node is None:
                        break
                    if node.session is not None and index + 1 < len(token_ids):
                        parent = node.session

            if parent is None:
                session_key = _key(token_ids)
                pair = int(session_key, 16) % len(self.pd_endpoints)
                route = Route(
                    endpoint=self.pd_endpoints[pair],
                    pair=pair,
                    local_prefill=False,
                    session_key=session_key,
                )
                session = Session(token_ids, pair, session_key)
                node = self.trie
                for token in token_ids:
                    node = node.children.setdefault(token, TrieNode())
                node.session = session
                self.session_count += 1
            else:
                route = Route(
                    endpoint=self.decode_endpoints[parent.pair],
                    pair=parent.pair,
                    local_prefill=True,
                    session_key=parent.key,
                )
            self.exact_routes[token_ids] = route
            return route

    async def append_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        async with self.log_lock:
            with self.event_log.open("a", encoding="utf-8") as handle:
                handle.write(line)

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "mode": self.mode,
                "pairs": len(self.pd_endpoints),
                "sessions": self.session_count,
                "requests": len(self.exact_routes),
                "open_fds": len(os.listdir("/proc/self/fd")),
                "fd_soft_limit": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            }
        )

    async def generate(self, request: web.Request) -> web.Response:
        started = time.monotonic()
        body = await request.json()
        input_ids = body.get("input_ids")
        if not isinstance(input_ids, list) or not input_ids:
            return web.json_response({"error": "input_ids must be a non-empty list"}, status=400)
        token_ids = tuple(int(token) for token in input_ids)
        route = await self.choose_route(token_ids)
        forwarded = dict(body)
        if route.local_prefill:
            forwarded["disagg_local_prefill"] = True
        else:
            forwarded.pop("disagg_local_prefill", None)

        assert self.client is not None
        status = 502
        response_body = b""
        response_headers: dict[str, str] = {}
        error: str | None = None
        try:
            async with self.client.post(
                f"{route.endpoint}/generate", json=forwarded
            ) as upstream:
                status = upstream.status
                response_body = await upstream.read()
                content_type = upstream.headers.get("Content-Type")
                if content_type:
                    response_headers["Content-Type"] = content_type
        except Exception as exc:  # logged and surfaced; never silently reroute
            error = f"{type(exc).__name__}: {exc}"
            response_body = json.dumps({"error": error}).encode()
            response_headers["Content-Type"] = "application/json"

        meta: dict[str, Any] = {}
        output_tokens = 0
        try:
            decoded = json.loads(response_body)
            if isinstance(decoded, dict):
                meta = decoded.get("meta_info") or {}
                output_tokens = len(decoded.get("output_ids") or [])
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        await self.append_event(
            {
                "epoch": time.time(),
                "elapsed_seconds": time.monotonic() - started,
                "status": status,
                "error": error,
                "mode": "decode_local_prefill" if route.local_prefill else "ordinary_pd",
                "pair": route.pair,
                "session_key": route.session_key,
                "input_tokens": len(token_ids),
                "prompt_tokens": meta.get("prompt_tokens"),
                "cached_tokens": meta.get("cached_tokens"),
                "output_tokens": output_tokens,
                "finish_reason": meta.get("finish_reason"),
            }
        )
        return web.Response(body=response_body, status=status, headers=response_headers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pd-endpoints", required=True)
    parser.add_argument("--decode-endpoints", required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "ordinary"), default="local")
    return parser.parse_args()


def main() -> None:
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_limit = min(hard_limit, 65536)
    if soft_limit < target_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
    args = parse_args()
    proxy = AffinityProxy(
        args.pd_endpoints.split(","),
        args.decode_endpoints.split(","),
        args.event_log,
        args.mode,
    )
    app = web.Application(client_max_size=128 * 1024**2)
    app.router.add_get("/health", proxy.health)
    app.router.add_post("/generate", proxy.generate)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
