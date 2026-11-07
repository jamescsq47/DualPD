"""Exact-token, multi-turn GAIA-text harness with audited PD lifecycle hooks."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from dualpd.agentic_kv_request import (
    add_agentic_kv_metadata,
    build_agentic_extra_key,
    confirm_agentic_generation_final,
    confirm_agentic_generation_tool,
    generation_has_visible_content,
    lifecycle_enabled,
)
from dualpd.pd_metrics import sglang_meta_attrs
from dualpd.runtime.tracing import span as dashboard_span
from dualpd.runtime.generation import GenerateState
from dualpd.runtime.http_utils import post
from dualpd.runtime.types import Sample

from .tools import TOOL_SPECS, WebBackendError, WebTools, execute_call, validate_call


LOG = logging.getLogger(__name__)
_FINAL = re.compile(r"<final_answer>\s*(.*?)\s*</final_answer>", re.DOTALL)


def parse_action(text: str) -> tuple[str, Any]:
    visible = text.rsplit("</think>", 1)[-1]
    finals = _FINAL.findall(visible)
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", visible, flags=re.DOTALL)
    if finals and blocks:
        raise ValueError("A turn cannot contain both a final answer and a tool call")
    if finals:
        if len(finals) != 1:
            raise ValueError("Return exactly one final_answer block")
        return "final", finals[0].strip()
    if "<tool_call>" in visible and not blocks:
        raise ValueError("Unclosed tool_call")
    if not blocks:
        raise ValueError("Return tool calls or one final_answer block")
    if len(blocks) > 8:
        raise ValueError("At most 8 tool calls are allowed per model turn")
    calls = [json.loads(block) for block in blocks]
    for call in calls:
        validate_call(call)
    return "tools", calls


def initial_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages,
        tools=TOOL_SPECS,
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def observation_tokens(tokenizer: Any, observations: list[Any]) -> list[int]:
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "x"},
    ]
    prefix_text = tokenizer.apply_chat_template(
        base, tokenize=False, add_generation_prompt=False
    )
    messages = base + [
        {"role": "tool", "content": json.dumps(value, ensure_ascii=False)}
        for value in observations
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix = list(tokenizer.encode(prefix_text, add_special_tokens=False))
    full = list(tokenizer.encode(full_text, add_special_tokens=False))
    if full[: len(prefix)] != prefix:
        raise ValueError("Tokenizer tool template is not append-only")
    return full[len(prefix) :]


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    metadata = dict(sample.metadata or {})
    sample.metadata = metadata
    dataset_id = str(metadata.get("dataset_id") or "gaia")
    options = getattr(args, "workload_dataset_options", {}).get(dataset_id, {})
    state = GenerateState(args)
    initial = initial_tokens(state.tokenizer, list(sample.prompt))
    response: list[int] = []
    web = WebTools(
        backend=str(options.get("search_backend", "auto")),
        snippets=True,
        timeout=int(options.get("tool_timeout_seconds", 20)),
        max_chars=int(options.get("page_max_chars", 16000)),
    )
    max_turns = int(options.get("max_turns", 16))
    per_turn_cap = int(options.get("max_tokens_per_turn", 8192))
    total_budget = int(options.get("max_response_tokens", 32768))
    context_limit = int(
        getattr(args, "max_seq_len", None)
        or getattr(args, "sglang_context_length", None)
        or 40960
    )
    max_tool_errors = max(1, int(options.get("max_tool_errors", 3)))
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    started = time.monotonic()
    completed_turns = tool_tokens = tool_errors = 0
    search_calls = code_calls = 0
    tool_seconds = 0.0
    last_generation: int | None = None
    stop_reason = "max_turns"
    status = Sample.Status.TRUNCATED
    params_base = dict(sampling_params)

    try:
        for turn in range(max_turns):
            remaining = min(
                total_budget - len(response),
                context_limit - len(initial) - len(response) - 1,
            )
            if remaining <= 0:
                stop_reason = "budget"
                break
            params = dict(params_base)
            params["max_new_tokens"] = min(per_turn_cap, remaining)
            request_id = None
            if lifecycle_enabled():
                params, request_id = add_agentic_kv_metadata(
                    params,
                    trajectory_metadata=metadata,
                    generation=turn,
                    tokenizer=state.tokenizer,
                    tool_type="gaia",
                    tool_suffix_markers=("</tool_call>",),
                    terminal_markers=("</final_answer>",),
                )
            payload: dict[str, Any] = {
                "input_ids": initial + response,
                "sampling_params": params,
                "return_logprob": False,
            }
            if request_id is not None:
                payload["extra_key"] = build_agentic_extra_key(request_id, params)
            with dashboard_span(
                args,
                sample,
                "generation_turn",
                attrs={
                    "task_type": "gaia_text",
                    "turn": turn + 1,
                    "max_new_tokens": params["max_new_tokens"],
                    "route_mode": "strict_pd" if lifecycle_enabled() else "colocated",
                },
            ) as span:
                output = await post(url, payload)
                span.update(sglang_meta_attrs(output.get("meta_info", {})))
            last_generation = turn
            meta = output["meta_info"]
            finish_type = meta["finish_reason"]["type"]
            ids = [int(value) for value in (output.get("output_ids") or [])]
            response.extend(ids)
            completed_turns += 1
            if finish_type == "abort":
                status, stop_reason = Sample.Status.ABORTED, "abort"
                break
            if not generation_has_visible_content(ids, state.tokenizer):
                status, stop_reason = Sample.Status.TRUNCATED, "empty_generation"
                break
            if finish_type == "length":
                status, stop_reason = Sample.Status.TRUNCATED, "length"
                break

            text = output.get("text", "")
            try:
                action, value = parse_action(text)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                action = "tools"
                value = [None]
                observations = [
                    {
                        "error": str(exc),
                        "instruction": "Correct the format and retry with a tool call or final_answer.",
                    }
                ]
                metadata["format_repairs"] = int(metadata.get("format_repairs", 0)) + 1

            if action == "final":
                metadata["predicted_answer"] = value
                confirm_agentic_generation_final(
                    metadata,
                    turn,
                    p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
                )
                status, stop_reason = Sample.Status.COMPLETED, "final"
                break

            confirm_agentic_generation_tool(
                metadata,
                turn,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )
            if value != [None]:
                observations = []
                for call in value:
                    tool_started = time.monotonic()
                    try:
                        result = await execute_call(web, call)
                    except WebBackendError as exc:
                        tool_errors += 1
                        result = {
                            "error": str(exc),
                            "instruction": (
                                "This web call failed. Try a different query or source, use another tool, "
                                "or provide the best supported final answer."
                            ),
                        }
                        if tool_errors >= max_tool_errors:
                            result["instruction"] = (
                                "The live web backend is currently unreliable. Do not repeat the same "
                                "call; use existing evidence, Python if useful, or return a final answer."
                            )
                    finally:
                        tool_seconds += time.monotonic() - tool_started
                    observations.append(result)
                    if call["name"] == "search_engine_query":
                        search_calls += 1
                    elif call["name"] == "code_interpreter":
                        code_calls += 1
            suffix = observation_tokens(state.tokenizer, observations)
            if ids and ids[-1] == getattr(state.tokenizer, "eos_token_id", None):
                suffix = list(state.tokenizer.encode("\n", add_special_tokens=False)) + suffix
            available = min(
                total_budget - len(response),
                context_limit - len(initial) - len(response) - 1,
            )
            if len(suffix) > available:
                status, stop_reason = Sample.Status.TRUNCATED, "tool_context_budget"
                break
            response.extend(suffix)
            tool_tokens += len(suffix)
        else:
            status, stop_reason = Sample.Status.TRUNCATED, "max_turns"
    except asyncio.CancelledError:
        status, stop_reason = Sample.Status.ABORTED, "cancelled"
        raise
    except Exception as exc:
        status = Sample.Status.FAILED
        stop_reason = f"tool_or_harness_error:{type(exc).__name__}"
        metadata["harness_error"] = str(exc)
    finally:
        if last_generation is not None:
            confirm_agentic_generation_final(
                metadata,
                last_generation,
                p_ready_dir=str(getattr(args, "pd_p_ready_dir", "") or ""),
            )

    sample.status = status
    sample.tokens = initial + response
    sample.response = state.tokenizer.decode(response, skip_special_tokens=False)
    sample.response_length = len(response)
    sample.sample_time = time.monotonic() - started
    sample.tool_time = tool_seconds
    sample.tool_token_count = tool_tokens
    sample.tool_call_count = len(web.events) + code_calls
    sample.search_call_count = search_calls
    sample.code_call_count = code_calls
    metadata.update(
        {
            "num_turns": completed_turns,
            "stop_reason": stop_reason,
            "web_tool_events": web.events,
            "search_backend": web.backend,
            "tool_error_count": tool_errors,
            "tool_time": tool_seconds,
            "sample_time": sample.sample_time,
        }
    )
    LOG.info(
        "GAIA sample=%s turns=%d stop=%s web=%d code=%d tool_seconds=%.3f",
        sample.index,
        completed_turns,
        stop_reason,
        len(web.events),
        code_calls,
        tool_seconds,
    )
    return sample
