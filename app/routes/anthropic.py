"""Anthropic Messages API /v1/messages — stream + non-stream."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.routes.auth import require_any_key
from app.deepseek import files as ds_files
from app.deepseek.client import DeepSeekClient
from app.routes.openai_chat import (
    build_cache_text,
    parse_model,
    prepend_system,
    resolve_session,
    _cache_turn,
)
from app.tools.inject import normalize_anthropic_tools, tool_system_block
from app.tools.parser import CLOSE as TOOL_CLOSE, OPEN as TOOL_OPEN, ToolCallParser
from app.tools.prune import prune_tool_result

router = APIRouter()
log = logging.getLogger(__name__)

PING_INTERVAL = 5.0


def _estimate_tokens(text: str) -> int:
    # Rough 1 token ≈ 4 chars; good enough for Claude Code's usage display.
    # Neither DeepSeek nor Z.AI exposes a tokenizer so the bar is always an
    # approximation — slightly under-counts code/JSON, slightly over-counts prose.
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_input_tokens(turns: list[tuple[str, str]], tools: list[dict] | None) -> int:
    """Estimate the FULL input token cost (history + tools), not just the new
    prompt. Claude Code's usage bar reads `input_tokens` so reporting only the
    last user turn after a cache hit makes the bar swing wildly."""
    body = "\n\n".join(t for _, t in turns)
    tools_text = json.dumps(tools or []) if tools else ""
    return _estimate_tokens(body) + _estimate_tokens(tools_text)


_require_key = require_any_key


# ---- schema ----

class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant"]
    content: str | list[dict]


class AnthropicRequest(BaseModel):
    # Claude Code sends many optional fields we don't need to act on.
    # `extra=ignore` keeps the proxy from 422'ing on new/beta params.
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    max_tokens: int = 1024
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    metadata: dict | None = None
    service_tier: str | None = None
    tool_choice: dict | None = None
    tools: list[dict] | None = None
    thinking: dict | None = None  # {"type":"enabled", "budget_tokens":...}


def _anthropic_system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for p in system:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n\n".join(parts)


def _flatten_block(block: dict) -> str:
    t = block.get("type")
    if t == "text":
        return block.get("text", "")
    if t == "tool_use":
        return (
            f"{TOOL_OPEN}\n"
            f"{json.dumps({'name': block.get('name', ''), 'arguments': block.get('input', {})})}"
            f"\n{TOOL_CLOSE}"
        )
    if t == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if not isinstance(c, dict):
                    continue
                ct = c.get("type")
                if ct == "text":
                    parts.append(c.get("text", ""))
                elif ct == "image":
                    parts.append("[image omitted]")
            content = "\n".join(p for p in parts if p)
        err = " error" if block.get("is_error") else ""
        tid = block.get("tool_use_id", "")
        return f"<tool_result id=\"{tid}\"{err}>\n{prune_tool_result(content)}\n</tool_result>"
    if t == "thinking" or t == "redacted_thinking":
        return ""  # don't replay thinking to the model
    if t == "image":
        return "[image omitted]"
    if t == "document":
        # File is passed separately via ref_file_ids; hint in prompt.
        src = block.get("source", {})
        if isinstance(src, dict) and src.get("type") == "file":
            return "[See attached document.]"
        return "[document omitted]"
    return ""


async def _collect_ref_files(req: "AnthropicRequest") -> list[str]:
    out: list[str] = []
    for m in req.messages:
        if not isinstance(m.content, list):
            continue
        for b in m.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "document":
                src = b.get("source", {})
                fid = src.get("file_id") if isinstance(src, dict) else None
                if not fid:
                    continue
                info = await ds_files.get_mapping(fid)
                if info and info.get("deepseek_file_id"):
                    out.append(info["deepseek_file_id"])
                elif fid.startswith("file-"):
                    out.append(fid)
    return out


def _anthropic_to_turns(req: AnthropicRequest) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    sys_text = _anthropic_system_text(req.system)
    if sys_text:
        turns.append(("system", sys_text))
    for m in req.messages:
        if isinstance(m.content, str):
            turns.append((m.role, m.content))
            continue
        parts = [_flatten_block(b) for b in m.content if isinstance(b, dict)]
        text = "\n".join(p for p in parts if p).strip()
        if text:
            turns.append((m.role, text))
    return turns


# ---- SSE helpers ----

_JSON_SEPS = (",", ":")


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=_JSON_SEPS)}\n\n".encode()


@router.post("/v1/messages/count_tokens")
async def count_tokens(req: AnthropicRequest, _: None = Depends(_require_key)):
    # Claude Code calls this to size prompts before sending. Estimate from
    # the flattened prompt since the upstream doesn't expose a tokenizer.
    turns = _anthropic_to_turns(req)
    body = "\n\n".join(t for _, t in turns)
    tools_text = json.dumps(req.tools or [])
    return {"input_tokens": _estimate_tokens(body) + _estimate_tokens(tools_text)}


@router.post("/v1/messages")
async def messages(req: AnthropicRequest, request: Request, _: None = Depends(_require_key)):
    # Reuse shared suffix parser (handles :search, :think, :nothink, MCP aliases)
    base_model, thinking_enabled, search, mcp_servers = parse_model(req.model)
    # Anthropic thinking param and model-name heuristics override parse_model defaults.
    # Check against the RAW model name since Claude Code sends model strings like
    # "claude-3-opus" that parse_model canonicalizes away to "deepseek-chat".
    model_lower = req.model.lower()
    is_reasoner = "reasoner" in model_lower or "thinking" in model_lower or "opus" in model_lower
    if req.thinking and req.thinking.get("type") == "enabled":
        thinking_enabled = True
    elif is_reasoner:
        thinking_enabled = True

    turns = _anthropic_to_turns(req)

    tools_norm = normalize_anthropic_tools(req.tools or [])
    # tool_choice: {"type":"any"} ~ "required"; {"type":"tool","name":...} forces one
    force_note = ""
    if isinstance(req.tool_choice, dict):
        tc_type = req.tool_choice.get("type")
        if tc_type == "none":
            tools_norm = []
        elif tc_type == "any":
            force_note = "\n\nYou MUST call one of the tools above. Do not answer directly."
        elif tc_type == "tool" and req.tool_choice.get("name"):
            force_note = (
                f"\n\nYou MUST call the `{req.tool_choice['name']}` tool. "
                "Do not answer directly."
            )
    if tools_norm:
        block = tool_system_block(tools_norm) + force_note
        turns = prepend_system(turns, block)

    # Inline cached text-file content into the last user turn (reliability)
    file_text_parts: list[str] = []
    for m in req.messages:
        if not isinstance(m.content, list):
            continue
        for b in m.content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "document":
                src = b.get("source", {})
                fid = src.get("file_id") if isinstance(src, dict) else None
                if not fid:
                    continue
                info = await ds_files.get_mapping(fid)
                if info and info.get("text"):
                    file_text_parts.append(
                        f"--- FILE: {info.get('filename', fid)} ---\n{info['text']}\n--- END FILE ---"
                    )
    if file_text_parts and turns:
        ft = "\n\n".join(file_text_parts)
        role, content = turns[-1]
        if role == "user":
            turns = turns[:-1] + [(role, f"{ft}\n\n{content}")]
        else:
            turns = turns + [("user", ft)]

    client: DeepSeekClient = request.app.state.ds
    alias = request.headers.get("x-ds-session")
    session_id, parent_id, prompt = await resolve_session(client, turns, alias=alias)
    # Cache-hit => prompt is just the new user turn, which means DeepSeek no
    # longer sees the tool contract we injected into the flattened transcript.
    # Re-prepend it so the <tool_call> protocol survives across turns.
    if tools_norm and turns and prompt == turns[-1][1]:
        prompt = tool_system_block(tools_norm) + force_note + "\n\n" + prompt
    ref_file_ids = await _collect_ref_files(req)
    has_tools = bool(tools_norm)

    input_tokens = _estimate_input_tokens(turns, req.tools)
    if req.stream:
        return StreamingResponse(
            _stream_anthropic(
                client, session_id, parent_id, turns, prompt, thinking_enabled, search,
                req.model, has_tools, ref_file_ids, base_model, mcp_servers,
                input_tokens,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(
        await _buffered_anthropic(
            client, session_id, parent_id, turns, prompt, thinking_enabled, search,
            req.model, has_tools, ref_file_ids, base_model, mcp_servers,
            input_tokens,
        )
    )


async def _stream_anthropic(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | str | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool,
    ref_file_ids: list[str] | None = None,
    base_model: str | None = None,
    mcp_servers: list[str] | None = None,
    input_tokens: int = 0,
) -> AsyncIterator[bytes]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    parser = ToolCallParser() if has_tools else None
    block_idx = -1
    current_block: str | None = None  # "thinking" | "text" | "tool_use"
    assistant_buf: list[str] = []
    tool_calls: list[dict] = []
    stop_reason = "end_turn"
    final_msg_id: int | None = None
    final_session_id: str = session_id

    def open_block(kind: str, extra: dict) -> bytes:
        nonlocal block_idx, current_block
        block_idx += 1
        current_block = kind
        return _sse(
            "content_block_start",
            {"type": "content_block_start", "index": block_idx, "content_block": extra},
        )

    def close_block() -> bytes:
        nonlocal current_block
        if current_block is None:
            return b""
        out = _sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})
        current_block = None
        return out

    # Drive the upstream stream from a background task and feed events through
    # a queue. This lets us emit `ping` heartbeats independently of upstream
    # liveness — critical because PoW solve, session creation, and prefix
    # compression all happen inside `stream_completion()` BEFORE the first
    # upstream byte arrives. Without an out-of-band ping the SSE channel goes
    # silent for 30-90s on big tasks and Claude Code aborts the request.
    queue: asyncio.Queue = asyncio.Queue()

    async def _producer() -> None:
        try:
            async for ev in client.stream_completion(
                session_id=session_id,
                prompt=prompt,
                parent_message_id=parent_id,
                thinking=thinking,
                search=search,
                ref_file_ids=ref_file_ids,
                model=base_model,
                mcp_servers=mcp_servers,
            ):
                await queue.put(("ev", ev))
        except Exception as e:  # surfaced to the consumer as a visible error
            log.warning("anthropic stream upstream error: %s", e)
            await queue.put(("err", str(e)))
        finally:
            await queue.put(("end", None))

    producer = asyncio.create_task(_producer())
    pending_get: asyncio.Task | None = None

    try:
      while True:
        if pending_get is None:
            pending_get = asyncio.create_task(queue.get())
        try:
            kind, val = await asyncio.wait_for(
                asyncio.shield(pending_get), timeout=PING_INTERVAL
            )
            pending_get = None
        except asyncio.TimeoutError:
            yield _sse("ping", {"type": "ping"})
            continue

        if kind == "end":
            # Producer finished without an explicit `done` event (e.g. after
            # an err frame, or upstream closed early). Wrap up gracefully.
            if current_block:
                yield close_block()
            output_tokens = _estimate_tokens("".join(assistant_buf))
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})
            break

        if kind == "err":
            # Make the failure visible inside the response instead of leaving
            # Claude Code with a blank turn after a silent abort.
            if current_block and current_block != "text":
                yield close_block()
            if current_block != "text":
                yield open_block("text", {"type": "text", "text": ""})
            err_text = f"\n\n[proxy: upstream error: {val}]"
            assistant_buf.append(err_text)
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": err_text},
                },
            )
            # also surface the raw error event for SDKs that key on it
            yield _sse(
                "error",
                {"type": "error", "error": {"type": "upstream_error", "message": str(val)}},
            )
            continue  # producer will follow with `end`

        ev = val
        if ev["type"] == "thinking":
            if current_block != "thinking":
                if current_block:
                    yield close_block()
                yield open_block("thinking", {"type": "thinking", "thinking": ""})
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "thinking_delta", "thinking": ev["text"]},
                },
            )
            continue
        if ev["type"] == "done":
            final_msg_id = ev.get("message_id")
            final_session_id = ev.get("session_id") or session_id
            if parser:
                for pev in parser.flush():
                    if pev["type"] == "text":
                        if current_block != "text":
                            if current_block:
                                yield close_block()
                            yield open_block("text", {"type": "text", "text": ""})
                        assistant_buf.append(pev["text"])
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {"type": "text_delta", "text": pev["text"]},
                            },
                        )
            if current_block:
                yield close_block()
            if tool_calls:
                stop_reason = "tool_use"
            output_tokens = _estimate_tokens("".join(assistant_buf))
            yield _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            yield _sse("message_stop", {"type": "message_stop"})
            cache_text = build_cache_text("".join(assistant_buf), tool_calls)
            await _cache_turn(turns, cache_text, final_session_id, final_msg_id)
            break
        if ev["type"] != "content":
            continue

        text = ev["text"]
        if not parser:
            if current_block == "thinking":
                yield close_block()
            if current_block != "text":
                yield open_block("text", {"type": "text", "text": ""})
            assistant_buf.append(text)
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "text_delta", "text": text},
                },
            )
            continue

        for pev in parser.feed(text):
            if pev["type"] == "text":
                if current_block == "thinking":
                    yield close_block()
                if current_block != "text":
                    yield open_block("text", {"type": "text", "text": ""})
                assistant_buf.append(pev["text"])
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": pev["text"]},
                    },
                )
            elif pev["type"] == "tool_call_start":
                if current_block:
                    yield close_block()
                yield open_block(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "id": pev["id"],
                        "name": pev["name"],
                        "input": {},
                    },
                )
            elif pev["type"] == "tool_call_arg_delta":
                if current_block != "tool_use":
                    continue
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": pev["delta"],
                        },
                    },
                )
            elif pev["type"] == "tool_call_end":
                if current_block == "tool_use":
                    yield close_block()
                tool_calls.append(pev)
    finally:
        if pending_get is not None and not pending_get.done():
            pending_get.cancel()
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except (asyncio.CancelledError, Exception):
            pass


async def _buffered_anthropic(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | str | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool,
    ref_file_ids: list[str] | None = None,
    base_model: str | None = None,
    mcp_servers: list[str] | None = None,
    input_tokens: int = 0,
) -> dict:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    message_id: int | str | None = None
    final_session_id: str = session_id
    parser = ToolCallParser() if has_tools else None

    async for ev in client.stream_completion(
        session_id=session_id,
        prompt=prompt,
        parent_message_id=parent_id,
        thinking=thinking,
        search=search,
        ref_file_ids=ref_file_ids,
        model=base_model,
        mcp_servers=mcp_servers,
    ):
        if ev["type"] == "thinking":
            thinking_parts.append(ev["text"])
        elif ev["type"] == "content":
            if not parser:
                content_parts.append(ev["text"])
            else:
                for pev in parser.feed(ev["text"]):
                    if pev["type"] == "text":
                        content_parts.append(pev["text"])
                    elif pev["type"] == "tool_call":
                        tool_calls.append(pev)
        elif ev["type"] == "done":
            message_id = ev["message_id"]
            final_session_id = ev.get("session_id") or session_id
            break
    if parser:
        for pev in parser.flush():
            if pev["type"] == "text":
                content_parts.append(pev["text"])

    content = "".join(content_parts)
    cache_text = build_cache_text(content, tool_calls)
    await _cache_turn(turns, cache_text, final_session_id, message_id)

    blocks: list[dict] = []
    if thinking_parts:
        blocks.append({"type": "thinking", "thinking": "".join(thinking_parts)})
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in tool_calls:
        blocks.append(
            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
        )

    output_tokens = _estimate_tokens(content) + _estimate_tokens("".join(thinking_parts))
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
        },
    }
