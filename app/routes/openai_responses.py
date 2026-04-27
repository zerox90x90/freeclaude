"""OpenAI Responses API subset — /v1/responses."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.deepseek.client import DeepSeekClient
from app.routes.auth import require_bearer_key
from app.routes.openai_chat import (
    ChatMessage,
    canon_turns,
    collect_ref_file_ids,
    parse_model,
    prepend_system,
    resolve_session,
    _cache_turn,
)
from app.tools.inject import normalize_openai_tools, tool_system_block
from app.tools.parser import ToolCallParser
from app.tools.structured import structured_system_block

router = APIRouter()

_require_key = require_bearer_key


class ResponsesRequest(BaseModel):
    model: str
    input: Any  # string | list[dict]
    instructions: str | None = None
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: Any = None
    text: dict | None = None  # {"format":{"type":"json_object"|"json_schema", ...}}
    reasoning: dict | None = None
    max_output_tokens: int | None = None


def _input_to_messages(input_val: Any) -> list[ChatMessage]:
    if isinstance(input_val, str):
        return [ChatMessage(role="user", content=input_val)]
    messages: list[ChatMessage] = []
    if not isinstance(input_val, list):
        return messages
    for item in input_val:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "user")
        content = item.get("content", "")
        if isinstance(content, list):
            # Responses input_text -> OpenAI content[{type:text}]
            flat_parts: list[dict] = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t in ("input_text", "text"):
                    flat_parts.append({"type": "text", "text": p.get("text", "")})
                elif t == "input_file":
                    fid = p.get("file_id")
                    if fid:
                        flat_parts.append({"type": "file", "file": {"file_id": fid}})
            content = flat_parts or ""
        messages.append(ChatMessage(role=role if role in ("system", "user", "assistant", "tool") else "user", content=content))
    return messages


def _response_format_from_text(text: dict | None) -> dict | None:
    if not text:
        return None
    fmt = text.get("format")
    if not isinstance(fmt, dict):
        return None
    return fmt  # already OpenAI response_format shape


_JSON_SEPS = (",", ":")


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=_JSON_SEPS)}\n\n".encode()


@router.post("/v1/responses")
async def responses(req: ResponsesRequest, request: Request, _: None = Depends(_require_key)):
    base_model, thinking, search, mcp_servers = parse_model(req.model)
    # Reasoning override
    if req.reasoning:
        thinking = True

    messages = _input_to_messages(req.input)
    if req.instructions:
        messages = [ChatMessage(role="system", content=req.instructions)] + messages

    turns = canon_turns(messages)

    response_format = _response_format_from_text(req.text)
    tools_norm = normalize_openai_tools(req.tools or [])
    if tools_norm:
        turns = prepend_system(turns, tool_system_block(tools_norm))
    sb = structured_system_block(response_format)
    if sb:
        turns = prepend_system(turns, sb)

    client: DeepSeekClient = request.app.state.ds
    alias = request.headers.get("x-ds-session")
    session_id, parent_id, prompt = await resolve_session(client, turns, alias=alias)
    ref_file_ids = await collect_ref_file_ids(messages)
    has_tools = bool(tools_norm)

    if req.stream:
        return StreamingResponse(
            _stream_responses(
                client, session_id, parent_id, turns, prompt, thinking, search,
                req.model, has_tools, ref_file_ids, base_model, mcp_servers,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(
        await _buffered_responses(
            client, session_id, parent_id, turns, prompt, thinking, search,
            req.model, has_tools, ref_file_ids, base_model, mcp_servers,
        )
    )


async def _buffered_responses(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool,
    ref_file_ids: list[str] | None,
    base_model: str | None = None,
    mcp_servers: list[str] | None = None,
) -> dict:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    parser = ToolCallParser() if has_tools else None
    message_id: int | None = None

    async for ev in client.stream_completion(
        session_id=session_id, prompt=prompt, parent_message_id=parent_id,
        thinking=thinking, search=search, ref_file_ids=ref_file_ids,
        model=base_model, mcp_servers=mcp_servers,
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
            break
    if parser:
        for pev in parser.flush():
            if pev["type"] == "text":
                content_parts.append(pev["text"])
    content = "".join(content_parts)
    await _cache_turn(turns, content, session_id, message_id)

    output_items: list[dict] = []
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    if thinking_parts:
        output_items.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "".join(thinking_parts)}],
            }
        )
    if content:
        output_items.append(
            {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )
    for tc in tool_calls:
        output_items.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"]),
            }
        )

    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output_items,
        "output_text": content,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


async def _stream_responses(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool,
    ref_file_ids: list[str] | None,
    base_model: str | None = None,
    mcp_servers: list[str] | None = None,
) -> AsyncIterator[bytes]:
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "response.created",
        {"type": "response.created", "response": {"id": resp_id, "object": "response", "status": "in_progress", "model": model}},
    )

    parser = ToolCallParser() if has_tools else None
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    output_text: list[str] = []
    item_added = False
    tool_calls: list[dict] = []
    assistant_buf: list[str] = []

    def _text_delta(text: str) -> list[bytes]:
        """Emit text delta frames, opening the output item on first call."""
        nonlocal item_added
        frames: list[bytes] = []
        if not item_added:
            frames.append(_sse("response.output_item.added", {"type": "response.output_item.added", "output_index": 0, "item": {"id": item_id, "type": "message", "role": "assistant"}}))
            item_added = True
        assistant_buf.append(text)
        output_text.append(text)
        frames.append(_sse("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "delta": text}))
        return frames

    try:
        async for ev in client.stream_completion(
            session_id=session_id, prompt=prompt, parent_message_id=parent_id,
            thinking=thinking, search=search, ref_file_ids=ref_file_ids,
            model=base_model,
        ):
            if ev["type"] == "thinking":
                yield _sse(
                    "response.reasoning_summary_text.delta",
                    {"type": "response.reasoning_summary_text.delta", "delta": ev["text"]},
                )
                continue
            if ev["type"] == "done":
                if parser:
                    for pev in parser.flush():
                        if pev["type"] == "text":
                            for frame in _text_delta(pev["text"]):
                                yield frame
                if item_added:
                    yield _sse("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "text": "".join(output_text)})
                    yield _sse("response.output_item.done", {"type": "response.output_item.done", "output_index": 0, "item": {"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "".join(output_text)}]}})
                for tc in tool_calls:
                    yield _sse("response.output_item.done", {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": tc["id"], "name": tc["name"], "arguments": json.dumps(tc["arguments"])}})
                yield _sse(
                    "response.completed",
                    {"type": "response.completed", "response": {"id": resp_id, "status": "completed", "output_text": "".join(output_text)}},
                )
                break
            if ev["type"] != "content":
                continue
            text = ev["text"]
            if not parser:
                for frame in _text_delta(text):
                    yield frame
                continue
            for pev in parser.feed(text):
                if pev["type"] == "text":
                    for frame in _text_delta(pev["text"]):
                        yield frame
                elif pev["type"] == "tool_call":
                    tool_calls.append(pev)
    except Exception as e:
        yield _sse("response.error", {"type": "response.error", "error": {"message": str(e)}})
    else:
        await _cache_turn(turns, "".join(assistant_buf), session_id, None)
