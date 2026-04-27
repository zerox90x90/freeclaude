"""OpenAI-compatible /v1/chat/completions + /v1/models."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import BACKEND
from app.routes.auth import require_bearer_key
from app.deepseek import files as ds_files
from app.deepseek import sessions
from app.deepseek.client import DeepSeekClient
from app.deepseek.compress import maybe_compress
from app.tools.inject import normalize_openai_tools, tool_system_block
from app.tools.parser import CLOSE as TOOL_CLOSE, OPEN as TOOL_OPEN, ToolCallParser, serialize_tool_calls
from app.tools.prune import prune_tool_result
from app.tools.structured import structured_system_block, validate_structured

router = APIRouter()


# ---- auth ----

_require_key = require_bearer_key


# ---- schema ----

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    prefix: bool | None = None  # beta
    reasoning_content: str | None = None  # accepted & dropped per official guidance


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    frequency_penalty: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    response_format: dict | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    temperature: float | None = None
    top_p: float | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    thinking: dict | None = None  # {"type":"enabled"|"disabled"}


# ---- model suffix parser ----

# MCP suffix -> Z.AI MCP server id (see /api/models meta.mcpServerIds)
_MCP_SUFFIXES: dict[str, str] = {
    "deepresearch":  "deep-research",
    "advsearch":     "advanced-search",
    "deepwebsearch": "deep-web-search",
    "imagesearch":   "image-search",
    "ppt":           "ppt-maker",
    "vibecoding":    "vibe-coding",
}


def parse_model(name: str) -> tuple[str, bool, bool, list[str]]:
    """Return (base, thinking, search, mcp_servers). Suffixes compose:
    `:search`, `:think`, `:nothink`, and Z.AI MCP aliases (`:deepresearch`,
    `:advsearch`, `:deepwebsearch`, `:imagesearch`, `:ppt`, `:vibecoding`).

    DeepSeek: `deepseek-reasoner` implies thinking=True; `deepseek-chat` off.
               MCP suffixes are accepted but ignored.
    Z.AI:     thinking defaults on for reasoning-capable GLM-5 variants;
              `:nothink` disables it. `:search` toggles upstream web search.
    """
    search = False
    force_think: bool | None = None
    mcp_servers: list[str] = []
    # Parse trailing :suffixes (order-independent within the trailing run)
    while ":" in name:
        base, _, suffix = name.rpartition(":")
        s = suffix.lower()
        if s == "search":
            search = True
        elif s == "think":
            force_think = True
        elif s == "nothink":
            force_think = False
        elif s in _MCP_SUFFIXES:
            srv = _MCP_SUFFIXES[s]
            if srv not in mcp_servers:
                mcp_servers.append(srv)
        else:
            break
        name = base

    if BACKEND == "zai":
        lname = name.lower()
        thinking_default = lname in (
            "glm-5.1", "glm-5-1", "glm-5", "glm-5-turbo",
        )
        thinking = force_think if force_think is not None else thinking_default
        base = name if name else "glm-5-turbo"
    else:
        thinking_default = name == "deepseek-reasoner"
        thinking = force_think if force_think is not None else thinking_default
        base = name if name in ("deepseek-chat", "deepseek-reasoner") else "deepseek-chat"
        mcp_servers = []  # DeepSeek backend has no MCP server concept
    return base, thinking, search, mcp_servers


# ---- message flattening ----

def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)


def _extract_file_ids(content: Any) -> list[str]:
    """Pull file_id refs out of OpenAI-style content parts."""
    if not isinstance(content, list):
        return []
    ids = []
    for p in content:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t == "file":
            f = p.get("file", {})
            fid = f.get("file_id") if isinstance(f, dict) else None
            if fid:
                ids.append(fid)
        elif t == "image_file":
            fid = p.get("image_file", {}).get("file_id")
            if fid:
                ids.append(fid)
    return ids


def canon_turns(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    """Flatten OpenAI messages into (role, text) pairs. Past tool calls are
    re-serialized into the <tool_call> convention; tool results are shown as
    `Tool (id=...): ...` blocks so the model can read prior execution state.
    """
    out: list[tuple[str, str]] = []
    for m in messages:
        text = _flatten_content(m.content)
        extras: list[str] = []
        if m.tool_calls:
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                    except json.JSONDecodeError:
                        args_obj = args
                else:
                    args_obj = args
                extras.append(
                    f"{TOOL_OPEN}\n"
                    f"{json.dumps({'name': name, 'arguments': args_obj})}"
                    f"\n{TOOL_CLOSE}"
                )
        if extras:
            text = (text + "\n" + "\n".join(extras)).strip()
        if m.role == "tool":
            tid = m.tool_call_id or ""
            text = f"<tool_result id=\"{tid}\">\n{prune_tool_result(text)}\n</tool_result>"
        if text:
            out.append((m.role, text))
    return out


_ROLE_TAGS = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}


def flatten_prefix(turns: list[tuple[str, str]]) -> str:
    """Render a transcript for the very first call when no cached session exists.

    Uses bracketed role markers (not bare `User:` / `Assistant:` labels) so the
    model doesn't fall into "continue the transcript" mode and echo prior turns
    back verbatim. No trailing `Assistant:` cue — DeepSeek is already replying
    as the assistant on a completion call.
    """
    if not turns:
        return ""
    parts = []
    for r, c in turns:
        t = _ROLE_TAGS.get(r, r.upper())
        parts.append(f"[{t}]\n{c}\n[/{t}]")
    return "\n\n".join(parts)


def prepend_system(turns: list[tuple[str, str]], block: str) -> list[tuple[str, str]]:
    if not block:
        return turns
    return [("system", block)] + turns


async def collect_ref_file_ids(messages: list[ChatMessage]) -> list[str]:
    """Walk messages, find file_id references, resolve to DeepSeek file-<uuid>."""
    out: list[str] = []
    for m in messages:
        for fid in _extract_file_ids(m.content):
            info = await ds_files.get_mapping(fid)
            if info and info.get("deepseek_file_id"):
                out.append(info["deepseek_file_id"])
            elif fid.startswith("file-"):
                out.append(fid)  # allow passing DeepSeek IDs directly
    return out


async def inline_file_text(messages: list[ChatMessage]) -> str:
    """For small text files we've cached, inline their contents so the model
    reliably sees them (DeepSeek's ref_file_ids pipeline is flaky for text)."""
    parts: list[str] = []
    for m in messages:
        for fid in _extract_file_ids(m.content):
            info = await ds_files.get_mapping(fid)
            if not info:
                continue
            text = info.get("text")
            if text:
                parts.append(
                    f"--- FILE: {info.get('filename', fid)} ---\n{text}\n--- END FILE ---"
                )
    return "\n\n".join(parts)


# ---- OpenAI SSE frame helpers ----

_JSON_SEPS = (",", ":")


def _sse(data: dict | str) -> bytes:
    if isinstance(data, dict):
        data = json.dumps(data, separators=_JSON_SEPS)
    return f"data: {data}\n\n".encode()


SYSTEM_FINGERPRINT = "fp_deepseek_proxy_v1"


def _chunk(id_: str, model: str, delta: dict, finish_reason: str | None = None, created: int | None = None) -> dict:
    return {
        "id": id_,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
    }


def _usage(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int = 0) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": prompt_tokens,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }


# ---- routes ----

@router.get("/v1/models")
def list_models(_: None = Depends(_require_key)):
    if BACKEND == "zai":
        ids = (
            "glm-5.1", "glm-5-turbo", "glm-5v-turbo",
            "glm-5.1:search", "glm-5-turbo:search",
            "glm-5.1:nothink", "glm-5-turbo:nothink",
            "glm-5.1:deepresearch", "glm-5.1:advsearch",
            "glm-5.1:search:deepresearch",
        )
        owner = "z.ai"
    else:
        ids = (
            "deepseek-chat", "deepseek-reasoner",
            "deepseek-chat:search", "deepseek-reasoner:search",
        )
        owner = "deepseek"
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": owner} for i in ids]}


async def resolve_session(
    client: DeepSeekClient,
    turns: list[tuple[str, str]],
    alias: str | None = None,
) -> tuple[str, int | str | None, str]:
    """Return (session_id, parent_message_id, prompt_to_send).

    If the entire history except the trailing user turn matches a cached
    conversation, reuse it and send only the new user turn. Otherwise open a
    fresh session and flatten the whole history into the single prompt.

    If `alias` is provided and resolves to a pinned session, reuse that
    session directly with the trailing user turn — regardless of whether the
    client's local history matches. This lets callers pick up a conversation
    mid-stream without replaying prior turns.
    """
    if not turns:
        raise HTTPException(400, "messages empty")
    if turns[-1][0] != "user":
        # Odd: client asked for completion but last message isn't user. Flatten.
        sid = await client.create_session()
        return sid, None, flatten_prefix(turns)

    new_user = turns[-1][1]

    if alias:
        pinned = await sessions.get_alias(alias)
        if pinned:
            return pinned["session_id"], pinned["parent_message_id"], new_user

    prefix = turns[:-1]
    if prefix:
        prefix_hash = sessions.hash_turns(prefix)
        cached = await sessions.get(prefix_hash)
        if cached:
            return cached["session_id"], cached["parent_message_id"], new_user

    # Cache miss on a potentially long history — try compression, then re-check.
    compressed = await maybe_compress(client, turns)
    if compressed is not turns and len(compressed) >= 1 and compressed[-1][0] == "user":
        c_prefix = compressed[:-1]
        if c_prefix:
            c_hash = sessions.hash_turns(c_prefix)
            c_cached = await sessions.get(c_hash)
            if c_cached:
                return c_cached["session_id"], c_cached["parent_message_id"], compressed[-1][1]
        turns = compressed

    sid = await client.create_session()
    return sid, None, flatten_prefix(turns)


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, _: None = Depends(_require_key)):
    model, thinking, search, mcp_servers = parse_model(req.model)
    # Explicit thinking param overrides model suffix
    if req.thinking:
        t = req.thinking.get("type")
        if t == "enabled":
            thinking = True
        elif t == "disabled":
            thinking = False
    turns = canon_turns(req.messages)

    # Resolve tool_choice: "none" strips tools; specific function forces call
    tools_norm = normalize_openai_tools(req.tools or [])
    force_tool: str | None = None
    if req.tool_choice == "none":
        tools_norm = []
    elif isinstance(req.tool_choice, dict):
        fn = req.tool_choice.get("function", {})
        if isinstance(fn, dict) and fn.get("name"):
            force_tool = fn["name"]
    elif req.tool_choice == "required":
        force_tool = "*"  # any

    # Inject tool + structured system blocks at the FRONT of history
    sys_blocks: list[str] = []
    if tools_norm:
        block = tool_system_block(tools_norm)
        if force_tool == "*":
            block += "\n\nYou MUST call one of the tools above. Do not answer directly."
        elif force_tool:
            block += f"\n\nYou MUST call the `{force_tool}` tool. Do not answer directly."
        sys_blocks.append(block)
    sb = structured_system_block(req.response_format)
    if sb:
        sys_blocks.append(sb)
    for block in reversed(sys_blocks):
        turns = prepend_system(turns, block)

    # Inline text-file content into the prompt so the model sees it reliably
    file_text = await inline_file_text(req.messages)
    if file_text and turns:
        # Attach to latest user turn so it's included even on cached-session single-turn sends
        role, content = turns[-1]
        if role == "user":
            turns = turns[:-1] + [(role, f"{file_text}\n\n{content}")]
        else:
            turns = turns + [("user", file_text)]

    client: DeepSeekClient = request.app.state.ds
    alias = request.headers.get("x-ds-session")
    session_id, parent_id, prompt = await resolve_session(client, turns, alias=alias)
    # Cache-hit => prompt is only the new user turn; re-prepend the tool +
    # structured-output contracts so DeepSeek keeps honoring them across turns.
    if sys_blocks and turns and prompt == turns[-1][1]:
        prompt = "\n\n".join(sys_blocks) + "\n\n" + prompt
    # If we've inlined text already, skip ref_file_ids (upstream sometimes hangs on PENDING files)
    ref_file_ids: list[str] = [] if file_text else await collect_ref_file_ids(req.messages)

    has_tools = bool(tools_norm)
    include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))
    if req.stream:
        return StreamingResponse(
            _stream(
                client, session_id, parent_id, turns, prompt, thinking, search, req.model,
                has_tools=has_tools, response_format=req.response_format,
                ref_file_ids=ref_file_ids, include_usage=include_usage,
                base_model=model, mcp_servers=mcp_servers,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return JSONResponse(
        await _buffered(
            client, session_id, parent_id, turns, prompt, thinking, search, req.model,
            has_tools=has_tools, response_format=req.response_format,
            ref_file_ids=ref_file_ids,
            base_model=model, mcp_servers=mcp_servers,
        )
    )


async def _stream(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | str | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool = False,
    response_format: dict | None = None,
    ref_file_ids: list[str] | None = None,
    include_usage: bool = False,
    base_model: str | None = None,
    mcp_servers: list[str] | None = None,
) -> AsyncIterator[bytes]:
    id_ = f"chatcmpl-{uuid.uuid4().hex}"
    yield _sse(_chunk(id_, model, {"role": "assistant", "content": ""}))

    parser = ToolCallParser() if has_tools else None
    tool_index = 0
    tool_calls_emitted: list[dict] = []
    assistant_buf: list[str] = []
    final_msg_id: int | None = None
    final_session_id: str = session_id

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
            if ev["type"] == "thinking":
                yield _sse(_chunk(id_, model, {"reasoning_content": ev["text"]}))
                continue
            if ev["type"] == "done":
                final_msg_id = ev["message_id"]
                final_session_id = ev.get("session_id") or session_id
                if parser:
                    for pev in parser.flush():
                        if pev["type"] == "text":
                            assistant_buf.append(pev["text"])
                            yield _sse(_chunk(id_, model, {"content": pev["text"]}))
                finish = "tool_calls" if tool_calls_emitted else ev["finish_reason"]
                yield _sse(_chunk(id_, model, {}, finish_reason=finish))
                if include_usage:
                    completion_tokens = len("".join(assistant_buf).split())
                    prompt_tokens = len(prompt.split()) if prompt else 0
                    usage_chunk = {
                        "id": id_,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "system_fingerprint": SYSTEM_FINGERPRINT,
                        "choices": [],
                        "usage": _usage(prompt_tokens, completion_tokens),
                    }
                    yield _sse(usage_chunk)
                break
            if ev["type"] != "content":
                continue
            text = ev["text"]
            if not parser:
                assistant_buf.append(text)
                yield _sse(_chunk(id_, model, {"content": text}))
                continue
            for pev in parser.feed(text):
                if pev["type"] == "text":
                    assistant_buf.append(pev["text"])
                    yield _sse(_chunk(id_, model, {"content": pev["text"]}))
                elif pev["type"] == "tool_call":
                    tc_delta = {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": pev["id"],
                                "type": "function",
                                "function": {
                                    "name": pev["name"],
                                    "arguments": json.dumps(pev["arguments"]),
                                },
                            }
                        ]
                    }
                    tool_calls_emitted.append(pev)
                    tool_index += 1
                    yield _sse(_chunk(id_, model, tc_delta))
    except Exception as e:
        yield _sse({"error": {"message": str(e), "type": "upstream_error"}})
    else:
        cache_text = build_cache_text("".join(assistant_buf), tool_calls_emitted)
        await _cache_turn(turns, cache_text, final_session_id, final_msg_id)
    yield _sse("[DONE]")


async def _buffered(
    client: DeepSeekClient,
    session_id: str,
    parent_id: int | str | None,
    turns: list[tuple[str, str]],
    prompt: str,
    thinking: bool,
    search: bool,
    model: str,
    has_tools: bool = False,
    response_format: dict | None = None,
    ref_file_ids: list[str] | None = None,
    base_model: str | None = None,
    mcp_servers: list[str] | None = None,
) -> dict:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict] = []
    parser_tool_calls: list[dict] = []
    message_id: int | str | None = None
    finish_reason = "stop"
    final_session_id: str = session_id

    parser = ToolCallParser() if has_tools else None

    async def process(ev: dict):
        nonlocal message_id, finish_reason, final_session_id
        if ev["type"] == "thinking":
            thinking_parts.append(ev["text"])
            return
        if ev["type"] == "done":
            message_id = ev["message_id"]
            finish_reason = ev["finish_reason"]
            final_session_id = ev.get("session_id") or session_id
            return
        if ev["type"] != "content":
            return
        text = ev["text"]
        if not parser:
            content_parts.append(text)
            return
        for pev in parser.feed(text):
            if pev["type"] == "text":
                content_parts.append(pev["text"])
            elif pev["type"] == "tool_call":
                parser_tool_calls.append(pev)
                tool_calls.append(
                    {
                        "id": pev["id"],
                        "type": "function",
                        "function": {
                            "name": pev["name"],
                            "arguments": json.dumps(pev["arguments"]),
                        },
                    }
                )

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
        await process(ev)
        if ev["type"] == "done":
            break
    if parser:
        for pev in parser.flush():
            if pev["type"] == "text":
                content_parts.append(pev["text"])

    content = "".join(content_parts)

    # Structured output validation (best-effort, one retry)
    if response_format and not tool_calls:
        ok, err = validate_structured(content, response_format)
        if not ok:
            retry_turns = list(turns) + [
                ("assistant", content),
                (
                    "user",
                    f"Previous reply failed JSON validation: {err}. "
                    f"Reply again with ONLY the JSON object, nothing else.",
                ),
            ]
            retry_prompt = flatten_prefix(retry_turns)
            retry_sid = await client.create_session()
            content_parts = []
            async for ev in client.stream_completion(
                session_id=retry_sid, prompt=retry_prompt, thinking=thinking, search=search
            ):
                if ev["type"] == "content":
                    content_parts.append(ev["text"])
                elif ev["type"] == "done":
                    message_id = ev["message_id"]
                    final_session_id = ev.get("session_id") or retry_sid
                    break
            content = "".join(content_parts)

    cache_text = build_cache_text(content, parser_tool_calls)
    await _cache_turn(turns, cache_text, final_session_id, message_id)

    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if thinking_parts:
        msg["reasoning_content"] = "".join(thinking_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    reasoning_tokens = len("".join(thinking_parts).split()) if thinking_parts else 0
    completion_tokens = len(content.split()) if content else 0
    prompt_tokens = len(prompt.split()) if prompt else 0

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": finish_reason}],
        "usage": _usage(prompt_tokens, completion_tokens, reasoning_tokens),
    }


def build_cache_text(content: str, tool_calls: list[dict]) -> str:
    """Assemble the assistant cache key from text content + serialized tool calls."""
    return "\n".join(
        p for p in [content, serialize_tool_calls(tool_calls)] if p
    ).strip()


async def _cache_turn(
    turns: list[tuple[str, str]],
    assistant_text: str,
    session_id: str,
    message_id: int | str | None,
) -> None:
    """Cache post-completion state so the client's next turn (which will include
    this assistant reply in its messages array) can resume without replaying.

    `assistant_text` must equal what the client will reflatten — text plus the
    serialized `<tool_call>...</tool_call>` envelopes — so the prefix-hash for
    the next request matches. Tool-only responses still cache (the empty-text
    case is normal in agentic loops).
    """
    if message_id is None:
        return
    full = turns + [("assistant", assistant_text)]
    await sessions.put(sessions.hash_turns(full), session_id, message_id)
