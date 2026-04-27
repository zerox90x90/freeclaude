"""Conversation-prefix compression.

When the flattened history exceeds a token threshold, summarize the oldest
middle window via a one-off upstream call and replace it with a synthetic
`user` turn so the prefix stays small. Summaries are cached by the hash of
the window they replace so the same long conversation doesn't pay the
summarization cost twice.

Backend-agnostic: the `client` parameter is duck-typed against the surface
shared by `DeepSeekClient` and `ZaiClient` — `create_session()` and
`stream_completion(session_id, prompt, parent_message_id, thinking, search)`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Protocol

from app.config import STATE_DIR
from app.deepseek.sessions import hash_turns


class _Streamer(Protocol):
    async def create_session(self) -> str: ...
    def stream_completion(
        self,
        *,
        session_id: str,
        prompt: str,
        parent_message_id: int | str | None = ...,
        thinking: bool = ...,
        search: bool = ...,
    ) -> AsyncIterator[dict[str, Any]]: ...


log = logging.getLogger(__name__)

SUMMARIES_FILE = STATE_DIR / "summaries.json"
THRESHOLD_TOKENS = int(os.environ.get("DS_COMPRESS_THRESHOLD_TOKENS", "24000"))
KEEP_TAIL_TURNS = int(os.environ.get("DS_COMPRESS_KEEP_TAIL", "6"))
SUMMARY_PREFIX = "[Summary of earlier turns]\n"

_lock = asyncio.Lock()

_SUMMARY_INSTRUCTION = (
    "Summarize the following conversation so a later assistant can continue it "
    "without loss of intent. Preserve: key facts, user decisions, file paths "
    "mentioned, code identifiers, open questions, and the current task. Omit: "
    "pleasantries, repeated context, rejected approaches. Reply with only the "
    "summary — no preamble, no sign-off."
)


def approx_tokens(turns: list[tuple[str, str]]) -> int:
    total = 0
    for _, content in turns:
        total += max(1, len(content) // 4)
    return total


def _load() -> dict[str, str]:
    if not SUMMARIES_FILE.exists():
        return {}
    try:
        data = json.loads(SUMMARIES_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _save(data: dict[str, str]) -> None:
    tmp = SUMMARIES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(SUMMARIES_FILE)


async def _get_cached(key: str) -> str | None:
    async with _lock:
        return _load().get(key)


async def _put_cached(key: str, summary: str) -> None:
    async with _lock:
        data = _load()
        data[key] = summary
        if len(data) > 200:
            for k in list(data.keys())[:50]:
                del data[k]
        _save(data)


_ROLE_TAGS = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}


def _format_window(window: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for r, c in window:
        t = _ROLE_TAGS.get(r, r.upper())
        parts.append(f"[{t}]\n{c}\n[/{t}]")
    return "\n\n".join(parts)


async def _summarize(client: _Streamer, window: list[tuple[str, str]]) -> str:
    session_id = await client.create_session()
    prompt = f"{_SUMMARY_INSTRUCTION}\n\n{_format_window(window)}"
    chunks: list[str] = []
    async for ev in client.stream_completion(
        session_id=session_id,
        prompt=prompt,
        parent_message_id=None,
        thinking=False,
        search=False,
    ):
        if ev["type"] == "content":
            chunks.append(ev["text"])
        elif ev["type"] == "done":
            break
    return "".join(chunks).strip()


async def maybe_compress(
    client: _Streamer,
    turns: list[tuple[str, str]],
    threshold: int | None = None,
) -> list[tuple[str, str]]:
    """If history exceeds token threshold, replace the oldest middle window
    with a synthetic `[Summary of earlier turns]` user turn. Returns the
    (possibly unchanged) turn list. Never raises — on any failure the caller
    gets the original turns back."""
    try:
        limit = threshold if threshold is not None else THRESHOLD_TOKENS
        if limit <= 0 or approx_tokens(turns) <= limit:
            return turns
        if len(turns) <= KEEP_TAIL_TURNS + 1:
            return turns

        head: list[tuple[str, str]] = []
        body_start = 0
        if turns and turns[0][0] == "system":
            head = [turns[0]]
            body_start = 1

        tail_start = max(body_start, len(turns) - KEEP_TAIL_TURNS)
        window = turns[body_start:tail_start]
        if not window:
            return turns

        key = hash_turns(window)
        summary = await _get_cached(key)
        if summary is None:
            log.info(
                "compress: triggering summarization (turns=%d, ~%d tokens, window=%d turns)",
                len(turns), approx_tokens(turns), len(window),
            )
            t0 = time.monotonic()
            summary = await _summarize(client, window)
            log.info("compress: summarization took %.2fs", time.monotonic() - t0)
            if not summary:
                log.warning("compress: empty summary, skipping")
                return turns
            await _put_cached(key, summary)

        synthetic: tuple[str, str] = ("user", SUMMARY_PREFIX + summary)
        compressed = head + [synthetic] + turns[tail_start:]
        log.info(
            "compress: %d->%d turns, ~%d->~%d tokens",
            len(turns),
            len(compressed),
            approx_tokens(turns),
            approx_tokens(compressed),
        )
        return compressed
    except Exception as e:
        log.warning("compress: fallthrough (%s)", e)
        return turns
