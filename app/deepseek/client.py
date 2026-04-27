"""Async client for chat.deepseek.com/api/v0.

Yields canonical events from stream_completion:
    {"type": "thinking", "text": str}
    {"type": "content",  "text": str}
    {"type": "search_status", "status": str}
    {"type": "search_results", "results": list[dict]}
    {"type": "done", "message_id": int, "finish_reason": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from app.config import APP_VERSION, BASE_URL
from app.deepseek import auth
from app.deepseek.pow import solve_challenge

log = logging.getLogger(__name__)


_STATIC_HEADERS: dict[str, str] = {
    "content-type": "application/json",
    "x-app-version": APP_VERSION,
    "x-client-platform": "web",
    "x-client-version": "1.0.0-always",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "accept": "*/*",
    "origin": "https://chat.deepseek.com",
    "referer": "https://chat.deepseek.com/",
}


def _headers(token: str, pow_resp: str | None = None) -> dict[str, str]:
    h = {**_STATIC_HEADERS, "authorization": f"Bearer {token}"}
    if pow_resp:
        h["x-ds-pow-response"] = pow_resp
    return h


class DeepSeekClient:
    def __init__(self):
        # Long read timeout: reasoner streams can pause >2min between deltas
        # while thinking. The original 120s killed those mid-stream.
        self._http = httpx.AsyncClient(http2=True, timeout=httpx.Timeout(600.0, connect=15.0))

    async def aclose(self):
        await self._http.aclose()

    async def _state(self, force: bool = False) -> dict[str, Any]:
        return await auth.get_state(force_refresh=force)

    async def _post(self, path: str, json_body: dict, *, pow_resp: str | None = None) -> httpx.Response:
        state = await self._state()
        for attempt in range(3):
            r = await self._http.post(
                f"{BASE_URL}{path}",
                headers=_headers(state["userToken"], pow_resp),
                cookies=auth.cookies_dict(state),
                json=json_body,
            )
            if r.status_code == 401:
                state = await self._state(force=True)
                continue
            if r.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            # Upstream sometimes returns 200 with biz_code indicating rate limit
            try:
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
            except Exception:
                body = None
            if body and (body.get("data") or {}).get("biz_code") == 7:
                await asyncio.sleep(2 ** attempt + 1)
                continue
            return r
        return r

    async def create_session(self) -> str:
        r = await self._post("/chat_session/create", {"character_id": None})
        r.raise_for_status()
        body = r.json()
        biz = (body.get("data") or {}).get("biz_data")
        if not biz:
            raise RuntimeError(f"create_session failed: {body}")
        return biz["id"]

    async def _pow(self, target: str) -> str:
        t0 = time.monotonic()
        r = await self._post("/chat/create_pow_challenge", {"target_path": target})
        r.raise_for_status()
        body = r.json()
        biz = (body.get("data") or {}).get("biz_data")
        if not biz:
            raise RuntimeError(f"create_pow failed: {body}")
        challenge = biz["challenge"]
        challenge["target_path"] = target
        resp = solve_challenge(challenge)
        log.info("deepseek pow %s solved in %.2fs", target, time.monotonic() - t0)
        return resp

    async def stream_completion(
        self,
        *,
        session_id: str,
        prompt: str,
        parent_message_id: int | None = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: list[str] | None = None,
        model: str | None = None,  # unused; kept for backend parity
        mcp_servers: list[str] | None = None,  # unused; Z.AI-only
    ) -> AsyncIterator[dict[str, Any]]:
        target = "/api/v0/chat/completion"
        log.info(
            "deepseek stream begin: session=%s prompt_len=%d thinking=%s search=%s ref_files=%d",
            session_id, len(prompt or ""), thinking, search, len(ref_file_ids or []),
        )
        pow_resp = await self._pow(target)
        state = await self._state()
        body = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking,
            "search_enabled": search,
        }
        for attempt in range(4):
            yielded_any = False
            retryable_error: str | None = None
            try:
                async with self._http.stream(
                    "POST",
                    f"{BASE_URL}{target.replace('/api/v0', '')}",
                    headers=_headers(state["userToken"], pow_resp),
                    cookies=auth.cookies_dict(state),
                    json=body,
                ) as r:
                    if r.status_code == 429:
                        await r.aread()
                        retryable_error = "HTTP 429"
                    elif r.status_code != 200:
                        body_bytes = await r.aread()
                        raise RuntimeError(
                            f"completion HTTP {r.status_code}: {body_bytes.decode(errors='replace')[:300]}"
                        )
                    else:
                        first_byte_at: float | None = None
                        post_t0 = time.monotonic()
                        async for ev in _parse_stream(r):
                            if first_byte_at is None:
                                first_byte_at = time.monotonic()
                                log.info(
                                    "deepseek first event after %.2fs",
                                    first_byte_at - post_t0,
                                )
                            yielded_any = True
                            # Echo the upstream session_id on done so the
                            # routes layer can treat both backends uniformly.
                            if ev.get("type") == "done" and "session_id" not in ev:
                                ev = {**ev, "session_id": session_id}
                            yield ev
                        log.info(
                            "deepseek stream complete in %.2fs",
                            time.monotonic() - post_t0,
                        )
                        return
            except RuntimeError as e:
                msg = str(e)
                if yielded_any or "biz_code=7" not in msg:
                    raise
                retryable_error = msg
            # Retry: re-solve PoW + backoff
            await asyncio.sleep(2 ** attempt + 1)
            pow_resp = await self._pow(target)
        raise RuntimeError(f"completion failed after retries: {retryable_error}")


async def _parse_stream(r: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    event: str | None = None
    current_path: str | None = None
    response_msg_id: int | None = None

    async for line in r.aiter_lines():
        if not line:
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            chunk = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue

        # Biz-level error frame (rate limit, quota, etc.) — surface as runtime error
        if isinstance(chunk, dict) and isinstance(chunk.get("data"), dict):
            biz_code = chunk["data"].get("biz_code")
            if biz_code not in (None, 0):
                raise RuntimeError(f"upstream biz_code={biz_code}: {chunk['data'].get('biz_msg')}")

        if event == "ready":
            response_msg_id = chunk.get("response_message_id")
            event = None
            continue
        if event == "finish":
            yield {"type": "done", "message_id": response_msg_id, "finish_reason": "stop"}
            return
        if event in ("update_session", "title", "close"):
            event = None
            continue
        event = None

        p = chunk.get("p")
        v = chunk.get("v")
        if p:
            current_path = p

        if current_path == "response/content" and isinstance(v, str):
            yield {"type": "content", "text": v}
        elif current_path == "response/thinking_content" and isinstance(v, str):
            yield {"type": "thinking", "text": v}
        elif current_path == "response/search_status":
            yield {"type": "search_status", "status": v}
        elif current_path == "response/search_results":
            yield {"type": "search_results", "results": v or []}
