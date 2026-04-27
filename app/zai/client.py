"""Async client for chat.z.ai.

Yields canonical events identical to app.deepseek.client.stream_completion so
the routes layer can swap backends via app.state.backend:
    {"type": "thinking", "text": str}
    {"type": "content",  "text": str}
    {"type": "search_status", "status": str}
    {"type": "search_results", "results": list[dict]}
    {"type": "done", "message_id": str, "finish_reason": str}

Upstream SSE shape:
    {"type":"add","data":{
        "delta_content":"...","phase":"thinking|answering|done",
        "done":bool,"usage":{...},"error":...
    }}

Reverse-engineered against prod-fe-1.1.14:
- Endpoint: POST /api/v2/chat/completions?<urlParams>
- urlParams carry timestamp/requestId/user_id/version/platform/token + browser
  fingerprint. Server validates only signature + ts/rid/uid (others are telemetry).
- Headers: authorization, content-type, accept-language, x-fe-version, x-signature.
- Body: {stream, model, messages, signature_prompt, params, extra, features,
         variables, chat_id, id, current_user_message_id,
         current_user_message_parent_id, background_tasks}
- Signature: see app/zai/signature.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from app.config import ZAI_BASE_URL, ZAI_CONTINUATION, ZAI_FE_VERSION
from app.zai import auth
from app.zai.signature import generate as _sign

log = logging.getLogger(__name__)

# Marker for client-side session handles ZaiClient.create_session() returns.
# Real Z.AI chat_ids never start with this string, so the routes can pass us
# back either a placeholder (no continuation possible) or a real chat_id.
_PLACEHOLDER_PREFIX = "zai-placeholder-"


# Upstream model IDs from /api/models on chat.z.ai.
_MODEL_ITEMS: dict[str, dict[str, str]] = {
    "GLM-5.1":      {"name": "GLM-5.1",      "owned_by": "openai"},
    "GLM-5-Turbo":  {"name": "GLM-5-Turbo",  "owned_by": "openai"},
    "GLM-5V-Turbo": {"name": "GLM-5V-Turbo", "owned_by": "openai"},
}


_MODEL_ALIASES: dict[str, str] = {
    "glm-5.1":      "GLM-5.1",
    "glm-5-1":      "GLM-5.1",
    "glm-5":        "GLM-5.1",
    "glm-5-turbo":  "GLM-5-Turbo",
    "glm-5v":       "GLM-5V-Turbo",
    "glm-5v-turbo": "GLM-5V-Turbo",
}


def _resolve_model(name: str | None) -> str:
    if not name:
        return "GLM-5.1"
    n = name.lower().strip()
    if n in _MODEL_ALIASES:
        return _MODEL_ALIASES[n]
    if name in _MODEL_ITEMS:
        return name
    return "GLM-5.1"


_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/147.0.0.0 Safari/537.36"
)


# Aliyun WAF cookies rotate on every response. Sending the values persisted
# in state.json — even if not yet expired by date — gets the request 405'd
# once the server has issued a newer pair. Strip them from any explicit
# `cookies=` param so the AsyncClient jar (populated from `/chats/new` Set-
# Cookie) supplies the live values instead.
_WAF_COOKIES = {"acw_tc", "cdn_sec_tc", "ssxmod_itna", "ssxmod_itna2"}


def _strip_waf(cookies: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in cookies.items() if k not in _WAF_COOKIES}


# Tried in order whenever the server returns 426 (client version rejected).
# `ZAI_FE_VERSION` from config is appended first so an explicit override always
# wins. The active version is cached in a module global so once we discover a
# working one, every subsequent request uses it for the lifetime of the proxy.
_FE_FALLBACKS: tuple[str, ...] = (
    "prod-fe-1.1.14",
    "prod-fe-1.1.0",
    "prod-fe-1.0.94",
)
_active_fe_version: str = ZAI_FE_VERSION


def _fe_candidates() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in (_active_fe_version, ZAI_FE_VERSION, *_FE_FALLBACKS):
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _base_headers(token: str, signature: str, fe_version: str | None = None) -> dict[str, str]:
    # Header set + ordering mirrors a captured browser request (see
    # ~/.zai-proxy/capture.json). Aliyun's WAF in front of chat.z.ai
    # 405s requests that include `origin` or a non-empty `accept` on
    # /api/v2/chat/completions even though the body is otherwise valid.
    return {
        "x-fe-version": fe_version or _active_fe_version,
        "sec-ch-ua-platform": '"macOS"',
        "authorization": f"Bearer {token}",
        "referer": "",
        "accept-language": "en-US",
        "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
        "sec-ch-ua-mobile": "?0",
        "x-signature": signature,
        "user-agent": _UA,
        "content-type": "application/json",
    }


def _build_url_params(
    *, timestamp_ms: int, request_id: str, user_id: str, token: str, chat_id: str
) -> str:
    """Mirror frontend Zs() output. Server only validates ts/rid/uid/signature
    but extra fingerprint fields keep us indistinguishable from a browser.
    """
    now = datetime.now(timezone.utc)
    iso_local = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    utc_str = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    pairs: list[tuple[str, str]] = [
        ("timestamp", str(timestamp_ms)),
        ("requestId", request_id),
        ("user_id", user_id),
        ("version", "0.0.1"),
        ("platform", "web"),
        ("token", token),
        ("user_agent", _UA),
        ("language", "en-US"),
        ("languages", "en-US,en"),
        ("timezone", "UTC"),
        ("cookie_enabled", "true"),
        ("screen_width", "1280"),
        ("screen_height", "720"),
        ("screen_resolution", "1280x720"),
        ("viewport_height", "720"),
        ("viewport_width", "1280"),
        ("viewport_size", "1280x720"),
        ("color_depth", "24"),
        ("pixel_ratio", "1"),
        ("current_url", f"{ZAI_BASE_URL}/c/{chat_id}"),
        ("pathname", f"/c/{chat_id}"),
        ("search", ""),
        ("hash", ""),
        ("host", "chat.z.ai"),
        ("hostname", "chat.z.ai"),
        ("protocol", "https:"),
        ("referrer", ""),
        ("title", "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5"),
        ("timezone_offset", "0"),
        ("local_time", iso_local),
        ("utc_time", utc_str),
        ("is_mobile", "false"),
        ("is_touch", "false"),
        ("max_touch_points", "0"),
        ("browser_name", "Chrome"),
        ("os_name", "Mac OS"),
        ("signature_timestamp", str(timestamp_ms)),
    ]
    # quote_plus (default for urlencode) emits `+` for spaces, matching the
    # frontend's URLSearchParams output. The Aliyun WAF in front of chat.z.ai
    # 405s the percent-encoded `%20` form — keep this default.
    return urllib.parse.urlencode(pairs)


class ZaiClient:
    def __init__(self):
        # Long read timeout: GLM thinking phase can pause for several minutes.
        self._http = httpx.AsyncClient(http2=True, timeout=httpx.Timeout(600.0, connect=15.0))

    async def aclose(self):
        await self._http.aclose()

    async def _state(self, force: bool = False) -> dict[str, Any]:
        return await auth.get_state(force_refresh=force)

    async def create_session(self) -> str:
        # Placeholder handle for the routes layer; real chat_id is allocated
        # server-side in `_new_chat`. The prefix lets stream_completion tell
        # placeholders apart from real chat_ids passed back via the cache.
        return f"{_PLACEHOLDER_PREFIX}{uuid.uuid4()}"

    async def _new_chat(
        self, *, token: str, model_id: str, prompt: str, user_msg_id: str,
        thinking: bool, search: bool, cookies: dict[str, str],
    ) -> str:
        """POST /api/v1/chats/new → returns server-assigned chat id."""
        chat_obj = {
            "id": "",
            "title": "New Chat",
            "models": [model_id],
            "params": {},
            "history": {
                "messages": {
                    user_msg_id: {
                        "id": user_msg_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": prompt,
                        "timestamp": int(time.time()),
                        "models": [model_id],
                    }
                },
                "currentId": user_msg_id,
            },
            "tags": [],
            "flags": [],
            "features": [],
            "mcp_servers": [],
            "enable_thinking": bool(thinking),
            "auto_web_search": bool(search),
            "message_version": 1,
            "extra": {},
            "timestamp": int(time.time() * 1000),
            "type": "chat",
        }
        global _active_fe_version
        tried: set[str] = set()
        last_status: int = 0
        last_body: str = ""
        for _ in range(len(_fe_candidates()) + 1):
            headers = {
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
                "accept": "application/json",
                "accept-language": "en-US",
                "user-agent": _UA,
                "x-fe-version": _active_fe_version,
            }
            r = await self._http.post(
                f"{ZAI_BASE_URL}/api/v1/chats/new",
                headers=headers, cookies=_strip_waf(cookies),
                json={"chat": chat_obj},
            )
            if r.status_code == 426:
                tried.add(_active_fe_version)
                last_status, last_body = 426, r.text[:300]
                remaining = [v for v in _fe_candidates() if v not in tried]
                if not remaining:
                    break
                log.warning(
                    "zai chats/new 426 on x-fe-version=%s; trying %s",
                    _active_fe_version, remaining[0],
                )
                _active_fe_version = remaining[0]
                continue
            if r.status_code != 200:
                last_status, last_body = r.status_code, r.text[:300]
                break
            data = r.json()
            cid = data.get("id") or (data.get("chat") or {}).get("id")
            if not cid:
                raise RuntimeError(
                    f"z.ai chats/new: no id in response: {r.text[:300]}"
                )
            return cid
        raise RuntimeError(
            f"z.ai chats/new HTTP {last_status}: {last_body}"
        )

    async def stream_completion(
        self,
        *,
        session_id: str,
        prompt: str,
        parent_message_id: str | None = None,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: list[str] | None = None,
        model: str | None = None,
        mcp_servers: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        state = await self._state()
        token = state["token"]
        user_id = state.get("user_id") or ""
        cookies = auth.cookies_dict(state)

        model_id = _resolve_model(model)
        message_id = str(uuid.uuid4())
        user_msg_id = str(uuid.uuid4())

        # Continuation: capture confirmed /api/v2/chat/completions accepts a
        # reused chat_id with parent_message_id chained from the prior turn's
        # assistant `id`. No /api/v1/chats append round-trip is needed — the
        # server keeps history server-side keyed by chat_id.
        is_real_chat = (
            isinstance(session_id, str)
            and not session_id.startswith(_PLACEHOLDER_PREFIX)
        )
        continuation = (
            ZAI_CONTINUATION
            and is_real_chat
            and isinstance(parent_message_id, str)
            and parent_message_id
        )

        log.info(
            "zai stream begin: session=%s model=%s prompt_len=%d thinking=%s search=%s continuation=%s",
            session_id, model_id, len(prompt or ""), thinking, search, continuation,
        )

        if continuation:
            chat_id = session_id
        else:
            new_chat_t0 = time.monotonic()
            chat_id = await self._new_chat(
                token=token, model_id=model_id, prompt=prompt,
                user_msg_id=user_msg_id, thinking=thinking, search=search,
                cookies=cookies,
            )
            log.info("zai chats/new in %.2fs (chat_id=%s)", time.monotonic() - new_chat_t0, chat_id)
            parent_message_id = None  # fresh chat: no parent

        signature_prompt = prompt.strip()
        request_id = str(uuid.uuid4())
        signature, ts = _sign(
            message=signature_prompt, request_id=request_id, user_id=user_id
        )

        # Body shape mirrors the captured browser POST. The Aliyun WAF on
        # chat.z.ai checks for the full feature/vlm flag set; missing keys
        # surface as a 405 even though the smaller body is technically valid.
        features: dict[str, Any] = {
            "image_generation": False,
            "web_search": bool(search),
            "auto_web_search": bool(search),
            "preview_mode": True,
            "flags": [],
            "vlm_tools_enable": False,
            "vlm_web_search_enable": False,
            "vlm_website_mode": False,
            "enable_thinking": bool(thinking),
        }

        body = {
            "stream": True,
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "signature_prompt": signature_prompt,
            "params": {},
            "extra": {},
            "features": features,
            "variables": {
                "{{USER_NAME}}": "user",
                "{{USER_LOCATION}}": "Unknown",
                "{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "{{CURRENT_DATE}}": datetime.now().strftime("%Y-%m-%d"),
                "{{CURRENT_TIME}}": datetime.now().strftime("%H:%M:%S"),
                "{{CURRENT_WEEKDAY}}": datetime.now().strftime("%A"),
                "{{CURRENT_TIMEZONE}}": "UTC",
                "{{USER_LANGUAGE}}": "en-US",
            },
            "chat_id": chat_id,
            "id": message_id,
            "current_user_message_id": user_msg_id,
            "current_user_message_parent_id": parent_message_id,
            "background_tasks": {"title_generation": True, "tags_generation": True},
        }
        if mcp_servers:
            body["mcp_servers"] = list(mcp_servers)

        url_params = _build_url_params(
            timestamp_ms=ts, request_id=request_id,
            user_id=user_id, token=token, chat_id=chat_id,
        )
        url = f"{ZAI_BASE_URL}/api/v2/chat/completions?{url_params}"
        headers = _base_headers(token, signature)

        for attempt in range(3):
            try:
                async with self._http.stream(
                    "POST", url, headers=headers,
                    cookies=_strip_waf(cookies),
                    json=body,
                ) as r:
                    if r.status_code == 401:
                        await r.aread()
                        state = await self._state(force=True)
                        token = state["token"]
                        user_id = state.get("user_id") or ""
                        cookies = auth.cookies_dict(state)
                        request_id = str(uuid.uuid4())
                        signature, ts = _sign(
                            message=signature_prompt, request_id=request_id,
                            user_id=user_id,
                        )
                        url_params = _build_url_params(
                            timestamp_ms=ts, request_id=request_id,
                            user_id=user_id, token=token, chat_id=chat_id,
                        )
                        url = f"{ZAI_BASE_URL}/api/v2/chat/completions?{url_params}"
                        headers = _base_headers(token, signature)
                        continue
                    if r.status_code == 426:
                        await r.aread()
                        # Server rejected our client version. Walk the fallback
                        # list looking for one the server accepts; promote the
                        # winner so subsequent requests skip the discovery cost.
                        global _active_fe_version
                        candidates = [v for v in _fe_candidates() if v != _active_fe_version]
                        if not candidates:
                            raise RuntimeError(
                                "z.ai rejected client version — set ZAI_FE_VERSION "
                                "to a current `prod-fe-*` release "
                                f"(tried {_active_fe_version})"
                            )
                        log.warning(
                            "zai 426 on x-fe-version=%s; trying %s",
                            _active_fe_version, candidates[0],
                        )
                        _active_fe_version = candidates[0]
                        headers = _base_headers(token, signature)
                        continue
                    if r.status_code == 405:
                        # Aliyun WAF returns 405 when the cookies it issued
                        # (acw_tc / cdn_sec_tc / ssxmod_*) are stale. The
                        # client jar normally rotates them via Set-Cookie on
                        # each response, but a long-lived process can drift.
                        # Drop the WAF-issued cookies and try a silent refresh
                        # via the persisted Playwright profile before retrying.
                        await r.aread()
                        for name in ("acw_tc", "cdn_sec_tc",
                                      "ssxmod_itna", "ssxmod_itna2"):
                            self._http.cookies.delete(name)
                        cookies = {k: v for k, v in cookies.items()
                                    if k not in {"acw_tc", "cdn_sec_tc",
                                                  "ssxmod_itna", "ssxmod_itna2"}}
                        if attempt == 0:
                            refreshed = await auth.refresh_cookies_silently()
                            if refreshed:
                                token = refreshed["token"]
                                user_id = refreshed.get("user_id") or ""
                                cookies = auth.cookies_dict(refreshed)
                                request_id = str(uuid.uuid4())
                                signature, ts = _sign(
                                    message=signature_prompt,
                                    request_id=request_id, user_id=user_id,
                                )
                                url_params = _build_url_params(
                                    timestamp_ms=ts, request_id=request_id,
                                    user_id=user_id, token=token, chat_id=chat_id,
                                )
                                url = f"{ZAI_BASE_URL}/api/v2/chat/completions?{url_params}"
                                headers = _base_headers(token, signature)
                            continue
                        if attempt < 2:
                            await asyncio.sleep(2 * attempt)
                            continue
                        raise RuntimeError(
                            "z.ai HTTP 405 on /api/v2/chat/completions. "
                            "Most common cause: state.json is a GUEST account — "
                            "Z.AI's WAF allows guest tokens on /chats/new but "
                            "blocks them on /chat/completions. Decode "
                            "~/.zai-proxy/state.json `token` JWT; if email is "
                            "guest-*@guest.com, re-run "
                            "`.venv/bin/python -m probe.zai_login` and sign in "
                            "with a real account. If already a real account, "
                            "WAF cookies may need a fresh capture."
                        )
                    if r.status_code == 429:
                        await r.aread()
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if r.status_code != 200:
                        body_bytes = await r.aread()
                        raise RuntimeError(
                            f"z.ai completion HTTP {r.status_code}: "
                            f"{body_bytes.decode(errors='replace')[:400]}"
                        )
                    first_byte_at: float | None = None
                    post_t0 = time.monotonic()
                    async for ev in _parse_stream(r, message_id):
                        if first_byte_at is None:
                            first_byte_at = time.monotonic()
                            log.info(
                                "zai first event after %.2fs",
                                first_byte_at - post_t0,
                            )
                        # Tag done events with the real chat_id so the routes
                        # layer can cache it as the canonical session handle.
                        if ev.get("type") == "done" and "session_id" not in ev:
                            ev = {**ev, "session_id": chat_id}
                        yield ev
                    log.info(
                        "zai stream complete in %.2fs",
                        time.monotonic() - post_t0,
                    )
                    waf_cookies = {
                        c.name: c.value for c in self._http.cookies.jar
                        if c.name in {"acw_tc", "cdn_sec_tc",
                                      "ssxmod_itna", "ssxmod_itna2"}
                    }
                    if waf_cookies:
                        try:
                            auth.merge_jar_into_state(waf_cookies)
                        except Exception as e:  # noqa: BLE001
                            log.debug("zai cookie persist skipped: %s", e)
                    return
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("z.ai completion failed after retries")


async def _parse_stream(
    r: httpx.Response, message_id: str
) -> AsyncIterator[dict[str, Any]]:
    async for line in r.aiter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            if payload == "[DONE]":
                yield {"type": "done", "message_id": message_id, "finish_reason": "stop"}
                return
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        data = chunk.get("data") if isinstance(chunk, dict) else None
        if not isinstance(data, dict):
            continue

        err = chunk.get("error") or data.get("error")
        if err:
            raise RuntimeError(f"z.ai upstream error: {err}")

        phase = data.get("phase")
        delta = data.get("delta_content") or data.get("edit_content")
        done = bool(data.get("done")) or phase == "done"

        if phase in ("search", "tool_call") and delta:
            yield {"type": "search_status", "status": str(delta)}
            continue
        results = data.get("web_search") or data.get("search_results")
        if isinstance(results, list) and results:
            yield {"type": "search_results", "results": results}

        if delta and isinstance(delta, str):
            if phase == "thinking":
                yield {"type": "thinking", "text": delta}
            else:
                yield {"type": "content", "text": delta}

        if done:
            yield {"type": "done", "message_id": message_id, "finish_reason": "stop"}
            return
