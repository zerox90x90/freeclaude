"""Load persisted Z.AI web session. Re-login via Playwright on demand.

State layout (~/.zai-proxy/state.json):
  {
    "token":   "<JWT bearer>",
    "user_id": "<parsed from JWT>",
    "cookies": [{"name":..., "value":..., ...}, ...]
  }
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

from playwright.async_api import async_playwright

from app.config import ZAI_PROFILE_DIR, ZAI_STATE_FILE

_lock = asyncio.Lock()


def _read_state() -> dict[str, Any] | None:
    if not ZAI_STATE_FILE.exists():
        return None
    try:
        return json.loads(ZAI_STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None


def _write_state(state: dict[str, Any]) -> None:
    ZAI_STATE_FILE.write_text(json.dumps(state, indent=2))


def _jwt_payload(token: str) -> dict | None:
    """Decode JWT payload without verification. Returns None on malformed tokens."""
    try:
        _, payload_b64, _ = token.split(".")
        pad = "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
    except (ValueError, json.JSONDecodeError):
        return None


def _jwt_user_id(token: str) -> str:
    payload = _jwt_payload(token)
    if not payload:
        return ""
    for key in ("id", "user_id", "uid", "sub"):
        val = payload.get(key)
        if val:
            return str(val)
    return ""


def _is_guest_token(token: str) -> bool:
    payload = _jwt_payload(token)
    if not payload:
        return False
    email = str(payload.get("email", ""))
    return email.startswith("guest-") and email.endswith("@guest.com")


async def _capture_token(page, accept_guest: bool = True) -> str | None:
    """Poll common storage locations for the auth token. When
    `accept_guest=False`, ignore tokens whose JWT email is `guest-*` so
    the polling loop keeps waiting until the user signs in for real."""
    for expr in (
        "() => window.localStorage.getItem('token')",
        "() => window.localStorage.getItem('auth_token')",
        "() => document.cookie.split('; ').find(c => c.startsWith('token='))?.split('=')[1]",
    ):
        try:
            val = await page.evaluate(expr)
        except Exception:
            continue
        if isinstance(val, str) and val and val != "null" and len(val) > 20:
            tok = val.strip('"')
            if not accept_guest and _is_guest_token(tok):
                continue
            return tok
    return None


async def _launch_login() -> dict[str, Any]:
    """Launch a fresh Chromium and let the user sign in. We use the
    non-persistent `launch()` + `new_context()` form rather than
    `launch_persistent_context` because the latter regularly crashes on
    macOS when a stale `--user-data-dir` is reused (crashpad SIGTRAP /
    "Target page closed"). Auth survives via state.json regardless."""
    ZAI_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            viewport={"width": 1200, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/147.0.0.0 Safari/537.36"
            ),
        )
        page = await ctx.new_page()
        await page.goto("https://chat.z.ai/")
        print("[zai-auth] sign in with a real account in the browser "
              "(guest tokens are ignored — keep going until you reach the "
              "logged-in chat UI). Polling up to 10 min...")

        token: str | None = None
        for _ in range(600):
            token = await _capture_token(page, accept_guest=False)
            if token:
                break
            await asyncio.sleep(1)
        if not token:
            await ctx.close()
            await browser.close()
            raise TimeoutError(
                "z.ai real-account token not captured in 10 min "
                "(only guest tokens seen — sign in via the browser)"
            )
        cookies = await ctx.cookies()
        await ctx.close()
        await browser.close()

    uid = _jwt_user_id(token)
    state = {"token": token, "user_id": uid, "cookies": cookies}
    _write_state(state)
    print(f"[zai-auth] saved state token={token[:16]}... user_id={uid or '?'} cookies={len(cookies)}")
    if "guest" in token.lower() or _jwt_email(token).startswith("guest-"):
        print("[zai-auth] WARNING: guest account detected — Z.AI's WAF will 405 "
              "on /chat/completions. Sign in with a real account.")
    return state


def _jwt_email(token: str) -> str:
    payload = _jwt_payload(token)
    return str(payload.get("email", "")) if payload else ""


_warned_guest = False


def _warn_if_guest(token: str) -> None:
    global _warned_guest
    if _warned_guest or not token:
        return
    email = _jwt_email(token)
    if email.startswith("guest-") and email.endswith("@guest.com"):
        import logging
        logging.getLogger(__name__).warning(
            "zai auth: state.json is a GUEST account (%s). Z.AI's WAF blocks "
            "guest tokens on /api/v2/chat/completions (HTTP 405). Re-run "
            "`.venv/bin/python -m probe.zai_login` and sign in with a real "
            "account to unblock chat completions.", email,
        )
        _warned_guest = True


async def get_state(force_refresh: bool = False) -> dict[str, Any]:
    async with _lock:
        if not force_refresh:
            state = _read_state()
            if state and state.get("token"):
                _warn_if_guest(state["token"])
                return state
        state = await _launch_login()
        _warn_if_guest(state.get("token", ""))
        return state


def cookies_dict(state: dict[str, Any]) -> dict[str, str]:
    """Return cookies as name→value, dropping any with a past expiry.

    Aliyun WAF cookies (`acw_tc`, `cdn_sec_tc`, `ssxmod_*`) rotate every
    few hours; sending an expired one to /api/v2/chat/completions yields
    HTTP 405. Filter them out so the request relies on the AsyncClient's
    cookie jar, which absorbs fresh values from each response.
    """
    now = time.time()
    out: dict[str, str] = {}
    for c in state.get("cookies", []):
        exp = c.get("expires")
        if isinstance(exp, (int, float)) and 0 < exp < now:
            continue
        out[c["name"]] = c["value"]
    return out


def merge_jar_into_state(jar_cookies: dict[str, str]) -> None:
    """Persist freshly issued cookies (e.g. acw_tc rotated by Aliyun on the
    last response) back into state.json so process restarts don't have to
    rediscover them."""
    if not jar_cookies:
        return
    state = _read_state() or {}
    cookies = list(state.get("cookies", []))
    by_name = {c.get("name"): i for i, c in enumerate(cookies)}
    now = time.time()
    for name, value in jar_cookies.items():
        entry = {"name": name, "value": value,
                 "domain": ".chat.z.ai", "path": "/",
                 "expires": now + 3600}
        if name in by_name:
            cookies[by_name[name]].update(entry)
        else:
            cookies.append(entry)
    state["cookies"] = cookies
    _write_state(state)


async def refresh_cookies_silently() -> dict[str, Any] | None:
    """Stub — kept for client.py compatibility. The previous implementation
    used `launch_persistent_context` which crashes on macOS with stale
    profiles. We rely on the AsyncClient cookie jar absorbing fresh
    Set-Cookie values from /api/v1/chats/new instead."""
    return None
