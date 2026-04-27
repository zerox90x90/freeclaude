"""Shared API-key auth dependency for all route modules.

Two flavors:
- `require_bearer_key`: checks Authorization: Bearer <key> (OpenAI convention)
- `require_any_key`:    checks x-api-key OR Authorization: Bearer (Anthropic convention)

Both are no-ops when PROXY_API_KEY is unset (open proxy).
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from app.config import PROXY_API_KEY


def require_bearer_key(request: Request) -> None:
    """OpenAI-style: Authorization: Bearer <key>."""
    if not PROXY_API_KEY:
        return
    got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if got != PROXY_API_KEY:
        raise HTTPException(401, "invalid api key")


def require_any_key(request: Request) -> None:
    """Anthropic-style: x-api-key header OR Authorization: Bearer <key>."""
    if not PROXY_API_KEY:
        return
    got = request.headers.get("x-api-key") or request.headers.get(
        "authorization", ""
    ).removeprefix("Bearer ").strip()
    if got != PROXY_API_KEY:
        raise HTTPException(401, "invalid api key")
