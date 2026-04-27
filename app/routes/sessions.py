"""/v1/sessions — pin, list, and evict cached conversation sessions.

Exposes the existing prefix-hash cache (app/deepseek/sessions.py) as a small
REST surface so clients can name a session, resume it by name, and invalidate
stale ones without editing sessions.json by hand.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from app.deepseek import sessions as ds_sessions
from app.routes.auth import require_any_key

router = APIRouter()

_require_key = require_any_key


class PinRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    prefix_hash: str | None = None


@router.get("/v1/sessions")
async def list_sessions(_: None = Depends(_require_key)):
    return await ds_sessions.list_all()


@router.post("/v1/sessions")
async def pin_session(body: PinRequest, _: None = Depends(_require_key)):
    if not body.name or body.name.startswith("_"):
        raise HTTPException(400, "invalid session name")
    try:
        prefix_hash = await ds_sessions.put_alias(body.name, body.prefix_hash)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"name": body.name, "prefix_hash": prefix_hash}


@router.delete("/v1/sessions/{name}")
async def delete_session(name: str, _: None = Depends(_require_key)):
    ok = await ds_sessions.delete_alias(name, drop_entry=True)
    if not ok:
        raise HTTPException(404, f"unknown session: {name}")
    return {"deleted": name}


@router.delete("/v1/sessions")
async def flush_sessions(_: None = Depends(_require_key)):
    count = await ds_sessions.clear_all()
    return {"cleared": count}
