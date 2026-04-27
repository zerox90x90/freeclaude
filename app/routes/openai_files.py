"""/v1/files — OpenAI-compatible file upload, backed by chat.deepseek.com."""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.config import BACKEND
from app.deepseek import files as ds_files
from app.routes.auth import require_bearer_key

router = APIRouter()

_require_key = require_bearer_key


def _file_obj(openai_id: str, filename: str, size: int, purpose: str, ds_id: str) -> dict:
    return {
        "id": openai_id,
        "object": "file",
        "bytes": size,
        "created_at": int(time.time()),
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
        "deepseek_file_id": ds_id,
    }


@router.post("/v1/files")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
    _: None = Depends(_require_key),
):
    content = await file.read()
    mime = file.content_type or "application/octet-stream"
    filename = file.filename or "upload"

    if BACKEND == "zai":
        # No Z.AI upload endpoint wired yet. For text payloads we cache the
        # decoded bytes locally so the chat routes' inline-text fallback can
        # splice the file into the prompt. Binary uploads still fail loudly so
        # callers see why nothing reached the model.
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=415,
                detail=(
                    "binary file upload is not supported on the Z.AI backend. "
                    "Send a UTF-8 text file (it will be inlined into the "
                    "prompt) or switch to BACKEND=deepseek for upstream upload."
                ),
            ) from e
        openai_id = f"file-{uuid.uuid4().hex[:24]}"
        synthetic_ds_id = f"zai-local-{openai_id}"
        obj = _file_obj(openai_id, filename, len(content), purpose, synthetic_ds_id)
        await ds_files.store_mapping(openai_id, obj, content=content)
        return obj

    http = request.app.state.ds._http  # reuse client's http connection pool
    info = await ds_files.upload(http, filename, content, mime)

    openai_id = f"file-{uuid.uuid4().hex[:24]}"
    obj = _file_obj(openai_id, info.get("file_name", filename), info.get("file_size", len(content)), purpose, info["id"])
    await ds_files.store_mapping(openai_id, obj, content=content)
    return obj


@router.get("/v1/files")
async def list_files(request: Request, _: None = Depends(_require_key)):
    return {"object": "list", "data": await ds_files.list_mappings()}


@router.get("/v1/files/{file_id}")
async def get_file(file_id: str, request: Request, _: None = Depends(_require_key)):
    info = await ds_files.get_mapping(file_id)
    if not info:
        raise HTTPException(404, "file not found")
    return info


@router.delete("/v1/files/{file_id}")
async def delete_file(file_id: str, request: Request, _: None = Depends(_require_key)):
    ok = await ds_files.delete_mapping(file_id)
    return {"id": file_id, "object": "file", "deleted": ok}
