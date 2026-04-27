"""Backend selector: construct the upstream client based on config.BACKEND.

Both DeepSeekClient and ZaiClient expose the same async surface used by the
routes layer: create_session(), stream_completion(...), aclose(). Routes access
the active instance via request.app.state.ds regardless of which backend is
live, so existing code paths keep working.
"""
from __future__ import annotations

from app.config import BACKEND


def build_client():
    if BACKEND == "zai":
        from app.zai.client import ZaiClient
        return ZaiClient()
    from app.deepseek.client import DeepSeekClient
    return DeepSeekClient()


def backend_label() -> str:
    return "z.ai (GLM)" if BACKEND == "zai" else "deepseek"


def default_model() -> str:
    if BACKEND == "zai":
        return "glm-5.1:search"
    return "deepseek-reasoner:search"


def default_fast_model() -> str:
    if BACKEND == "zai":
        return "glm-5-turbo"
    return "deepseek-chat"
