"""Z.AI X-Signature generator for /api/v2/chat/completions.

Reverse-engineered from prod-fe-1.1.14 chunk m87BJU5M.js (function `ei`):

  derived = HMAC-SHA256(key=KEY, msg=str(window_index)).hex()
  signature = HMAC-SHA256(key=derived_hex_utf8, msg=d).hex()

where:
  window_index = floor(ts_ms / 300_000)
  d = sortedPayload + "|" + base64(utf8(prompt)) + "|" + str(ts_ms)
  sortedPayload = ",".join("k,v" for k,v in sorted({timestamp,requestId,user_id}.items()))
                = "requestId,<rid>,timestamp,<ts>,user_id,<uid>"

KEY = "key-@@@@)))()((9))-xxxx&&&%%%%%". Override via env ZAI_SIG_KEY.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time

from app.config import ZAI_SIG_KEY


def _hmac_hex(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def _sorted_payload(timestamp_ms: int, request_id: str, user_id: str) -> str:
    pairs = sorted(
        [("timestamp", str(timestamp_ms)),
         ("requestId", request_id),
         ("user_id", user_id)],
        key=lambda kv: kv[0],
    )
    return ",".join(f"{k},{v}" for k, v in pairs)


def generate(
    *,
    message: str,
    request_id: str,
    user_id: str,
    timestamp_ms: int | None = None,
    secret: str | None = None,
) -> tuple[str, int]:
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    key = (secret or ZAI_SIG_KEY).encode()
    window = ts // (5 * 60 * 1000)
    derived_hex = _hmac_hex(key, str(window).encode())
    payload = _sorted_payload(ts, request_id, user_id)
    b64 = base64.b64encode(message.encode("utf-8")).decode("ascii")
    canonical = f"{payload}|{b64}|{ts}"
    sig = _hmac_hex(derived_hex.encode("utf-8"), canonical.encode("utf-8"))
    return sig, ts
