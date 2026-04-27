import os
from pathlib import Path

BACKEND = os.environ.get("BACKEND", "deepseek").lower()
if BACKEND in ("glm", "zai", "z.ai", "z"):
    BACKEND = "zai"
elif BACKEND in ("deepseek", "ds"):
    BACKEND = "deepseek"

# DeepSeek paths (legacy names kept so existing modules keep working)
STATE_DIR = Path(os.path.expanduser("~/.deepseek-proxy"))
STATE_FILE = STATE_DIR / "state.json"
PROFILE_DIR = STATE_DIR / "chromium-profile"
WASM_PATH = Path(__file__).parent / "wasm" / "sha3_wasm_bg.wasm"

BASE_URL = "https://chat.deepseek.com/api/v0"
APP_VERSION = os.environ.get("DS_APP_VERSION", "20241129.1")

# Z.AI paths
ZAI_STATE_DIR = Path(os.path.expanduser("~/.zai-proxy"))
ZAI_STATE_FILE = ZAI_STATE_DIR / "state.json"
ZAI_PROFILE_DIR = ZAI_STATE_DIR / "chromium-profile"
ZAI_BASE_URL = os.environ.get("ZAI_BASE_URL", "https://chat.z.ai")
ZAI_FE_VERSION = os.environ.get("ZAI_FE_VERSION", "prod-fe-1.1.14")
ZAI_SIG_KEY = os.environ.get("ZAI_SIG_KEY", "key-@@@@)))()((9))-xxxx&&&%%%%%")
# Reuse the server-assigned chat_id across turns instead of allocating a
# fresh chat per request. Confirmed via probe/zai_capture.py multi-turn:
# /api/v2/chat/completions accepts the same chat_id with a chain of
# `current_user_message_parent_id` values pointing at the previous turn's
# assistant message id. Set to "0" to disable for debugging.
ZAI_CONTINUATION = os.environ.get("ZAI_CONTINUATION", "1") == "1"

PROXY_API_KEY = os.environ.get("PROXY_API_KEY")  # optional gateway auth

STATE_DIR.mkdir(parents=True, exist_ok=True)
ZAI_STATE_DIR.mkdir(parents=True, exist_ok=True)
