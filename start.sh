#!/usr/bin/env bash
# Start the proxy server in the background, then launch Claude Code pointed at
# it via ANTHROPIC_BASE_URL. Kills the server on exit.
#
# Usage:
#   ./start.sh                  # default backend (deepseek)
#   ./start.sh deepseek [args]  # DeepSeek (chat.deepseek.com)
#   ./start.sh glm [args]       # Z.AI / GLM (chat.z.ai)
#   ./start.sh zai [args]       # alias for glm
#
# Probe flags (intercepted, not forwarded to claude):
#   freeseek glm --tools      # streaming tool-call smoke
#   freeseek glm --continue   # multi-turn chat-id reuse smoke
#   freeseek glm --dump       # dump /api/models
#   freeseek glm --probe "hi" # arbitrary probe prompt
#   freeseek glm --relogin    # re-run browser login (refresh state.json)
#
# Env overrides:
#   BACKEND           deepseek | zai  (positional arg wins if provided)
#   PORT              default 8765
#   MODEL             default depends on backend
#   FAST_MODEL        default depends on backend
#   PROXY_API_KEY     optional; also gates the proxy if set

set -euo pipefail

INVOKE_DIR="$PWD"
cd "$(dirname "$0")"

# Parse optional leading backend positional argument
case "${1:-}" in
  deepseek|ds)  BACKEND=deepseek; shift ;;
  glm|zai|z.ai|z) BACKEND=zai;    shift ;;
esac
BACKEND="${BACKEND:-deepseek}"

# Intercept proxy probe flags so they don't reach Claude Code (which doesn't
# know them). `freeseek glm --tools` runs the streaming-tool-call smoke test;
# `freeseek glm --continue` runs the multi-turn chat-id reuse smoke;
# `freeseek glm --dump` dumps /api/models; everything else forwards to claude.
case "${1:-}" in
  --tools|--continue|--dump|--probe)
    PROBE_ARG="$1"; shift
    if [ "$BACKEND" = "zai" ]; then
      PROBE_MOD="probe.zai_probe"
    else
      PROBE_MOD="probe.probe"
    fi
    if [ "$PROBE_ARG" = "--probe" ]; then
      exec .venv/bin/python -m "$PROBE_MOD" "$@"
    else
      exec .venv/bin/python -m "$PROBE_MOD" "$PROBE_ARG" "$@"
    fi
    ;;
esac

if [ "$BACKEND" = "zai" ]; then
  DEFAULT_MODEL="glm-5-turbo:search"
  DEFAULT_FAST="glm-5-turbo"
  STATE_FILE="$HOME/.zai-proxy/state.json"
  LOGIN_MOD="probe.zai_login"
else
  DEFAULT_MODEL="deepseek-reasoner:search"
  DEFAULT_FAST="deepseek-chat"
  STATE_FILE="$HOME/.deepseek-proxy/state.json"
  LOGIN_MOD="probe.login"
fi

# Force a fresh browser login. Wipes saved state so the next launch
# triggers the login flow.
case "${1:-}" in
  --relogin|--login|relogin|login)
    shift
    echo "Forcing re-login for backend=$BACKEND ($LOGIN_MOD)..."
    rm -f "$STATE_FILE"
    exec .venv/bin/python -m "$LOGIN_MOD" "$@"
    ;;
esac

PORT="${PORT:-8765}"
MODEL="${MODEL:-$DEFAULT_MODEL}"
FAST_MODEL="${FAST_MODEL:-$DEFAULT_FAST}"
LOGFILE="${LOGFILE:-/tmp/freeseek-proxy.log}"
PROXY_API_KEY="${PROXY_API_KEY:-local-dev-key}"

if [ ! -x .venv/bin/python ]; then
  echo "error: .venv missing. Run: python3 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

if [ ! -f "$STATE_FILE" ]; then
  echo "No saved $BACKEND login. Running login flow (a browser will open)..."
  .venv/bin/python -m "$LOGIN_MOD"
fi

if command -v claude >/dev/null 2>&1; then
  CLAUDE_BIN="$(command -v claude)"
else
  echo "error: 'claude' CLI not found in PATH. Install via: npm i -g @anthropic-ai/claude-code"
  exit 1
fi

if lsof -ti:"$PORT" >/dev/null 2>&1; then
  echo "error: port $PORT already in use"
  exit 1
fi

echo "Starting $BACKEND proxy on 127.0.0.1:$PORT (logs -> $LOGFILE)..."
BACKEND="$BACKEND" PROXY_API_KEY="$PROXY_API_KEY" \
  .venv/bin/python -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$PORT" --log-level info \
    >"$LOGFILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "Stopping proxy (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 30); do
  if curl -sS -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
    break
  fi
  sleep 0.5
done
if ! curl -sS -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
  echo "error: proxy failed to start. Check $LOGFILE"
  exit 1
fi
echo "Proxy ready (backend=$BACKEND)."

export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
export ANTHROPIC_AUTH_TOKEN="$PROXY_API_KEY"
unset ANTHROPIC_API_KEY
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$FAST_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_DEFAULT_SONNET_MODEL:-$FAST_MODEL}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-$MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_DEFAULT_HAIKU_MODEL:-$FAST_MODEL}"

echo "Launching Claude Code (backend=$BACKEND, model=$MODEL, fast=$FAST_MODEL) in $INVOKE_DIR..."
echo ""
cd "$INVOKE_DIR"
"$CLAUDE_BIN" --dangerously-skip-permissions "$@"
CLAUDE_EXIT=$?
exit $CLAUDE_EXIT
