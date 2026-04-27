"""Streaming parser that filters <tool_call>...</tool_call> blocks out of the
text stream and emits structured ToolCall events.

Input: incremental text chunks from DeepSeek (the model's reply text).
Output: iterator of events:
    {"type": "text", "text": str}                      # text OUTSIDE tool_call tags
    {"type": "tool_call_start", "id": str, "name": str}
    {"type": "tool_call_arg_delta", "id": str, "delta": str}
    {"type": "tool_call_end", "id": str, "name": str, "arguments": dict}
    {"type": "tool_call", "id": str, "name": str, "arguments": dict}  # legacy composite, emitted after end

The parser emits `tool_call_start` as soon as the `"name":"..."` field is
readable inside the envelope, then streams the raw bytes of the `arguments`
JSON value as `tool_call_arg_delta` events as they arrive. This lets the
Anthropic route forward `input_json_delta` frames to the client without
waiting for the full tool call to be buffered.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

_NAME_RE = re.compile(r'"name"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
_ARGS_RE = re.compile(r'"arguments"\s*:\s*')

# Bare-JSON tool-call detector: matches the start of a `{"name":"...",
# "arguments":...}` object. Used when the upstream's stream filter has
# stripped the surrounding tag (Z.AI strips `<tool_call>`).
_BARE_JSON_START_RE = re.compile(r'\{\s*"name"\s*:\s*"')

# GLM/Z.AI native envelope:  Name<arg_key>k</arg_key><arg_value>v</arg_value>...
_XML_OPEN_KEY = "<arg_key>"
_XML_CLOSE_KEY = "</arg_key>"
_XML_OPEN_VAL = "<arg_value>"
_XML_CLOSE_VAL = "</arg_value>"


class ToolCallParser:
    def __init__(self) -> None:
        self._buf = ""
        self._in_tag = False
        self._bare_mode = False  # True when entered via bare-JSON detection
        self._raw_buf = ""
        self._name: str | None = None
        self._call_id: str | None = None
        # JSON mode state
        self._args_started = False
        self._args_done = False
        self._args_depth = 0
        self._args_in_string = False
        self._args_escape = False
        # Mode: None | "json" | "xml" — decided after enough envelope text arrives.
        self._mode: str | None = None
        # XML mode state
        self._xml_pos = 0
        self._xml_state = "expect_key"  # expect_key | in_key | expect_value | in_value
        self._xml_key: str | None = None
        self._xml_pairs: list[tuple[str, str]] = []
        self._xml_brace_open = False

    def feed(self, chunk: str) -> Iterator[dict[str, Any]]:
        """Feed raw text; yield parsed events."""
        self._buf += chunk
        while True:
            if self._in_tag:
                if self._bare_mode:
                    # No CLOSE tag to scan for — stream chars to the JSON
                    # consumer and finish when the wrapper `}` lands.
                    yield from self._consume_inside(self._buf)
                    self._buf = ""
                    if self._args_done:
                        end = _balanced_json_end(self._raw_buf, 0)
                        leftover = self._raw_buf[end:] if end is not None else ""
                        yield from self._finish_tool_call()
                        self._reset_tag_state()
                        if leftover:
                            self._buf = leftover + self._buf
                        continue
                    return

                idx = self._buf.find(CLOSE)
                if idx < 0:
                    # Don't consume the last len(CLOSE)-1 chars: they might be
                    # the start of the close tag.
                    safe = max(0, len(self._buf) - (len(CLOSE) - 1))
                    if safe > 0:
                        yield from self._consume_inside(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    return
                yield from self._consume_inside(self._buf[:idx])
                self._buf = self._buf[idx + len(CLOSE):]
                yield from self._finish_tool_call()
                self._reset_tag_state()
                continue

            # Not in tag. Look for next OPEN tag, or fall back to bare-JSON
            # tool-call detection (used when the upstream stripped the tag).
            open_idx = self._buf.find(OPEN)
            bare_match = _BARE_JSON_START_RE.search(self._buf)
            bare_idx = bare_match.start() if bare_match else -1

            if open_idx < 0 and bare_idx < 0:
                # Neither — keep enough buffer for a partial OPEN/bare match.
                margin = max(len(OPEN), 16)
                safe = max(0, len(self._buf) - margin)
                if safe > 0:
                    yield {"type": "text", "text": self._buf[:safe]}
                    self._buf = self._buf[safe:]
                return

            # Choose whichever marker comes first.
            if open_idx >= 0 and (bare_idx < 0 or open_idx <= bare_idx):
                if open_idx > 0:
                    yield {"type": "text", "text": self._buf[:open_idx]}
                self._buf = self._buf[open_idx + len(OPEN):]
                self._in_tag = True
                self._bare_mode = False
                self._call_id = f"call_{uuid.uuid4().hex[:16]}"
                continue

            # Bare-JSON path: enter in_tag with bare_mode so args stream
            # incrementally as chunks arrive. Leftover text after the
            # wrapper `}` is put back into self._buf.
            if bare_idx > 0:
                yield {"type": "text", "text": self._buf[:bare_idx]}
                self._buf = self._buf[bare_idx:]
            self._in_tag = True
            self._bare_mode = True
            self._call_id = f"call_{uuid.uuid4().hex[:16]}"
            continue

    def flush(self) -> Iterator[dict[str, Any]]:
        """Flush any remaining buffered text at end-of-stream."""
        if self._in_tag:
            return  # unterminated; drop
        if self._buf:
            yield {"type": "text", "text": self._buf}
            self._buf = ""

    # ---- internals ----

    def _consume_inside(self, piece: str) -> Iterator[dict[str, Any]]:
        if not piece:
            return
        prev_raw_len = len(self._raw_buf)
        self._raw_buf += piece

        # Decide mode once we have enough signal. JSON envelopes start with
        # `{...}`; GLM XML envelopes have a leading NAME followed by
        # `<arg_key>`. Both are detectable as soon as either marker appears.
        if self._mode is None:
            if _NAME_RE.search(self._raw_buf):
                self._mode = "json"
            elif _XML_OPEN_KEY in self._raw_buf:
                self._mode = "xml"
            else:
                return  # need more data

        if self._mode == "xml":
            yield from self._consume_xml()
            return

        # ---- JSON mode (DeepSeek default) ----
        if self._name is None:
            m = _NAME_RE.search(self._raw_buf)
            if m:
                self._name = m.group(1)
                yield {
                    "type": "tool_call_start",
                    "id": self._call_id,
                    "name": self._name,
                }

        if self._name is None or self._args_done:
            return

        if not self._args_started:
            m = _ARGS_RE.search(self._raw_buf)
            if not m:
                return
            self._args_started = True
            remainder = self._raw_buf[m.end():]
            yield from self._emit_args_chars(remainder)
            return

        # args already streaming; only emit the new piece
        new_piece = self._raw_buf[prev_raw_len:]
        yield from self._emit_args_chars(new_piece)

    def _consume_xml(self) -> Iterator[dict[str, Any]]:
        """Stream GLM-style <arg_key>/<arg_value> pairs.

        Emits incremental input_json_delta-friendly chunks: opening `{`, then
        each completed pair as `[,]"k":"v"`, then closing `}` on envelope
        close (handled in _finish_tool_call).
        """
        if self._name is None:
            idx = self._raw_buf.find(_XML_OPEN_KEY)
            if idx < 0:
                return
            name = self._raw_buf[:idx].strip()
            if not name:
                # malformed: no leading name; bail to flush-time fallback
                return
            self._name = name
            self._xml_pos = idx
            yield {
                "type": "tool_call_start",
                "id": self._call_id,
                "name": self._name,
            }
            yield {
                "type": "tool_call_arg_delta",
                "id": self._call_id,
                "delta": "{",
            }
            self._xml_brace_open = True

        # Walk forward through pairs that are now fully present in raw_buf.
        while True:
            if self._xml_state == "expect_key":
                i = self._raw_buf.find(_XML_OPEN_KEY, self._xml_pos)
                if i < 0:
                    return
                self._xml_pos = i + len(_XML_OPEN_KEY)
                self._xml_state = "in_key"
                continue
            if self._xml_state == "in_key":
                i = self._raw_buf.find(_XML_CLOSE_KEY, self._xml_pos)
                if i < 0:
                    return
                self._xml_key = self._raw_buf[self._xml_pos:i].strip()
                self._xml_pos = i + len(_XML_CLOSE_KEY)
                self._xml_state = "expect_value"
                continue
            if self._xml_state == "expect_value":
                i = self._raw_buf.find(_XML_OPEN_VAL, self._xml_pos)
                if i < 0:
                    return
                self._xml_pos = i + len(_XML_OPEN_VAL)
                self._xml_state = "in_value"
                continue
            if self._xml_state == "in_value":
                i = self._raw_buf.find(_XML_CLOSE_VAL, self._xml_pos)
                if i < 0:
                    return
                value = self._raw_buf[self._xml_pos:i]
                self._xml_pos = i + len(_XML_CLOSE_VAL)
                self._xml_state = "expect_key"
                key = self._xml_key or ""
                self._xml_key = None
                self._xml_pairs.append((key, value))
                yield from self._emit_xml_pair(key, value, first=len(self._xml_pairs) == 1)
                continue
            return

    def _emit_xml_pair(self, key: str, value: str, *, first: bool) -> Iterator[dict[str, Any]]:
        # GLM values are raw strings unless the model emits valid JSON;
        # try-decode lets numbers/objects round-trip cleanly.
        try:
            json.loads(value)
            value_json = value
        except (json.JSONDecodeError, ValueError):
            value_json = json.dumps(value)
        sep = "" if first else ","
        yield {
            "type": "tool_call_arg_delta",
            "id": self._call_id,
            "delta": f"{sep}{json.dumps(key)}:{value_json}",
        }

    def _emit_args_chars(self, chars: str) -> Iterator[dict[str, Any]]:
        out: list[str] = []
        for ch in chars:
            if self._args_done:
                break
            if self._args_escape:
                self._args_escape = False
                out.append(ch)
                continue
            if self._args_in_string:
                if ch == "\\":
                    self._args_escape = True
                    out.append(ch)
                    continue
                if ch == '"':
                    self._args_in_string = False
                    out.append(ch)
                    if self._args_depth == 0:
                        # primitive-string arg ended
                        self._args_done = True
                    continue
                out.append(ch)
                continue
            # not in string
            if ch == '"':
                self._args_in_string = True
                out.append(ch)
                continue
            if ch in "{[":
                self._args_depth += 1
                out.append(ch)
                continue
            if ch in "}]":
                if self._args_depth == 0:
                    # this `}` closes the envelope, not the args value
                    self._args_done = True
                    break
                self._args_depth -= 1
                out.append(ch)
                if self._args_depth == 0:
                    self._args_done = True
                continue
            out.append(ch)
        if out:
            yield {
                "type": "tool_call_arg_delta",
                "id": self._call_id,
                "delta": "".join(out),
            }

    def _finish_tool_call(self) -> Iterator[dict[str, Any]]:
        # XML mode: close the streamed `{...}` we opened in _consume_xml and
        # synthesize the final args dict from accumulated pairs.
        if self._mode == "xml" and self._name is not None:
            args: dict[str, Any] = {}
            for k, v in self._xml_pairs:
                try:
                    args[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    args[k] = v
            # Catch the malformed GLM variant the streaming pass can't recover.
            for pair in _XML_MALFORMED_PAIR_RE.finditer(self._raw_buf):
                k = pair.group(1).strip()
                if k in args:
                    continue
                args[k] = pair.group(2)
            if not self._xml_brace_open:
                # No pair ever streamed (e.g., name found but never any args).
                yield {
                    "type": "tool_call_arg_delta",
                    "id": self._call_id,
                    "delta": json.dumps(args, separators=(",", ":")),
                }
            else:
                # Mid-stream we emitted `{` plus zero or more pairs. If the
                # streamed pairs missed entries the malformed-regex pass
                # recovered after the fact, splice them in before the `}`
                # so input_json_delta consumers get the full args dict.
                streamed_keys = {k for k, _ in self._xml_pairs}
                missing = {k: v for k, v in args.items() if k not in streamed_keys}
                if missing:
                    fragments: list[str] = []
                    leading = "" if not self._xml_pairs else ","
                    for i, (k, v) in enumerate(missing.items()):
                        sep = leading if i == 0 else ","
                        fragments.append(
                            f"{sep}{json.dumps(k)}:{json.dumps(v)}"
                        )
                    yield {
                        "type": "tool_call_arg_delta",
                        "id": self._call_id,
                        "delta": "".join(fragments),
                    }
                yield {
                    "type": "tool_call_arg_delta",
                    "id": self._call_id,
                    "delta": "}",
                }
            yield {
                "type": "tool_call_end",
                "id": self._call_id,
                "name": self._name,
                "arguments": args,
            }
            yield {
                "type": "tool_call",
                "id": self._call_id,
                "name": self._name,
                "arguments": args,
            }
            return

        parsed = _parse_tool_call(self._raw_buf)
        if parsed is None:
            return
        # Ensure a start event was emitted even if name extraction failed earlier.
        if self._name is None:
            yield {
                "type": "tool_call_start",
                "id": self._call_id,
                "name": parsed["name"],
            }
        # If the streaming path never emitted arg deltas (e.g. XML-format fallback
        # where "arguments":... was never matched), flush the full args as one delta
        # so downstream clients still see input_json_delta frames.
        if not self._args_started or self._args_depth != 0:
            yield {
                "type": "tool_call_arg_delta",
                "id": self._call_id,
                "delta": json.dumps(parsed["arguments"], separators=(",", ":")),
            }
        yield {
            "type": "tool_call_end",
            "id": self._call_id,
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        }
        # Legacy composite event for routes that still consume the full call at once.
        yield {
            "type": "tool_call",
            "id": self._call_id,
            "name": parsed["name"],
            "arguments": parsed["arguments"],
        }

    def _reset_tag_state(self) -> None:
        self._in_tag = False
        self._bare_mode = False
        self._raw_buf = ""
        self._name = None
        self._call_id = None
        self._args_started = False
        self._args_done = False
        self._args_depth = 0
        self._args_in_string = False
        self._args_escape = False
        self._mode = None
        self._xml_pos = 0
        self._xml_state = "expect_key"
        self._xml_key = None
        self._xml_pairs = []
        self._xml_brace_open = False


def _balanced_json_end(buf: str, start: int) -> int | None:
    """Return the index AFTER the matching `}` of the JSON object that begins
    at buf[start]. Returns None if the object isn't closed yet (i.e., we need
    more chunks). String/escape aware so braces inside strings don't count.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(buf)):
        ch = buf[i]
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Re-serialize parsed tool_call events back into the canonical envelope
    text used by canon_turns / _flatten_block (backend-aware: `<tool_call>`
    for DeepSeek, `[[TOOL_CALL]]` for Z.AI). Used when caching the assistant
    turn so the prefix-hash matches what Claude Code reflattens on the next
    request, and when reflattening prior turns into transcript prompts.
    """
    parts: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                pass
        body = json.dumps({"name": name, "arguments": args})
        parts.append(f"{OPEN}\n{body}\n{CLOSE}")
    return "\n".join(parts)


_XML_PAIR_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>",
    flags=re.DOTALL,
)
_XML_NAME_RE = re.compile(r"\s*([A-Za-z_][\w.-]*)")
# Malformed GLM pattern: '<arg_key>KEY": "VALUE"\n<arg_key>...' — special
# close/open tokens rendered empty and the value was inlined JSON-style.
_XML_MALFORMED_PAIR_RE = re.compile(
    r'<arg_key>\s*([A-Za-z_]\w*)"\s*:\s*"(.*?)"\s*(?=<arg_key>|$)',
    flags=re.DOTALL,
)


def _parse_xml_style(raw: str) -> dict[str, Any] | None:
    """Fallback: GLM-5 emits '<tool_call>Name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>'.

    Tolerates the malformed variant where close/open special tokens rendered
    empty and the value was inlined: '<arg_key>KEY": "VAL"\\n<arg_key>...'.
    """
    if "<arg_key>" not in raw:
        return None
    m = _XML_NAME_RE.match(raw)
    if not m:
        return None
    name = m.group(1)
    args: dict[str, Any] = {}
    for pair in _XML_PAIR_RE.finditer(raw):
        k = pair.group(1).strip()
        v = pair.group(2)
        try:
            args[k] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            args[k] = v
    for pair in _XML_MALFORMED_PAIR_RE.finditer(raw):
        k = pair.group(1).strip()
        if k in args:
            continue
        args[k] = pair.group(2)
    return {
        "type": "tool_call",
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "name": name,
        "arguments": args,
    }


def _parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Parse '{"name":..., "arguments":{...}}' into a ToolCall event.

    Falls back to GLM-5 XML-style '<arg_key>/<arg_value>' format.
    """
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            name = obj.get("name")
            if isinstance(name, str):
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                return {
                    "type": "tool_call",
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "name": name,
                    "arguments": args if isinstance(args, dict) else {},
                }
        except json.JSONDecodeError:
            pass
    return _parse_xml_style(raw)
