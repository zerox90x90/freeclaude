"""Build a system-prompt contract that makes the upstream emit tool calls."""
from __future__ import annotations

import json
from typing import Any

from app.config import BACKEND

# DeepSeek-friendly template: emit JSON inside <tool_call>...</tool_call>.
# On Z.AI the wrapper tags are stripped from the model's output stream, but
# the parser detects the bare `{"name":..., "arguments":...}` JSON that
# survives — see app.tools.parser._BARE_JSON_START_RE.
TOOL_SYSTEM_TEMPLATE_JSON = """You have access to the following tools. To call a tool, emit EXACTLY this format — no code fences, no XML escaping, no commentary inside the tags:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

Rules:
- One JSON object per <tool_call> block. You may emit multiple <tool_call> blocks in one reply to call several tools.
- `arguments` MUST be a JSON object (use {{}} when a tool needs none), never a string.
- Do NOT wrap <tool_call>...</tool_call> in markdown code fences (```), do NOT indent, do NOT prefix with "Tool:" or similar.
- After emitting tool calls, stop. Wait for tool results in the next user turn before continuing. Do not guess tool output.
- If you do NOT need a tool, answer the user directly in plain text without any <tool_call> tags.
- If the previous user message contains <tool_result id="...">...</tool_result> blocks, those are outputs of tools you previously called. READ them silently, then continue the user's original task — either by emitting more <tool_call> blocks or by writing your final answer. DO NOT repeat, echo, quote, or paraphrase the <tool_result> content back at the user.
- Never start your reply with "User:", "Assistant:", "[tool_result", "<tool_result", "[USER]", "[ASSISTANT]", or any transcript marker. Your reply is the NEXT assistant turn; do not continue or imitate the transcript format.

Available tools (JSON schema):
{tools_json}
"""


# GLM-flavored template: matches the native `<arg_key>/<arg_value>` envelope
# GLM-5 emits when fine-tuned for tool use. The parser handles both this form
# and the JSON form (see app.tools.parser._consume_xml), so the only reason to
# pick this one is reliability — GLM follows its native format more cleanly
# than a foreign JSON contract.
TOOL_SYSTEM_TEMPLATE_GLM_XML = """You have access to the following tools. To call a tool, emit EXACTLY this format inside a <tool_call> envelope — no code fences, no commentary inside the tags:

<tool_call>tool_name
<arg_key>arg1</arg_key>
<arg_value>value1</arg_value>
<arg_key>arg2</arg_key>
<arg_value>value2</arg_value>
</tool_call>

Rules:
- The first line inside <tool_call> is the tool name (case-sensitive, bare identifier — no quotes, no "Tool:" prefix).
- One <arg_key>/<arg_value> pair per argument. <arg_value> contents may be raw text or a JSON literal (number, boolean, array, object).
- When the schema says a parameter is `string`, the value MUST be JSON-quoted (e.g. `<arg_value>"1"</arg_value>`, not `<arg_value>1</arg_value>`) — bare digits, `true`/`false`, or `null` get parsed as their JSON types and fail string validation. This applies to IDs, status enums, and any other string field that looks numeric or boolean.
- Emit multiple <tool_call>...</tool_call> blocks to call several tools in one reply.
- Do NOT wrap <tool_call>...</tool_call> in markdown code fences (```), do NOT indent.
- After emitting tool calls, stop. Wait for tool results in the next user turn before continuing. Do not invent tool output.
- If you do NOT need a tool, answer the user directly in plain text without any <tool_call> tags.
- If the previous user message contains <tool_result id="...">...</tool_result> blocks, those are outputs of tools you previously called. READ them silently, then continue the user's original task. DO NOT repeat, echo, quote, or paraphrase the <tool_result> content back at the user.
- Never start your reply with "User:", "Assistant:", "[tool_result", "<tool_result", "[USER]", "[ASSISTANT]", or any transcript marker. Your reply is the NEXT assistant turn; do not continue or imitate the transcript format.

Available tools (JSON schema):
{tools_json}
"""


# Claude Code ships these as Anthropic server-side tools that the proxy cannot
# execute. DeepSeek's :search suffix replaces WebSearch via upstream web-RAG.
_UNSUPPORTED_TOOL_NAMES = {"WebSearch"}
_UNSUPPORTED_TOOL_TYPES = {
    "web_search_20250305",
    "web_search",
}


def _is_unsupported(name: str, type_: str) -> bool:
    return name in _UNSUPPORTED_TOOL_NAMES or type_ in _UNSUPPORTED_TOOL_TYPES


def normalize_openai_tools(tools: list[dict]) -> list[dict]:
    """Accept OpenAI tool format: {type:"function", function:{name, description, parameters}}."""
    out = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            name = fn.get("name", "")
            if _is_unsupported(name, t.get("type", "")):
                continue
            out.append(
                {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
    return out


def normalize_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool format: {name, description, input_schema}."""
    out = []
    for t in tools:
        name = t.get("name", "")
        if _is_unsupported(name, t.get("type", "")):
            continue
        out.append(
            {
                "name": name,
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
        )
    return out


def tool_system_block(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    template = TOOL_SYSTEM_TEMPLATE_GLM_XML if BACKEND == "zai" else TOOL_SYSTEM_TEMPLATE_JSON
    return template.format(tools_json=json.dumps(tools, indent=2))
