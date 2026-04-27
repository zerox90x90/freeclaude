"""Structured output via response_format."""
from __future__ import annotations

import json

try:
    from jsonschema import Draft202012Validator
except ImportError:  # validated at runtime
    Draft202012Validator = None  # type: ignore


def structured_system_block(response_format: dict | None) -> str:
    if not response_format:
        return ""
    t = response_format.get("type")
    if t == "json_object":
        return (
            "You MUST reply with a single valid JSON object. No prose, no code "
            "fences, no commentary — only the JSON object."
        )
    if t == "json_schema":
        schema = (
            response_format.get("json_schema", {}).get("schema")
            or response_format.get("schema")
            or {}
        )
        return (
            "You MUST reply with a single valid JSON object conforming to this "
            f"JSON Schema. No prose or code fences, only the JSON object.\n\n"
            f"Schema:\n{json.dumps(schema, indent=2)}"
        )
    return ""


def extract_schema(response_format: dict | None) -> dict | None:
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        return (
            response_format.get("json_schema", {}).get("schema")
            or response_format.get("schema")
        )
    return None


def validate_structured(text: str, response_format: dict | None) -> tuple[bool, str]:
    """Return (ok, error_message). On json_object type, only checks parseability.
    On json_schema, validates against the schema."""
    if not response_format:
        return True, ""
    t = response_format.get("type")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return False, f"not valid JSON: {e}"
    if t == "json_schema":
        schema = extract_schema(response_format) or {}
        if Draft202012Validator is None:
            return True, ""
        errs = list(Draft202012Validator(schema).iter_errors(obj))
        if errs:
            return False, "; ".join(e.message for e in errs[:3])
    return True, ""
