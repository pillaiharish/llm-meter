from __future__ import annotations

import json
from typing import Any


class SSEParseError(Exception):
    pass


class SSEDoneSentinel:
    pass


DONE = SSEDoneSentinel()


def parse_sse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(":"):
        return None
    if ":" in stripped:
        field_name, _, field_value = stripped.partition(":")
        if field_value.startswith(" "):
            field_value = field_value[1:]
        return field_name, field_value
    return stripped, ""


def parse_sse_data(line: str) -> dict[str, Any] | SSEDoneSentinel:
    parsed = parse_sse_line(line)
    if parsed is None:
        return {}
    field_name, field_value = parsed
    if field_name != "data":
        return {}
    if field_value == "[DONE]":
        return DONE
    try:
        return json.loads(field_value)
    except json.JSONDecodeError as exc:
        raise SSEParseError(f"malformed JSON in SSE data: {exc}") from exc


def iter_sse_events(raw_stream: str):
    for line in raw_stream.splitlines():
        result = parse_sse_data(line)
        if isinstance(result, SSEDoneSentinel):
            yield DONE
        elif result:
            yield result