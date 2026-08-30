from __future__ import annotations

import pytest

from llm_meter.sse import DONE, SSEParseError, parse_sse_data, parse_sse_line


def test_parse_normal_data_line() -> None:
    result = parse_sse_data('data: {"choices": []}')
    assert isinstance(result, dict)
    assert result == {"choices": []}


def test_parse_done_sentinel() -> None:
    result = parse_sse_data("data: [DONE]")
    assert result is DONE


def test_parse_comment_line() -> None:
    assert parse_sse_data(": keepalive") == {}


def test_parse_empty_line() -> None:
    assert parse_sse_data("") == {}


def test_parse_non_data_field() -> None:
    assert parse_sse_data("event: ping") == {}


def test_parse_malformed_json() -> None:
    with pytest.raises(SSEParseError):
        parse_sse_data("data: {invalid json}")


def test_parse_sse_line_with_space_prefix() -> None:
    result = parse_sse_line("data: hello")
    assert result == ("data", "hello")


def test_parse_sse_line_no_space_after_colon() -> None:
    result = parse_sse_line("data:hello")
    assert result == ("data", "hello")
