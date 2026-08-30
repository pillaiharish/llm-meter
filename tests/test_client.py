from __future__ import annotations

import json
from typing import Any

import httpx

from llm_meter.client import FakeClock, stream_completion
from llm_meter.models import RunStatus, TokenCountSource


def _make_sse_response(
    events: list[dict[str, Any] | str],
    status_code: int = 200,
) -> httpx.MockTransport:
    lines = []
    for event in events:
        if event == "[DONE]":
            lines.append("data: [DONE]")
        elif isinstance(event, str):
            lines.append(event)
        else:
            lines.append(f"data: {json.dumps(event)}")
    sse_body = "\n".join(lines) + "\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
            request=request,
        )

    return httpx.MockTransport(handler)


def _content_delta(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}, "index": 0}]}


def _role_delta() -> dict[str, Any]:
    return {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}


def _finish(finish_reason: str = "stop", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}]}
    if usage is not None:
        data["usage"] = usage
    return data


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def test_successful_streaming_response() -> None:
    transport = _make_sse_response([
        _role_delta(),
        _content_delta("Hello"),
        _content_delta(" world"),
        _finish("stop", {"prompt_tokens": 5, "completion_tokens": 2}),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            max_output_tokens=64,
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is None
    assert result.completion is not None
    assert result.response_established is not None
    assert result.response_established.status_code == 200
    assert len(result.stream_events) == 4
    assert result.stream_events[0].event_type == "metadata"
    assert result.stream_events[1].event_type == "content"
    assert result.stream_events[2].event_type == "content"
    assert result.stream_events[3].event_type == "metadata"
    assert result.usage.output_tokens == 2
    assert result.usage.input_tokens == 5
    assert result.usage.source == TokenCountSource.SERVER_REPORTED


def test_metadata_only_before_content_not_ttft() -> None:
    transport = _make_sse_response([
        _role_delta(),
        _content_delta("Hello"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 40_000_000, 10_000_000, 10_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    content_events = [e for e in result.stream_events if e.event_type == "content"]
    assert len(content_events) == 1
    assert content_events[0].offset_ns == 55_000_000
    metadata_events = [e for e in result.stream_events if e.event_type == "metadata"]
    assert metadata_events[0].offset_ns == 15_000_000


def test_no_content_bearing_event() -> None:
    transport = _make_sse_response([
        _role_delta(),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    content_events = [e for e in result.stream_events if e.event_type == "content"]
    assert len(content_events) == 0


def test_response_with_usage() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
        _finish("stop", {"prompt_tokens": 3, "completion_tokens": 1}),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 1
    assert result.usage.source == TokenCountSource.SERVER_REPORTED


def test_response_without_usage() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.usage.output_tokens is None
    assert result.usage.source == TokenCountSource.UNKNOWN


def test_http_error() -> None:
    transport = _make_sse_response([], status_code=500)
    clock = FakeClock([10_000_000, 20_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is not None
    assert result.error.category == "http_error"
    assert result.error.status == 500
    assert result.completion is None
    assert result.response_established is not None
    assert result.response_established.status_code == 500


def test_malformed_sse_event() -> None:
    transport = _make_sse_response([
        "data: {invalid json}",
    ])
    clock = FakeClock([5_000_000, 10_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is not None
    assert result.error.category == "sse_parse"


def test_stream_ending_unexpectedly() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is not None
    assert result.error.category == "stream_unexpected_end"
    assert result.completion is None


def test_multiple_content_chunks() -> None:
    transport = _make_sse_response([
        _content_delta("a"),
        _content_delta("b"),
        _content_delta("c"),
        _content_delta("d"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([
        5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000,
    ])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    content_events = [e for e in result.stream_events if e.event_type == "content"]
    assert len(content_events) == 4
    assert [e.text_delta for e in content_events] == ["a", "b", "c", "d"]


def test_partial_artifact_on_error() -> None:
    transport = _make_sse_response([
        _content_delta("partial"),
        "data: {bad json}",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is not None
    assert result.error.category == "sse_parse"
    assert len(result.stream_events) == 1
    assert result.stream_events[0].text_delta == "partial"


def test_no_api_key_in_observations() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            api_key="secret-key-12345",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is None
    for event in result.stream_events:
        assert "secret-key-12345" not in json.dumps(event.usage or {})
    assert "secret-key-12345" not in json.dumps(result.__dict__, default=str)


def test_done_timestamp_defines_completion() -> None:
    transport = _make_sse_response([
        _content_delta("Hello"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 90_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is None
    assert result.completion is not None
    assert result.completion.offset_ns == 105_000_000


def test_trailing_data_after_done_ignored() -> None:
    transport = _make_sse_response([
        _content_delta("Hello"),
        "[DONE]",
        _content_delta("trailing"),
    ])
    clock = FakeClock([5_000_000, 10_000_000, 90_000_000, 100_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.error is None
    assert result.completion is not None
    assert result.completion.offset_ns == 105_000_000
    content_events = [e for e in result.stream_events if e.event_type == "content"]
    assert len(content_events) == 1
    assert content_events[0].text_delta == "Hello"
    assert "trailing" not in [e.text_delta for e in result.stream_events]


def test_eof_time_not_used_for_e2e_after_done() -> None:
    transport = _make_sse_response([
        _content_delta("Hello"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 90_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.completion is not None
    assert result.completion.offset_ns == 105_000_000
    from llm_meter.metrics import derive_metrics

    metrics = derive_metrics(result)
    assert metrics.e2e_latency_ns == 105_000_000


def test_receive_offset_captured_before_parsing() -> None:
    transport = _make_sse_response([
        _content_delta("Hello"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 30_000_000, 70_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert len(result.stream_events) == 1
    assert result.stream_events[0].offset_ns == 35_000_000
    assert result.completion is not None
    assert result.completion.offset_ns == 105_000_000


def test_empty_content_does_not_trigger_ttft() -> None:
    transport = _make_sse_response([
        _role_delta(),
        _content_delta(""),
        _content_delta("Hello"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 10_000_000, 40_000_000, 10_000_000, 10_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    content_events = [e for e in result.stream_events if e.event_type == "content"]
    assert len(content_events) == 1
    assert content_events[0].text_delta == "Hello"
    assert content_events[0].offset_ns == 65_000_000
    metadata_events = [e for e in result.stream_events if e.event_type == "metadata"]
    assert len(metadata_events) == 3
    empty_content_deltas = [
        e for e in result.stream_events if e.text_delta is None and e.event_type == "metadata"
    ]
    assert len(empty_content_deltas) >= 1


def test_response_established_captured() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.response_established is not None
    assert result.response_established.status_code == 200
    assert result.response_established.offset_ns == 5_000_000
    assert result.response_established.content_type == "text/event-stream"


def test_response_established_on_http_error() -> None:
    transport = _make_sse_response([], status_code=403)
    clock = FakeClock([10_000_000, 20_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    assert result.response_established is not None
    assert result.response_established.status_code == 403
    assert result.error is not None
    assert result.error.category == "http_error"


def test_run_status_completed() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
        _finish("stop"),
        "[DONE]",
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000, 30_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    from llm_meter.artifact import build_run
    from llm_meter.models import RunConfiguration

    run = build_run(
        run_id="test-id",
        started_at="2025-01-01T00:00:00Z",
        configuration=RunConfiguration(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            streaming=True,
        ),
        observations=result,
    )
    assert run.run_status == RunStatus.COMPLETED.value


def test_run_status_failed_on_http_error() -> None:
    transport = _make_sse_response([], status_code=500)
    clock = FakeClock([10_000_000, 20_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    from llm_meter.artifact import build_run
    from llm_meter.models import RunConfiguration

    run = build_run(
        run_id="test-id",
        started_at="2025-01-01T00:00:00Z",
        configuration=RunConfiguration(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            streaming=True,
        ),
        observations=result,
    )
    assert run.run_status == RunStatus.FAILED.value


def test_run_status_failed_on_unexpected_eof() -> None:
    transport = _make_sse_response([
        _content_delta("Hi"),
    ])
    clock = FakeClock([5_000_000, 10_000_000, 20_000_000])

    result = asyncio_run(
        stream_completion(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            prompt="hi",
            clock=clock,
            transport=transport,
        )
    )

    from llm_meter.artifact import build_run
    from llm_meter.models import RunConfiguration

    run = build_run(
        run_id="test-id",
        started_at="2025-01-01T00:00:00Z",
        configuration=RunConfiguration(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            streaming=True,
        ),
        observations=result,
    )
    assert run.run_status == RunStatus.FAILED.value
