from __future__ import annotations

from llm_meter.models import (
    RawObservations,
    RequestStart,
    ResponseEstablished,
    RunStatus,
    StreamEvent,
    TokenCountSource,
    Usage,
)


def test_token_count_source_enum_values() -> None:
    assert TokenCountSource.SERVER_REPORTED.value == "server_reported"
    assert TokenCountSource.ENGINE_REPORTED.value == "engine_reported"
    assert TokenCountSource.LOCALLY_TOKENIZED.value == "locally_tokenized"
    assert TokenCountSource.UNKNOWN.value == "unknown"


def test_run_status_enum_values() -> None:
    assert RunStatus.COMPLETED.value == "completed"
    assert RunStatus.FAILED.value == "failed"


def test_stream_event_defaults() -> None:
    event = StreamEvent(sequence=0, offset_ns=100, event_type="metadata")
    assert event.text_delta is None
    assert event.finish_reason is None
    assert event.usage is None


def test_usage_defaults() -> None:
    usage = Usage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.source == TokenCountSource.UNKNOWN


def test_raw_observations_defaults() -> None:
    obs = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
    )
    assert obs.response_established is None
    assert obs.completion is None
    assert obs.error is None
    assert obs.usage.source == TokenCountSource.UNKNOWN


def test_response_established_dataclass() -> None:
    re_obs = ResponseEstablished(
        offset_ns=5_000_000,
        status_code=200,
        content_type="text/event-stream",
    )
    assert re_obs.offset_ns == 5_000_000
    assert re_obs.status_code == 200
    assert re_obs.content_type == "text/event-stream"


def test_response_established_content_type_optional() -> None:
    re_obs = ResponseEstablished(offset_ns=10_000_000, status_code=500)
    assert re_obs.content_type is None
