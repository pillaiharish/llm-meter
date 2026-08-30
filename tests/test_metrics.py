from __future__ import annotations

from llm_meter.metrics import derive_metrics
from llm_meter.models import (
    Completion,
    RawObservations,
    RequestStart,
    StreamEvent,
    TokenCountSource,
    TpotStatus,
    Usage,
)


def _make_observations(
    *,
    stream_events: list[StreamEvent],
    completion: Completion | None = None,
    usage: Usage | None = None,
) -> RawObservations:
    return RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=stream_events,
        completion=completion,
        usage=usage or Usage(source=TokenCountSource.UNKNOWN),
    )


def test_monotonic_event_ordering() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
        StreamEvent(sequence=1, offset_ns=50_000_000, event_type="content", text_delta="Hello"),
        StreamEvent(sequence=2, offset_ns=80_000_000, event_type="content", text_delta=" world"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
    )

    metrics = derive_metrics(obs)

    offsets = [e.offset_ns for e in events]
    assert offsets == sorted(offsets)
    assert metrics.inter_chunk_latencies_ns == [40_000_000, 30_000_000]


def test_metadata_only_event_not_ttft() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
        StreamEvent(sequence=1, offset_ns=50_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
    )

    metrics = derive_metrics(obs)

    assert metrics.client_ttft_ns == 50_000_000


def test_first_content_event_determines_ttft() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
        StreamEvent(sequence=1, offset_ns=50_000_000, event_type="content", text_delta="first"),
        StreamEvent(sequence=2, offset_ns=80_000_000, event_type="content", text_delta="second"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
    )

    metrics = derive_metrics(obs)

    assert metrics.client_ttft_ns == 50_000_000


def test_no_content_event_no_ttft() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
    )

    metrics = derive_metrics(obs)

    assert metrics.client_ttft_ns is None


def test_inter_chunk_latency_from_all_events() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
        StreamEvent(sequence=1, offset_ns=30_000_000, event_type="content", text_delta="a"),
        StreamEvent(sequence=2, offset_ns=70_000_000, event_type="content", text_delta="b"),
        StreamEvent(sequence=3, offset_ns=90_000_000, event_type="metadata"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
    )

    metrics = derive_metrics(obs)

    assert metrics.inter_chunk_latencies_ns == [20_000_000, 40_000_000, 20_000_000]


def test_chunks_never_counted_as_tokens() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="a"),
        StreamEvent(sequence=1, offset_ns=20_000_000, event_type="content", text_delta="b"),
        StreamEvent(sequence=2, offset_ns=30_000_000, event_type="content", text_delta="c"),
        StreamEvent(sequence=3, offset_ns=40_000_000, event_type="content", text_delta="d"),
        StreamEvent(sequence=4, offset_ns=50_000_000, event_type="content", text_delta="e"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=60_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(output_tokens=None, source=TokenCountSource.UNKNOWN),
    )

    metrics = derive_metrics(obs)

    assert metrics.tpot_ns is None
    assert metrics.tpot_status == TpotStatus.NO_TOKEN_COUNT.value


def test_server_reported_usage_with_provenance() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(input_tokens=5, output_tokens=10, source=TokenCountSource.SERVER_REPORTED),
    )

    metrics = derive_metrics(obs)

    assert metrics.tpot_ns is not None
    assert metrics.tpot_status == TpotStatus.OK.value


def test_missing_token_usage_tpot_unavailable() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(output_tokens=None, source=TokenCountSource.UNKNOWN),
    )

    metrics = derive_metrics(obs)

    assert metrics.tpot_ns is None
    assert metrics.tpot_status == TpotStatus.NO_TOKEN_COUNT.value


def test_tpot_formula() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=110_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(output_tokens=11, source=TokenCountSource.SERVER_REPORTED),
    )

    metrics = derive_metrics(obs)

    assert metrics.client_ttft_ns == 10_000_000
    assert metrics.e2e_latency_ns == 110_000_000
    decode_window = 110_000_000 - 10_000_000
    expected_tpot = decode_window // (11 - 1)
    assert metrics.tpot_ns == expected_tpot
    assert metrics.tpot_status == TpotStatus.OK.value


def test_output_tokens_one_no_tpot() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(output_tokens=1, source=TokenCountSource.SERVER_REPORTED),
    )

    metrics = derive_metrics(obs)

    assert metrics.tpot_ns is None
    assert metrics.tpot_status == TpotStatus.INSUFFICIENT_TOKENS.value


def test_output_tokens_zero_no_tpot() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(output_tokens=0, source=TokenCountSource.SERVER_REPORTED),
    )

    metrics = derive_metrics(obs)

    assert metrics.tpot_ns is None
    assert metrics.tpot_status == TpotStatus.INSUFFICIENT_TOKENS.value


def test_no_completion_no_e2e() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="content", text_delta="Hello"),
    ]
    obs = _make_observations(stream_events=events, completion=None)

    metrics = derive_metrics(obs)

    assert metrics.e2e_latency_ns is None
    assert metrics.tpot_ns is None
    assert metrics.tpot_status == TpotStatus.NO_E2E.value


def test_no_ttft_no_tpot() -> None:
    events = [
        StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
        usage=Usage(output_tokens=10, source=TokenCountSource.SERVER_REPORTED),
    )

    metrics = derive_metrics(obs)

    assert metrics.client_ttft_ns is None
    assert metrics.tpot_ns is None
    assert metrics.tpot_status == TpotStatus.NO_TTFT.value


def test_empty_content_not_content_type() -> None:
    events = [
        StreamEvent(
            sequence=0,
            offset_ns=10_000_000,
            event_type="metadata",
            text_delta="",
        ),
        StreamEvent(
            sequence=1,
            offset_ns=50_000_000,
            event_type="content",
            text_delta="Hello",
        ),
    ]
    obs = _make_observations(
        stream_events=events,
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:00Z"),
    )

    metrics = derive_metrics(obs)

    assert metrics.client_ttft_ns == 50_000_000
