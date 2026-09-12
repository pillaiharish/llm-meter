from __future__ import annotations

import copy
import json

import pytest

from llm_meter.artifact import session_from_json, session_to_json
from llm_meter.models import (
    BenchmarkRun,
    BenchmarkSession,
    ClientMetrics,
    ErrorObservation,
    Provenance,
    RunConfiguration,
    SessionConfiguration,
    SessionRequest,
    TokenCountSource,
    Usage,
)
from llm_meter.summary import linear_percentile, summarize_session


def _run(
    *,
    status: str = "completed",
    ttft: int | None = 10,
    e2e: int | None = 100,
    tpot: int | None = 20,
    tpot_status: str = "ok",
    inter_chunk: list[int] | None = None,
    output_tokens: int | None = 10,
    source: TokenCountSource = TokenCountSource.SERVER_REPORTED,
    error_category: str | None = None,
) -> BenchmarkRun:
    return BenchmarkRun(
        schema_version="1",
        run_id="run",
        started_at="2025-01-01T00:00:00+00:00",
        run_status=status,
        configuration=RunConfiguration("http://example.test/v1", "model", True),
        provenance=Provenance("test"),
        error=(
            ErrorObservation(offset_ns=1, category=error_category)
            if error_category is not None
            else None
        ),
        usage=Usage(input_tokens=999, output_tokens=output_tokens, source=source),
        metrics=ClientMetrics(
            client_ttft_ns=ttft,
            e2e_latency_ns=e2e,
            tpot_ns=tpot,
            tpot_status=tpot_status,
            inter_chunk_latencies_ns=inter_chunk or [],
        ),
    )


def _request(
    phase: str,
    run: BenchmarkRun,
    start: int,
    finish: int,
    ordinal: int = 0,
) -> SessionRequest:
    return SessionRequest(phase, ordinal, start, finish, run)


def _session(requests: list[SessionRequest]) -> BenchmarkSession:
    return BenchmarkSession(
        schema_version="1",
        session_id="session",
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:00:10+00:00",
        status="completed",
        configuration=SessionConfiguration(
            endpoint="http://example.test/v1",
            model="model",
            warmup_requests=sum(r.phase == "warmup" for r in requests),
            measured_requests=sum(r.phase == "measured" for r in requests),
            concurrency=2,
            seed=0,
            seed_strategy="base_plus_global_ordinal",
            max_connections=2,
            max_keepalive_connections=2,
            prompt_source="manual",
            input_tokens_target=None,
            output_tokens_target=10,
            tokenizer_id=None,
        ),
        requests=requests,
        provenance=Provenance("test"),
    )


def test_linear_type7_percentiles() -> None:
    assert linear_percentile([10], 0.99) == 10
    assert linear_percentile([10, 20], 0.50) == 15
    assert linear_percentile([10, 20], 0.90) == 19
    assert linear_percentile([10, 20, 30, 40, 50], 0.95) == 48
    assert linear_percentile([50, 10, 30, 20, 40], 0.99) == pytest.approx(49.6)


def test_linear_percentile_rejects_empty_or_invalid_quantile() -> None:
    with pytest.raises(ValueError, match="at least one"):
        linear_percentile([], 0.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        linear_percentile([1], 1.1)


def test_measured_summary_excludes_every_warmup_observation() -> None:
    requests = [
        _request(
            "warmup",
            _run(
                status="failed",
                ttft=999_000,
                e2e=999_000,
                tpot=999_000,
                inter_chunk=[999_000],
                output_tokens=999,
                source=TokenCountSource.ENGINE_REPORTED,
                error_category="warmup_failure",
            ),
            0,
            100_000_000_000,
        ),
        _request(
            "measured",
            _run(ttft=10, e2e=100, tpot=20, inter_chunk=[1, 2]),
            1_000_000_000,
            2_000_000_000,
        ),
        _request(
            "measured",
            _run(
                ttft=20,
                e2e=200,
                tpot=777,
                tpot_status="no_token_count",
                inter_chunk=[3],
                output_tokens=20,
            ),
            2_000_000_000,
            4_000_000_000,
            1,
        ),
        _request(
            "measured",
            _run(ttft=30, e2e=None, tpot=40, inter_chunk=[], output_tokens=30),
            3_000_000_000,
            5_000_000_000,
            2,
        ),
        _request(
            "measured",
            _run(
                status="failed",
                ttft=888,
                e2e=888,
                tpot=888,
                inter_chunk=[888],
                output_tokens=888,
                error_category="http_error",
            ),
            2_500_000_000,
            4_500_000_000,
            3,
        ),
        _request(
            "measured",
            _run(
                status="failed", ttft=777, e2e=777, tpot=777, inter_chunk=[777], output_tokens=777
            ),
            3_500_000_000,
            4_000_000_000,
            4,
        ),
    ]

    summary = summarize_session(_session(requests))

    assert summary.summary_version == "1"
    assert summary.phase == "measured"
    assert summary.percentile_method == "linear_type7"
    assert summary.attempts.attempted == 5
    assert summary.attempts.completed == 3
    assert summary.attempts.failed == 2
    assert summary.attempts.success_rate == 0.6
    assert summary.attempts.error_rate == 0.4
    assert summary.attempts.errors_by_category == {"http_error": 1, "unknown": 1}
    assert summary.ttft.sample_count == 3
    assert summary.ttft.minimum == 10
    assert summary.ttft.maximum == 30
    assert summary.ttft.p50 == 20
    assert summary.ttft.p90 == 28
    assert summary.ttft.p95 == 29
    assert summary.ttft.p99 == pytest.approx(29.8)
    assert summary.e2e.sample_count == 2
    assert summary.tpot.sample_count == 2
    assert summary.tpot.p50 == 30
    assert summary.inter_chunk.sample_count == 3
    assert summary.inter_chunk.sample_unit == "chunk_interval"
    assert summary.inter_chunk.p50 == 2
    assert summary.throughput.measured_duration_ns == 4_000_000_000
    assert summary.throughput.attempted_requests_per_s == 1.25
    assert summary.throughput.completed_requests_per_s == 0.75
    assert summary.throughput.output_tokens_total == 60
    assert summary.throughput.output_tokens_per_s == 15
    assert summary.throughput.output_token_source == "server_reported"
    assert summary.throughput.output_token_status == "ok"


def test_all_failed_session_has_empty_distributions_and_attempt_throughput() -> None:
    session = _session(
        [
            _request("measured", _run(status="failed"), 0, 1_000_000_000),
            _request("measured", _run(status="failed"), 500_000_000, 2_000_000_000, 1),
        ]
    )

    summary = summarize_session(session)

    assert summary.attempts.success_rate == 0
    assert summary.attempts.error_rate == 1
    for distribution in (summary.ttft, summary.e2e, summary.tpot, summary.inter_chunk):
        assert distribution.sample_count == 0
        assert distribution.minimum is None
        assert distribution.maximum is None
        assert distribution.p50 is None
        assert distribution.p90 is None
        assert distribution.p95 is None
        assert distribution.p99 is None
    assert summary.throughput.attempted_requests_per_s == 1
    assert summary.throughput.completed_requests_per_s == 0
    assert summary.throughput.output_token_status == "no_successful_requests"
    assert summary.throughput.output_tokens_per_s is None


@pytest.mark.parametrize(
    ("runs", "finish", "expected"),
    [
        ([_run(output_tokens=None)], 1_000_000_000, "missing_output_token_counts"),
        (
            [
                _run(source=TokenCountSource.SERVER_REPORTED),
                _run(source=TokenCountSource.ENGINE_REPORTED),
            ],
            1_000_000_000,
            "mixed_token_count_sources",
        ),
        ([_run(source=TokenCountSource.UNKNOWN)], 1_000_000_000, "unknown_token_count_source"),
        ([_run()], 0, "non_positive_measured_duration"),
    ],
)
def test_output_token_throughput_unavailable_statuses(
    runs: list[BenchmarkRun], finish: int, expected: str
) -> None:
    requests = [_request("measured", run, 0, finish, ordinal) for ordinal, run in enumerate(runs)]
    throughput = summarize_session(_session(requests)).throughput

    assert throughput.output_token_status == expected
    assert throughput.output_tokens_total is None
    assert throughput.output_tokens_per_s is None
    assert throughput.output_token_source is None


def test_summarization_is_pure_and_deterministic() -> None:
    session = _session([_request("measured", _run(), 0, 1_000_000_000)])
    original = copy.deepcopy(session)

    first = summarize_session(session)
    second = summarize_session(session)

    assert first == second
    assert session == original


def test_summary_json_round_trip_and_raw_requests_preserved() -> None:
    session = _session([_request("measured", _run(), 0, 1_000_000_000)])
    session.summary = summarize_session(session)

    restored = session_from_json(session_to_json(session))

    assert restored.summary == session.summary
    assert restored.requests == session.requests


def test_old_session_without_summary_loads_as_none() -> None:
    session = _session([_request("measured", _run(), 0, 1_000_000_000)])
    data = json.loads(session_to_json(session))
    data.pop("summary")

    assert session_from_json(json.dumps(data)).summary is None


def test_missing_session_status_fails_deserialization() -> None:
    data = json.loads(session_to_json(_session([])))
    data.pop("status")

    with pytest.raises(KeyError, match="status"):
        session_from_json(json.dumps(data))


def test_no_large_numerical_dependencies() -> None:
    dependencies = __import__("tomllib").load(open("pyproject.toml", "rb"))["project"][
        "dependencies"
    ]
    names = {dependency.split(">=")[0].lower() for dependency in dependencies}
    assert names.isdisjoint({"numpy", "pandas", "torch"})
