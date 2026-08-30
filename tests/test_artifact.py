from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_meter import __version__
from llm_meter.artifact import (
    SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    build_run,
    from_json,
    sanitize_endpoint,
    to_json,
    write_artifact,
)
from llm_meter.models import (
    Completion,
    ErrorObservation,
    RawObservations,
    RequestStart,
    ResponseEstablished,
    RunConfiguration,
    RunStatus,
    StreamEvent,
    TokenCountSource,
    Usage,
    WorkloadProvenance,
)


def _make_run():
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=64,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        response_established=ResponseEstablished(
            offset_ns=5_000_000,
            status_code=200,
            content_type="text/event-stream",
        ),
        stream_events=[
            StreamEvent(sequence=0, offset_ns=10_000_000, event_type="metadata"),
            StreamEvent(
                sequence=1,
                offset_ns=50_000_000,
                event_type="content",
                text_delta="Hello",
            ),
            StreamEvent(
                sequence=2,
                offset_ns=80_000_000,
                event_type="content",
                text_delta=" world",
            ),
            StreamEvent(
                sequence=3,
                offset_ns=90_000_000,
                event_type="metadata",
                finish_reason="stop",
                usage={"prompt_tokens": 5, "completion_tokens": 2},
            ),
        ],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=5, output_tokens=2, source=TokenCountSource.SERVER_REPORTED),
    )
    return build_run(
        run_id="test-run-id",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
    )


def test_schema_version_present() -> None:
    run = _make_run()
    assert run.schema_version == SCHEMA_VERSION
    data = json.loads(to_json(run))
    assert data["schema_version"] == SCHEMA_VERSION


def test_json_round_trip() -> None:
    run = _make_run()
    json_str = to_json(run)
    restored = from_json(json_str)

    assert restored.schema_version == run.schema_version
    assert restored.run_id == run.run_id
    assert restored.started_at == run.started_at
    assert restored.run_status == run.run_status
    assert restored.configuration.endpoint == run.configuration.endpoint
    assert restored.configuration.model == run.configuration.model
    assert restored.configuration.streaming == run.configuration.streaming
    assert restored.request_start.offset_ns == run.request_start.offset_ns
    assert restored.response_established.status_code == run.response_established.status_code
    assert len(restored.stream_events) == len(run.stream_events)
    assert restored.completion.offset_ns == run.completion.offset_ns
    assert restored.usage.input_tokens == run.usage.input_tokens
    assert restored.usage.output_tokens == run.usage.output_tokens
    assert restored.usage.source == run.usage.source
    assert restored.metrics.client_ttft_ns == run.metrics.client_ttft_ns
    assert restored.metrics.e2e_latency_ns == run.metrics.e2e_latency_ns
    assert restored.metrics.inter_chunk_latencies_ns == run.metrics.inter_chunk_latencies_ns
    assert restored.metrics.tpot_ns == run.metrics.tpot_ns
    assert restored.metrics.tpot_status == run.metrics.tpot_status
    assert restored.provenance.llm_meter_version == run.provenance.llm_meter_version


def test_no_secrets_in_serialized_output() -> None:
    run = _make_run()
    json_str = to_json(run)
    assert "api_key" not in json_str
    assert "authorization" not in json_str
    assert "bearer" not in json_str.lower()
    assert "secret" not in json_str.lower()


def test_ns_fields_are_integers() -> None:
    run = _make_run()
    data = json.loads(to_json(run))

    assert isinstance(data["request_start"]["offset_ns"], int)
    assert isinstance(data["response_established"]["offset_ns"], int)
    assert isinstance(data["completion"]["offset_ns"], int)
    for event in data["stream_events"]:
        assert isinstance(event["offset_ns"], int)
    assert isinstance(data["metrics"]["client_ttft_ns"], int)
    assert isinstance(data["metrics"]["e2e_latency_ns"], int)
    for lag in data["metrics"]["inter_chunk_latencies_ns"]:
        assert isinstance(lag, int)


def test_absent_data_is_null_not_fabricated() -> None:
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=None,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        response_established=None,
        completion=None,
        error=ErrorObservation(
            offset_ns=100_000_000,
            category="http_error",
            status=500,
            message="server error",
        ),
        usage=Usage(source=TokenCountSource.UNKNOWN),
    )
    run = build_run(
        run_id="error-run-id",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
    )

    data = json.loads(to_json(run))

    assert data["response_established"] is None
    assert data["completion"] is None
    assert data["error"] is not None
    assert data["error"]["category"] == "http_error"
    assert data["error"]["status"] == 500
    assert data["usage"]["input_tokens"] is None
    assert data["usage"]["output_tokens"] is None
    assert data["usage"]["source"] == "unknown"
    assert data["metrics"]["client_ttft_ns"] is None
    assert data["metrics"]["e2e_latency_ns"] is None
    assert data["metrics"]["tpot_ns"] is None
    assert data["run_status"] == RunStatus.FAILED.value


def test_write_artifact_to_file(tmp_path: Path) -> None:
    run = _make_run()
    output_path = tmp_path / "runs" / "test-run.json"
    result = write_artifact(run, output_path)

    assert result == output_path
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert json.loads(content)["run_id"] == "test-run-id"


def test_provenance_has_version() -> None:
    run = _make_run()
    data = json.loads(to_json(run))
    assert data["provenance"]["llm_meter_version"] == __version__


def test_provenance_required_field() -> None:
    with pytest.raises(TypeError):
        BenchmarkRun = __import__(
            "llm_meter.models", fromlist=["BenchmarkRun"]
        ).BenchmarkRun
        BenchmarkRun(
            schema_version="1",
            run_id="test",
            started_at="2025-01-01T00:00:00Z",
            run_status="completed",
            configuration=RunConfiguration(
                endpoint="http://localhost:8000/v1",
                model="test",
                streaming=True,
            ),
        )


def test_unsupported_schema_version_rejected() -> None:
    future_json = json.dumps({
        "schema_version": "2",
        "run_id": "test",
        "started_at": "2025-01-01T00:00:00Z",
        "run_status": "completed",
        "configuration": {
            "endpoint": "http://localhost:8000/v1",
            "model": "test",
            "streaming": True,
        },
        "provenance": {"llm_meter_version": "0.1.0.dev0"},
    })
    with pytest.raises(UnsupportedSchemaVersion) as exc_info:
        from_json(future_json)
    assert "2" in str(exc_info.value)
    assert SCHEMA_VERSION in str(exc_info.value)


def test_missing_schema_version_rejected() -> None:
    no_version_json = json.dumps({
        "run_id": "test",
        "started_at": "2025-01-01T00:00:00Z",
        "run_status": "completed",
        "configuration": {
            "endpoint": "http://localhost:8000/v1",
            "model": "test",
            "streaming": True,
        },
        "provenance": {"llm_meter_version": "0.1.0.dev0"},
    })
    with pytest.raises(UnsupportedSchemaVersion) as exc_info:
        from_json(no_version_json)
    assert "missing" in str(exc_info.value).lower()


def test_endpoint_credentials_redacted() -> None:
    configuration = RunConfiguration(
        endpoint="https://user:password@example.test/v1?api_key=secret&region=us",
        model="test-model",
        streaming=True,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(source=TokenCountSource.UNKNOWN),
    )
    run = build_run(
        run_id="redact-test",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
    )

    json_str = to_json(run)
    assert "user:password" not in json_str
    assert "secret" not in json_str
    assert "password" not in json_str
    assert "example.test" in json_str
    assert "region=us" in json_str
    assert "***REDACTED***" in json_str


def test_error_message_url_redacted() -> None:
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        error=ErrorObservation(
            offset_ns=10_000_000,
            category="transport",
            status=None,
            exception_type="ConnectError",
            message=(
                "Connection failed: "
                "https://user:password@example.test/v1?api_key=super-secret&region=us"
            ),
        ),
        usage=Usage(source=TokenCountSource.UNKNOWN),
    )
    run = build_run(
        run_id="error-redact-test",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
    )

    json_str = to_json(run)
    assert "user:password" not in json_str
    assert "super-secret" not in json_str
    assert "password" not in json_str
    assert "example.test" in json_str
    assert "region=us" in json_str
    assert "***REDACTED***" in json_str
    assert "Connection failed" in json_str


def test_authorization_credential_redacted() -> None:
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        error=ErrorObservation(
            offset_ns=10_000_000,
            category="http_error",
            status=401,
            exception_type="HTTPStatusError",
            message="request failed: Authorization: Bearer sk-super-secret",
        ),
        usage=Usage(source=TokenCountSource.UNKNOWN),
    )
    run = build_run(
        run_id="bearer-redact-test",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
    )

    json_str = to_json(run)
    assert "sk-super-secret" not in json_str
    assert "Bearer" not in json_str
    assert "Authorization" in json_str
    assert "***REDACTED***" in json_str
    assert "request failed" in json_str


def test_run_status_in_artifact() -> None:
    run = _make_run()
    data = json.loads(to_json(run))
    assert data["run_status"] == RunStatus.COMPLETED.value


def test_sanitize_endpoint_strips_userinfo() -> None:
    result = sanitize_endpoint("https://user:pass@host.example/v1")
    assert "user:pass" not in result
    assert "host.example" in result
    assert "/v1" in result


def test_sanitize_endpoint_redacts_query_params() -> None:
    result = sanitize_endpoint("https://host.example/v1?api_key=secret&region=us")
    assert "secret" not in result
    assert "***REDACTED***" in result
    assert "region=us" in result


def test_sanitize_endpoint_preserves_clean_url() -> None:
    result = sanitize_endpoint("http://localhost:8000/v1")
    assert result == "http://localhost:8000/v1"


def test_sanitize_endpoint_preserves_encoded_values() -> None:
    result = sanitize_endpoint("https://host.example/v1?label=a%20b%26c&api_key=secret")
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(result)
    params = parse_qs(parsed.query)
    assert params["label"] == ["a b&c"]
    assert params["api_key"] == ["***REDACTED***"]
    assert "secret" not in result
    assert "host.example" in result


_PR2_ARTIFACT_JSON = json.dumps({
    "schema_version": "1",
    "run_id": "pr2-legacy-run",
    "started_at": "2025-01-01T00:00:00Z",
    "run_status": "completed",
    "configuration": {
        "endpoint": "http://localhost:8000/v1",
        "model": "test-model",
        "streaming": True,
        "max_output_tokens": 64,
    },
    "provenance": {"llm_meter_version": "0.1.0.dev0"},
    "request_start": {
        "offset_ns": 0,
        "wall_clock_utc": "2025-01-01T00:00:00Z",
    },
    "response_established": {
        "offset_ns": 5_000_000,
        "status_code": 200,
        "content_type": "text/event-stream",
    },
    "stream_events": [
        {
            "sequence": 0,
            "offset_ns": 10_000_000,
            "event_type": "content",
            "text_delta": "Hi",
            "finish_reason": None,
            "usage": None,
        },
        {
            "sequence": 1,
            "offset_ns": 20_000_000,
            "event_type": "metadata",
            "text_delta": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    ],
    "completion": {"offset_ns": 30_000_000, "wall_clock_utc": "2025-01-01T00:00:01Z"},
    "error": None,
    "usage": {"input_tokens": 2, "output_tokens": 1, "source": "server_reported"},
    "metrics": {
        "client_ttft_ns": 10_000_000,
        "e2e_latency_ns": 30_000_000,
        "inter_chunk_latencies_ns": [10_000_000],
        "tpot_ns": None,
        "tpot_status": "insufficient_tokens",
    },
})


def test_pr2_artifact_backward_compat_workload_none() -> None:
    restored = from_json(_PR2_ARTIFACT_JSON)
    assert restored.workload is None
    assert restored.run_id == "pr2-legacy-run"
    assert restored.schema_version == "1"
    assert restored.usage.input_tokens == 2


def test_pr2_artifact_no_workload_key() -> None:
    obj = json.loads(_PR2_ARTIFACT_JSON)
    assert "workload" not in obj
    restored = from_json(_PR2_ARTIFACT_JSON)
    assert restored.workload is None


def _make_workload_provenance() -> WorkloadProvenance:
    return WorkloadProvenance(
        source="builtin",
        seed=42,
        input_tokens_target=100,
        output_tokens_target=64,
        input_tokens_actual_local=98,
        resolution_status="nearest",
        prompt_sha256="abc123def456" + "0" * 52,
        prompt_chars=420,
        tokenizer_provider="fake",
        tokenizer_id="fake-test",
        tokenizer_revision=None,
    )


def test_workload_round_trip() -> None:
    wl = _make_workload_provenance()
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=64,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=5, output_tokens=2, source=TokenCountSource.SERVER_REPORTED),
    )
    run = build_run(
        run_id="workload-run",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
        workload=wl,
    )

    json_str = to_json(run)
    restored = from_json(json_str)

    assert restored.workload is not None
    assert restored.workload.source == "builtin"
    assert restored.workload.seed == 42
    assert restored.workload.input_tokens_target == 100
    assert restored.workload.output_tokens_target == 64
    assert restored.workload.input_tokens_actual_local == 98
    assert restored.workload.resolution_status == "nearest"
    assert restored.workload.prompt_sha256 == wl.prompt_sha256
    assert restored.workload.prompt_chars == 420
    assert restored.workload.tokenizer_provider == "fake"
    assert restored.workload.tokenizer_id == "fake-test"
    assert restored.workload.tokenizer_revision is None


def test_prompt_text_absent_from_artifact() -> None:
    wl = _make_workload_provenance()
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=64,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=5, output_tokens=2, source=TokenCountSource.SERVER_REPORTED),
    )
    run = build_run(
        run_id="no-prompt-run",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
        workload=wl,
    )

    json_str = to_json(run)
    data = json.loads(json_str)
    assert "prompt" not in data
    assert "prompt_text" not in data
    if data.get("workload"):
        assert "prompt" not in data["workload"]


def test_workload_tokenizer_revision_null_in_json() -> None:
    wl = _make_workload_provenance()
    assert wl.tokenizer_revision is None
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=64,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=5, output_tokens=2, source=TokenCountSource.SERVER_REPORTED),
    )
    run = build_run(
        run_id="null-rev-run",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
        workload=wl,
    )
    data = json.loads(to_json(run))
    assert data["workload"]["tokenizer_revision"] is None


def test_local_count_and_server_input_tokens_differ_and_survive() -> None:
    wl = WorkloadProvenance(
        source="manual",
        seed=0,
        input_tokens_target=0,
        output_tokens_target=64,
        input_tokens_actual_local=7,
        resolution_status="nearest",
        prompt_sha256="a" * 64,
        prompt_chars=7,
        tokenizer_provider="fake",
        tokenizer_id="fake-test",
        tokenizer_revision=None,
    )
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=64,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=15, output_tokens=2, source=TokenCountSource.SERVER_REPORTED),
    )
    run = build_run(
        run_id="differ-run",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
        workload=wl,
    )

    json_str = to_json(run)
    restored = from_json(json_str)

    assert restored.workload.input_tokens_actual_local == 7
    assert restored.usage.input_tokens == 15
    assert restored.workload.input_tokens_actual_local != restored.usage.input_tokens


def test_workload_none_default_in_build_run() -> None:
    configuration = RunConfiguration(
        endpoint="http://localhost:8000/v1",
        model="test-model",
        streaming=True,
        max_output_tokens=64,
    )
    observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        completion=Completion(offset_ns=100_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=5, output_tokens=2, source=TokenCountSource.SERVER_REPORTED),
    )
    run = build_run(
        run_id="no-wl-run",
        started_at="2025-01-01T00:00:00Z",
        configuration=configuration,
        observations=observations,
    )
    assert run.workload is None
