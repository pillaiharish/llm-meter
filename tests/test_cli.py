from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from llm_meter.cli import main
from llm_meter.models import (
    Completion,
    RawObservations,
    RequestStart,
    StreamEvent,
    TokenCountSource,
    Usage,
)


def test_run_one_help(capsys: object) -> None:
    try:
        main(["run-one", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "run-one" in captured.out
    assert "--endpoint" in captured.out
    assert "--model" in captured.out
    assert "--prompt" in captured.out


def test_run_one_produces_artifact(tmp_path: Path, capsys: object) -> None:
    fake_observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[
            StreamEvent(
                sequence=0,
                offset_ns=10_000_000,
                event_type="content",
                text_delta="Hello",
            ),
            StreamEvent(
                sequence=1,
                offset_ns=20_000_000,
                event_type="metadata",
                finish_reason="stop",
            ),
        ],
        completion=Completion(offset_ns=30_000_000, wall_clock_utc="2025-01-01T00:00:01Z"),
        usage=Usage(input_tokens=3, output_tokens=5, source=TokenCountSource.SERVER_REPORTED),
    )

    output_file = tmp_path / "run.json"

    with patch("llm_meter.cli.stream_completion", return_value=fake_observations):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hi",
            "--max-output-tokens", "64",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["schema_version"] == "1"
    assert data["configuration"]["model"] == "test-model"
    assert data["configuration"]["streaming"] is True
    assert data["configuration"]["endpoint"] == "http://localhost:8000/v1"
    assert data["configuration"]["max_output_tokens"] == 64
    assert data["usage"]["output_tokens"] == 5
    assert data["usage"]["source"] == "server_reported"
    assert data["metrics"]["client_ttft_ns"] == 10_000_000
    assert data["metrics"]["e2e_latency_ns"] == 30_000_000
    assert data["error"] is None

    captured = capsys.readouterr()
    assert "run_id" in captured.out
    assert "schema_version" in captured.out
    assert "client_ttft" in captured.out
    assert "e2e_latency" in captured.out


def test_run_one_error_exit_code(tmp_path: Path, capsys: object) -> None:
    from llm_meter.models import ErrorObservation

    fake_observations = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        error=ErrorObservation(
            offset_ns=10_000_000,
            category="http_error",
            status=500,
            message="server error",
        ),
        usage=Usage(source=TokenCountSource.UNKNOWN),
    )

    output_file = tmp_path / "error_run.json"

    with patch("llm_meter.cli.stream_completion", return_value=fake_observations):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hi",
            "--output", str(output_file),
        ])

    assert exit_code == 1
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["error"] is not None
    assert data["error"]["category"] == "http_error"
    assert data["error"]["status"] == 500

    captured = capsys.readouterr()
    assert "error" in captured.out
    assert "http_error" in captured.out