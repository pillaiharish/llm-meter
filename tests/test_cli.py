from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_meter.cli import main
from llm_meter.models import (
    Completion,
    RawObservations,
    RequestStart,
    StreamEvent,
    TokenCountSource,
    Usage,
)


def _fake_observations() -> RawObservations:
    return RawObservations(
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


def test_run_one_help(capsys: object) -> None:
    try:
        main(["run-one", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "run-one" in captured.out
    assert "Legacy experimental" in captured.out
    assert "--endpoint" in captured.out
    assert "--model" in captured.out
    assert "--prompt" in captured.out


def test_run_one_produces_artifact(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "run.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
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
    assert data["run_status"] == "completed"
    assert "prompt" not in data

    captured = capsys.readouterr()
    assert "run_id" in captured.out
    assert "schema_version" in captured.out
    assert "run_status" in captured.out
    assert "client_ttft" in captured.out
    assert "e2e_latency" in captured.out


def test_run_one_error_exit_code(tmp_path: Path, capsys: object) -> None:
    from llm_meter.models import ErrorObservation

    fake_obs = RawObservations(
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

    with patch("llm_meter.cli.stream_completion", return_value=fake_obs):
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
    assert data["run_status"] == "failed"

    captured = capsys.readouterr()
    assert "error" in captured.out
    assert "http_error" in captured.out
    assert "run_status" in captured.out


def test_workload_inspect_help(capsys: object) -> None:
    try:
        main(["workload", "inspect", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "inspect" in captured.out
    assert "--input-tokens" in captured.out
    assert "--output-tokens" in captured.out
    assert "--tokenizer" in captured.out
    assert "--seed" in captured.out


def test_workload_inspect_produces_output(capsys: object) -> None:
    exit_code = main([
        "workload", "inspect",
        "--tokenizer", "fake",
        "--input-tokens", "50",
        "--output-tokens", "64",
        "--seed", "42",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "input_tokens_target" in captured.out
    assert "input_tokens_actual" in captured.out
    assert "resolution" in captured.out
    assert "prompt_sha256" in captured.out
    assert "prompt_chars" in captured.out
    assert "tokenizer" in captured.out


def test_workload_inspect_show_prompt(capsys: object) -> None:
    exit_code = main([
        "workload", "inspect",
        "--tokenizer", "fake",
        "--input-tokens", "50",
        "--output-tokens", "64",
        "--seed", "42",
        "--show-prompt",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "--- prompt ---" in captured.out


def test_workload_inspect_requires_tokenizer(capsys: object) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([
            "workload", "inspect",
            "--input-tokens", "50",
            "--output-tokens", "64",
        ])
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "--tokenizer" in captured.err


def test_run_one_with_input_tokens(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "run_input_tokens.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--tokenizer", "fake",
            "--input-tokens", "50",
            "--max-output-tokens", "64",
            "--seed", "42",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["workload"] is not None
    assert data["workload"]["source"] == "builtin"
    assert data["workload"]["input_tokens_target"] == 50
    assert data["workload"]["output_tokens_target"] == 64
    assert data["workload"]["tokenizer_provider"] == "fake"
    assert data["workload"]["tokenizer_id"] == "fake"
    assert data["workload"]["tokenizer_revision"] is None
    assert "prompt" not in data


def test_run_one_prompt_xor_input_tokens_rejected(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with pytest.raises(SystemExit) as exc_info:
        main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hi",
            "--input-tokens", "50",
            "--tokenizer", "fake",
            "--max-output-tokens", "64",
            "--output", str(output_file),
        ])
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err
    assert not output_file.exists()


def test_run_one_input_tokens_without_tokenizer_rejected(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with pytest.raises(SystemExit) as exc_info:
        main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--input-tokens", "50",
            "--max-output-tokens", "64",
            "--output", str(output_file),
        ])
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "--tokenizer" in captured.err
    assert not output_file.exists()


def test_run_one_neither_prompt_nor_input_tokens_rejected(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with pytest.raises(SystemExit) as exc_info:
        main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--max-output-tokens", "64",
            "--output", str(output_file),
        ])
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "either" in captured.err.lower() or "--prompt" in captured.err


def test_run_one_manual_prompt_with_tokenizer(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "manual_with_tok.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello world",
            "--tokenizer", "fake",
            "--max-output-tokens", "64",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    data = json.loads(output_file.read_text())
    assert data["workload"]["source"] == "manual"
    assert data["workload"]["input_tokens_actual_local"] == 11
    assert data["workload"]["tokenizer_provider"] == "fake"
    assert data["workload"]["resolution_status"] == "not_applicable"
    assert data["workload"]["input_tokens_target"] is None
    assert data["workload"]["output_tokens_target"] == 64


def test_run_one_manual_prompt_without_tokenizer(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "manual_no_tok.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello world",
            "--max-output-tokens", "64",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    data = json.loads(output_file.read_text())
    assert data["workload"]["source"] == "manual"
    assert data["workload"]["input_tokens_actual_local"] is None
    assert data["workload"]["tokenizer_provider"] is None
    assert data["workload"]["tokenizer_id"] is None
    assert data["workload"]["tokenizer_revision"] is None
    assert data["workload"]["prompt_sha256"] != ""
    assert data["workload"]["prompt_chars"] == 11
    assert data["workload"]["resolution_status"] == "not_applicable"
    assert data["workload"]["input_tokens_target"] is None


def test_run_one_manual_prompt_without_max_output(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "manual_no_max.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello world",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    data = json.loads(output_file.read_text())
    assert data["workload"]["source"] == "manual"
    assert data["workload"]["output_tokens_target"] is None
    assert data["workload"]["input_tokens_target"] is None
    assert data["workload"]["resolution_status"] == "not_applicable"
    assert data["configuration"]["max_output_tokens"] is None


def test_run_one_local_and_server_input_tokens_differ(tmp_path: Path, capsys: object) -> None:
    fake_obs = RawObservations(
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
        usage=Usage(input_tokens=50, output_tokens=5, source=TokenCountSource.SERVER_REPORTED),
    )

    output_file = tmp_path / "differ_tokens.json"

    with patch("llm_meter.cli.stream_completion", return_value=fake_obs):
        exit_code = main([
            "run-one",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--tokenizer", "fake",
            "--input-tokens", "50",
            "--max-output-tokens", "64",
            "--seed", "42",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    data = json.loads(output_file.read_text())
    assert data["workload"]["input_tokens_actual_local"] is not None
    assert data["usage"]["input_tokens"] == 50


def test_run_one_input_tokens_zero_rejected_before_request(tmp_path: Path, capsys: object) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with patch("llm_meter.cli.stream_completion") as mock_stream, \
         patch("llm_meter.cli.load_tokenizer") as mock_load:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "run-one",
                "--endpoint", "http://localhost:8000/v1",
                "--model", "test-model",
                "--tokenizer", "fake",
                "--input-tokens", "0",
                "--max-output-tokens", "64",
                "--output", str(output_file),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "positive" in captured.err.lower() or "input-tokens" in captured.err.lower()
        assert not mock_stream.called
        assert not mock_load.called
        assert not output_file.exists()


def test_run_one_max_output_tokens_zero_rejected_before_request(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with patch("llm_meter.cli.stream_completion") as mock_stream, \
         patch("llm_meter.cli.load_tokenizer") as mock_load:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "run-one",
                "--endpoint", "http://localhost:8000/v1",
                "--model", "test-model",
                "--tokenizer", "fake",
                "--input-tokens", "50",
                "--max-output-tokens", "0",
                "--output", str(output_file),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "positive" in captured.err.lower() or "max-output-tokens" in captured.err.lower()
        assert not mock_stream.called
        assert not mock_load.called
        assert not output_file.exists()


def test_run_batch_help(capsys: object) -> None:
    try:
        main(["run-batch", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "run-batch" in captured.out
    assert "Legacy experimental" in captured.out
    assert "--endpoint" in captured.out
    assert "--model" in captured.out
    assert "--warmup-requests" in captured.out
    assert "--requests" in captured.out
    assert "--concurrency" in captured.out


def test_run_batch_builtin_produces_session(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "session.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run-batch",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--tokenizer", "fake",
            "--input-tokens", "50",
            "--max-output-tokens", "64",
            "--warmup-requests", "2",
            "--requests", "4",
            "--concurrency", "2",
            "--seed", "42",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert data["schema_version"] == "1"
    assert data["status"] == "completed"
    assert len(data["requests"]) == 6
    warmup = [r for r in data["requests"] if r["phase"] == "warmup"]
    measured = [r for r in data["requests"] if r["phase"] == "measured"]
    assert len(warmup) == 2
    assert len(measured) == 4
    assert data["summary"]["phase"] == "measured"
    assert data["summary"]["attempts"] == {
        "attempted": 4,
        "completed": 4,
        "failed": 0,
        "success_rate": 1.0,
        "error_rate": 0.0,
        "errors_by_category": {},
    }

    captured = capsys.readouterr()
    assert "Measured phase" in captured.out
    assert "attempted:        4" in captured.out
    assert "TTFT p50:" in captured.out
    assert "request throughput:" in captured.out
    assert "output throughput:" in captured.out
    assert "server_reported" in captured.out


def test_run_batch_manual_produces_session(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "manual_session.json"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run-batch",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello world",
            "--max-output-tokens", "64",
            "--warmup-requests", "1",
            "--requests", "3",
            "--concurrency", "1",
            "--output", str(output_file),
        ])

    assert exit_code == 0
    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert len(data["requests"]) == 4


def test_run_batch_requests_zero_rejected_before_tokenizer(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with patch("llm_meter.cli.stream_completion") as mock_stream, \
         patch("llm_meter.cli.load_tokenizer") as mock_load:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "run-batch",
                "--endpoint", "http://localhost:8000/v1",
                "--model", "test-model",
                "--tokenizer", "fake",
                "--input-tokens", "50",
                "--max-output-tokens", "64",
                "--requests", "0",
                "--output", str(output_file),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "requests" in captured.err.lower()
        assert not mock_stream.called
        assert not mock_load.called
        assert not output_file.exists()


def test_run_batch_concurrency_zero_rejected_before_tokenizer(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with patch("llm_meter.cli.stream_completion") as mock_stream, \
         patch("llm_meter.cli.load_tokenizer") as mock_load:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "run-batch",
                "--endpoint", "http://localhost:8000/v1",
                "--model", "test-model",
                "--tokenizer", "fake",
                "--input-tokens", "50",
                "--max-output-tokens", "64",
                "--requests", "4",
                "--concurrency", "0",
                "--output", str(output_file),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "concurrency" in captured.err.lower()
        assert not mock_stream.called
        assert not mock_load.called
        assert not output_file.exists()


def test_run_batch_negative_warmup_rejected_before_tokenizer(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with patch("llm_meter.cli.stream_completion") as mock_stream, \
         patch("llm_meter.cli.load_tokenizer") as mock_load:
        with pytest.raises(SystemExit) as exc_info:
            main([
                "run-batch",
                "--endpoint", "http://localhost:8000/v1",
                "--model", "test-model",
                "--tokenizer", "fake",
                "--input-tokens", "50",
                "--max-output-tokens", "64",
                "--warmup-requests", "-1",
                "--requests", "4",
                "--output", str(output_file),
            ])
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "warmup" in captured.err.lower()
        assert not mock_stream.called
        assert not mock_load.called
        assert not output_file.exists()


def test_run_batch_prompt_xor_input_tokens_rejected(
    tmp_path: Path, capsys: object
) -> None:
    output_file = tmp_path / "should_not_exist.json"
    with pytest.raises(SystemExit) as exc_info:
        main([
            "run-batch",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hi",
            "--input-tokens", "50",
            "--tokenizer", "fake",
            "--max-output-tokens", "64",
            "--requests", "4",
            "--output", str(output_file),
        ])
    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "mutually exclusive" in captured.err


def test_run_help(capsys: object) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--endpoint",
        "--model",
        "--prompt",
        "--input-tokens",
        "--requests",
        "--concurrency",
        "--output-dir",
        "--csv",
    ):
        assert option in output


def test_run_manual_writes_only_canonical_json_without_csv(
    tmp_path: Path, capsys: object
) -> None:
    output_dir = tmp_path / "manual"
    output_dir.mkdir()
    (output_dir / "notes.txt").write_text("keep me\n")

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello",
            "--requests", "2",
            "--output-dir", str(output_dir),
        ])

    assert exit_code == 0
    assert sorted(path.name for path in output_dir.iterdir()) == ["notes.txt", "session.json"]
    data = json.loads((output_dir / "session.json").read_text())
    assert data["configuration"]["prompt_source"] == "manual"
    assert data["configuration"]["warmup_requests"] == 0
    assert data["configuration"]["concurrency"] == 1
    assert len(data["requests"]) == 2
    output = capsys.readouterr().out
    assert "Measured phase" in output
    assert str(output_dir / "session.json") in output


def test_run_generated_writes_optional_csv_views(tmp_path: Path) -> None:
    output_dir = tmp_path / "generated"

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        exit_code = main([
            "run",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--tokenizer", "fake",
            "--input-tokens", "50",
            "--max-output-tokens", "64",
            "--warmup-requests", "1",
            "--requests", "2",
            "--concurrency", "2",
            "--seed", "42",
            "--output-dir", str(output_dir),
            "--csv",
        ])

    assert exit_code == 0
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "requests.csv",
        "session.json",
        "summary.csv",
    ]
    data = json.loads((output_dir / "session.json").read_text())
    assert data["configuration"]["prompt_source"] == "builtin"
    assert data["configuration"]["input_tokens_target"] == 50
    assert data["configuration"]["output_tokens_target"] == 64
    assert data["environment"]["version"] == "1"
    assert "stream_events" in data["requests"][0]["run"]
    request_header = (output_dir / "requests.csv").read_text().splitlines()[0]
    assert "environment" not in request_header
    assert "stream_events" not in request_header


def test_run_default_output_directory_uses_session_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch("llm_meter.cli.stream_completion", return_value=_fake_observations()):
        assert main([
            "run",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello",
            "--requests", "1",
        ]) == 0

    session_files = list((tmp_path / "llm-meter-runs").glob("*/session.json"))
    assert len(session_files) == 1
    session = json.loads(session_files[0].read_text())
    assert session_files[0].parent.name == session["session_id"]


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "--prompt", "hello",
            "--input-tokens", "5",
            "--tokenizer", "fake",
            "--max-output-tokens", "4",
        ],
        ["--input-tokens", "5", "--max-output-tokens", "4"],
        ["--input-tokens", "5", "--tokenizer", "fake"],
        ["--input-tokens", "0", "--tokenizer", "fake", "--max-output-tokens", "4"],
        ["--prompt", "hello", "--max-output-tokens", "0"],
        ["--prompt", "hello", "--requests", "0"],
        ["--prompt", "hello", "--concurrency", "0"],
        ["--prompt", "hello", "--warmup-requests", "-1"],
        [],
    ],
)
def test_run_validation_precedes_side_effects(
    arguments: list[str], tmp_path: Path
) -> None:
    output_dir = tmp_path / "must-not-exist"
    base = [
        "run",
        "--endpoint", "http://localhost:8000/v1",
        "--model", "test-model",
        "--requests", "1",
        "--output-dir", str(output_dir),
    ]

    with (
        patch("llm_meter.cli.load_tokenizer") as tokenizer,
        patch("llm_meter.cli.run_session") as runner,
        patch("llm_meter.cli.stream_completion") as network,
        pytest.raises(SystemExit),
    ):
        main(base + arguments)

    tokenizer.assert_not_called()
    runner.assert_not_called()
    network.assert_not_called()
    assert not output_dir.exists()


@pytest.mark.parametrize("reserved_name", ["session.json", "requests.csv", "summary.csv"])
@pytest.mark.parametrize("use_csv", [False, True])
def test_run_refuses_reserved_artifacts_before_execution(
    reserved_name: str, use_csv: bool, tmp_path: Path
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    artifact = output_dir / reserved_name
    artifact.write_text("original\n")

    arguments = [
        "run",
        "--endpoint", "http://localhost:8000/v1",
        "--model", "test-model",
        "--prompt", "hello",
        "--requests", "1",
        "--output-dir", str(output_dir),
    ]
    if use_csv:
        arguments.append("--csv")

    with (
        patch("llm_meter.cli.load_tokenizer") as tokenizer,
        patch("llm_meter.cli.run_session") as runner,
        patch("llm_meter.cli.stream_completion") as network,
        pytest.raises(SystemExit),
    ):
        main([
            *arguments,
        ])

    assert artifact.read_text() == "original\n"
    tokenizer.assert_not_called()
    runner.assert_not_called()
    network.assert_not_called()


def test_run_runner_failure_writes_no_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "failed"

    with patch("llm_meter.cli.run_session", side_effect=RuntimeError("runner failed")):
        with pytest.raises(RuntimeError, match="runner failed"):
            main([
                "run",
                "--endpoint", "http://localhost:8000/v1",
                "--model", "test-model",
                "--prompt", "hello",
                "--requests", "1",
                "--output-dir", str(output_dir),
                "--csv",
            ])

    assert not output_dir.exists()


def test_run_request_failure_still_writes_artifact_and_exits_zero(tmp_path: Path) -> None:
    from llm_meter.models import ErrorObservation

    failed = RawObservations(
        request_start=RequestStart(offset_ns=0, wall_clock_utc="2025-01-01T00:00:00Z"),
        stream_events=[],
        error=ErrorObservation(offset_ns=10, category="http_error", status=503),
    )
    output_dir = tmp_path / "request-failed"

    with patch("llm_meter.cli.stream_completion", return_value=failed):
        exit_code = main([
            "run",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello",
            "--requests", "1",
            "--output-dir", str(output_dir),
        ])

    assert exit_code == 0
    data = json.loads((output_dir / "session.json").read_text())
    assert data["status"] == "completed"
    assert data["requests"][0]["run"]["run_status"] == "failed"
    assert data["summary"]["attempts"]["failed"] == 1


def test_run_summary_csv_failure_removes_current_csv_exports(tmp_path: Path) -> None:
    output_dir = tmp_path / "csv-failed"

    with (
        patch("llm_meter.cli.stream_completion", return_value=_fake_observations()),
        patch("llm_meter.cli.write_summary_csv", side_effect=OSError("disk error")),
        pytest.raises(OSError, match="disk error"),
    ):
        main([
            "run",
            "--endpoint", "http://localhost:8000/v1",
            "--model", "test-model",
            "--prompt", "hello",
            "--requests", "1",
            "--output-dir", str(output_dir),
            "--csv",
        ])

    data = json.loads((output_dir / "session.json").read_text())
    assert data["status"] == "completed"
    assert not (output_dir / "requests.csv").exists()
    assert not (output_dir / "summary.csv").exists()
