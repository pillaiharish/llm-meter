from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm_meter.export import (
    CSV_SCHEMA_VERSION,
    REQUEST_COLUMNS,
    SUMMARY_COLUMNS,
    write_requests_csv,
    write_summary_csv,
)
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
    WorkloadProvenance,
)
from llm_meter.summary import summarize_session


def _run(run_id: str, *, failed: str | None = None) -> BenchmarkRun:
    return BenchmarkRun(
        schema_version="1",
        run_id=run_id,
        started_at="2025-01-01T00:00:00+00:00",
        run_status="failed" if failed else "completed",
        configuration=RunConfiguration(
            "https://user:secret@example.test/v1?api_key=secret",
            'mødel, "quoted"',
            True,
            64,
        ),
        provenance=Provenance("test"),
        workload=WorkloadProvenance(
            source="manual",
            seed=0,
            input_tokens_target=None,
            output_tokens_target=64,
            input_tokens_actual_local=None,
            prompt_sha256="abc123",
            prompt_chars=14,
        ),
        error=(
            ErrorObservation(offset_ns=3, category=failed, status=503)
            if failed
            else None
        ),
        usage=Usage(
            input_tokens=None if failed else 5,
            output_tokens=None if failed else 10,
            source=TokenCountSource.UNKNOWN if failed else TokenCountSource.SERVER_REPORTED,
        ),
        metrics=ClientMetrics(
            client_ttft_ns=None if failed else 10,
            e2e_latency_ns=None if failed else 100,
            inter_chunk_latencies_ns=[] if failed else [20, 30],
            tpot_ns=None if failed else 10,
            tpot_status="no_token_count" if failed else "ok",
        ),
    )


def _session() -> BenchmarkSession:
    requests = [
        SessionRequest("warmup", 0, 0, 50, _run("warmup")),
        SessionRequest("measured", 0, 100, 200, _run("measured-ok")),
        SessionRequest("measured", 1, 150, 250, _run("measured-z", failed="zeta,error")),
        SessionRequest("measured", 2, 200, 300, _run("measured-a", failed="alpha")),
    ]
    session = BenchmarkSession(
        schema_version="1",
        session_id="session-id",
        started_at="2025-01-01T00:00:00+00:00",
        completed_at="2025-01-01T00:00:01+00:00",
        status="completed",
        configuration=SessionConfiguration(
            endpoint="https://example.test/v1",
            model="model",
            warmup_requests=1,
            measured_requests=3,
            concurrency=2,
            seed=0,
            seed_strategy="base_plus_global_ordinal",
            max_connections=2,
            max_keepalive_connections=2,
            prompt_source="manual",
            input_tokens_target=None,
            output_tokens_target=64,
            tokenizer_id=None,
        ),
        requests=requests,
        provenance=Provenance("test"),
    )
    session.summary = summarize_session(session)
    return session


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_requests_csv_is_stable_complete_and_private(tmp_path: Path) -> None:
    session = _session()
    path = write_requests_csv(session, tmp_path / "requests.csv")
    second = write_requests_csv(session, tmp_path / "requests-copy.csv")
    rows = _read_csv(path)

    assert tuple(rows[0]) == REQUEST_COLUMNS
    assert path.read_bytes() == second.read_bytes()
    assert [row["run_id"] for row in rows] == [
        "warmup",
        "measured-ok",
        "measured-z",
        "measured-a",
    ]
    assert [row["phase"] for row in rows] == ["warmup", "measured", "measured", "measured"]
    assert [row["ordinal"] for row in rows] == ["0", "0", "1", "2"]
    assert all(row["csv_schema_version"] == CSV_SCHEMA_VERSION for row in rows)
    assert rows[1]["run_status"] == "completed"
    assert rows[1]["prompt_sha256"] == "abc123"
    assert rows[1]["token_count_source"] == "server_reported"
    assert rows[1]["inter_chunk_count"] == "2"
    assert rows[2]["run_status"] == "failed"
    assert rows[2]["error_category"] == "zeta,error"
    assert rows[2]["error_status"] == "503"
    assert rows[2]["usage_output_tokens"] == ""
    assert rows[2]["client_ttft_ns"] == ""
    assert rows[0]["endpoint"] == "https://example.test/v1?api_key=***REDACTED***"
    assert "user" not in path.read_text()
    assert "secret" not in path.read_text()
    assert "prompt" not in rows[0]
    assert '"mødel, ""quoted"""' in path.read_text()


def test_summary_csv_is_long_form_measured_only_and_deterministic(tmp_path: Path) -> None:
    session = _session()
    first = write_summary_csv(session, tmp_path / "summary-a.csv")
    second = write_summary_csv(session, tmp_path / "summary-b.csv")
    rows = _read_csv(first)
    keyed = {(row["section"], row["metric"]): row for row in rows}

    assert tuple(rows[0]) == SUMMARY_COLUMNS
    assert first.read_bytes() == second.read_bytes()
    assert keyed[("attempts", "attempted")]["value"] == "3"
    assert keyed[("attempts", "completed")]["value"] == "1"
    assert keyed[("attempts", "error_rate")]["value"] == str(2 / 3)
    assert keyed[("ttft", "p50")]["value"] == "10.0"
    assert keyed[("ttft", "p95")]["unit"] == "ns"
    assert keyed[("ttft", "p99")]["sample_count"] == "1"
    assert keyed[("throughput", "attempted_requests_per_s")]["unit"] == "requests/s"
    assert keyed[("throughput", "attempted_requests_per_s")]["value"] == "15000000.0"
    assert keyed[("throughput", "output_tokens_per_s")]["status"] == "ok"
    assert keyed[("throughput", "output_tokens_per_s")]["value"] == "50000000.0"
    assert keyed[("throughput", "output_token_source")]["value"] == "server_reported"
    error_metrics = [row["metric"] for row in rows if row["metric"].startswith("errors.")]
    assert error_metrics == ["errors.alpha", "errors.zeta,error"]


def test_summary_csv_uses_empty_value_for_unavailable_token_throughput(tmp_path: Path) -> None:
    session = _session()
    for request in session.measured_runs:
        request.run.usage.output_tokens = None
    session.summary = summarize_session(session)

    rows = _read_csv(write_summary_csv(session, tmp_path / "summary.csv"))
    token_rate = next(row for row in rows if row["metric"] == "output_tokens_per_s")

    assert token_rate["value"] == ""
    assert token_rate["status"] == "missing_output_token_counts"


def test_csv_writers_refuse_overwrite_and_missing_summary(tmp_path: Path) -> None:
    session = _session()
    path = write_requests_csv(session, tmp_path / "requests.csv")
    with pytest.raises(FileExistsError):
        write_requests_csv(session, path)

    session.summary = None
    with pytest.raises(ValueError, match="no summary"):
        write_summary_csv(session, tmp_path / "summary.csv")
