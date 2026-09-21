from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from llm_meter.artifact import sanitize_endpoint
from llm_meter.models import BenchmarkSession, DistributionSummary

CSV_SCHEMA_VERSION = "1"

REQUEST_COLUMNS = (
    "csv_schema_version",
    "session_id",
    "phase",
    "ordinal",
    "run_id",
    "run_status",
    "model",
    "endpoint",
    "prompt_sha256",
    "input_tokens_target",
    "input_tokens_actual_local",
    "output_tokens_target",
    "usage_input_tokens",
    "usage_output_tokens",
    "token_count_source",
    "client_ttft_ns",
    "e2e_latency_ns",
    "tpot_ns",
    "tpot_status",
    "inter_chunk_count",
    "session_start_offset_ns",
    "session_finish_offset_ns",
    "error_category",
    "error_status",
)

SUMMARY_COLUMNS = (
    "csv_schema_version",
    "session_id",
    "section",
    "metric",
    "value",
    "unit",
    "status",
    "sample_count",
    "sample_unit",
)


def _write_rows(path: str | Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_requests_csv(session: BenchmarkSession, path: str | Path) -> Path:
    rows: list[dict[str, Any]] = []
    for request in session.requests:
        run = request.run
        workload = run.workload
        error = run.error
        rows.append({
            "csv_schema_version": CSV_SCHEMA_VERSION,
            "session_id": session.session_id,
            "phase": request.phase,
            "ordinal": request.ordinal,
            "run_id": run.run_id,
            "run_status": run.run_status,
            "model": run.configuration.model,
            "endpoint": sanitize_endpoint(run.configuration.endpoint),
            "prompt_sha256": workload.prompt_sha256 if workload else None,
            "input_tokens_target": workload.input_tokens_target if workload else None,
            "input_tokens_actual_local": (
                workload.input_tokens_actual_local if workload else None
            ),
            "output_tokens_target": workload.output_tokens_target if workload else None,
            "usage_input_tokens": run.usage.input_tokens,
            "usage_output_tokens": run.usage.output_tokens,
            "token_count_source": run.usage.source.value,
            "client_ttft_ns": run.metrics.client_ttft_ns,
            "e2e_latency_ns": run.metrics.e2e_latency_ns,
            "tpot_ns": run.metrics.tpot_ns,
            "tpot_status": run.metrics.tpot_status,
            "inter_chunk_count": len(run.metrics.inter_chunk_latencies_ns),
            "session_start_offset_ns": request.session_start_offset_ns,
            "session_finish_offset_ns": request.session_finish_offset_ns,
            "error_category": error.category if error else None,
            "error_status": error.status if error else None,
        })
    return _write_rows(path, REQUEST_COLUMNS, rows)


def _summary_row(
    session_id: str,
    section: str,
    metric: str,
    value: Any,
    unit: str,
    *,
    status: str | None = None,
    sample_count: int | None = None,
    sample_unit: str | None = None,
) -> dict[str, Any]:
    return {
        "csv_schema_version": CSV_SCHEMA_VERSION,
        "session_id": session_id,
        "section": section,
        "metric": metric,
        "value": value,
        "unit": unit,
        "status": status,
        "sample_count": sample_count,
        "sample_unit": sample_unit,
    }


def _distribution_rows(
    session_id: str,
    section: str,
    distribution: DistributionSummary,
) -> list[dict[str, Any]]:
    return [
        _summary_row(
            session_id,
            section,
            metric,
            getattr(distribution, metric),
            distribution.unit,
            sample_count=distribution.sample_count,
            sample_unit=distribution.sample_unit,
        )
        for metric in ("minimum", "maximum", "p50", "p90", "p95", "p99")
    ]


def write_summary_csv(session: BenchmarkSession, path: str | Path) -> Path:
    summary = session.summary
    if summary is None:
        raise ValueError("session has no summary to export")

    session_id = session.session_id
    attempts = summary.attempts
    rows = [
        _summary_row(session_id, "attempts", "attempted", attempts.attempted, "count"),
        _summary_row(session_id, "attempts", "completed", attempts.completed, "count"),
        _summary_row(session_id, "attempts", "failed", attempts.failed, "count"),
        _summary_row(session_id, "attempts", "success_rate", attempts.success_rate, "fraction"),
        _summary_row(session_id, "attempts", "error_rate", attempts.error_rate, "fraction"),
    ]
    rows.extend(
        _summary_row(session_id, "attempts", f"errors.{category}", count, "count")
        for category, count in sorted(attempts.errors_by_category.items())
    )
    for section in ("ttft", "e2e", "tpot", "inter_chunk"):
        rows.extend(_distribution_rows(session_id, section, getattr(summary, section)))

    throughput = summary.throughput
    rows.extend([
        _summary_row(
            session_id,
            "throughput",
            "measured_duration_ns",
            throughput.measured_duration_ns,
            "ns",
        ),
        _summary_row(
            session_id,
            "throughput",
            "attempted_requests_per_s",
            throughput.attempted_requests_per_s,
            "requests/s",
        ),
        _summary_row(
            session_id,
            "throughput",
            "completed_requests_per_s",
            throughput.completed_requests_per_s,
            "requests/s",
        ),
        _summary_row(
            session_id,
            "throughput",
            "output_tokens_total",
            throughput.output_tokens_total,
            "tokens",
            status=throughput.output_token_status,
        ),
        _summary_row(
            session_id,
            "throughput",
            "output_tokens_per_s",
            throughput.output_tokens_per_s,
            "tokens/s",
            status=throughput.output_token_status,
        ),
        _summary_row(
            session_id,
            "throughput",
            "output_token_source",
            throughput.output_token_source,
            "",
            status=throughput.output_token_status,
        ),
    ])
    return _write_rows(path, SUMMARY_COLUMNS, rows)
