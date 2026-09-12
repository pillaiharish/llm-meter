from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from llm_meter.models import (
    AttemptSummary,
    BenchmarkSession,
    DistributionSummary,
    OutputTokenStatus,
    RunStatus,
    SessionSummary,
    ThroughputSummary,
    TokenCountSource,
    TpotStatus,
)

SUMMARY_VERSION = "1"
PERCENTILE_METHOD = "linear_type7"


def linear_percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _distribution(
    values: Sequence[int | float], sample_unit: str
) -> DistributionSummary:
    if not values:
        return DistributionSummary(
            sample_count=0,
            sample_unit=sample_unit,
            unit="ns",
            minimum=None,
            maximum=None,
            p50=None,
            p90=None,
            p95=None,
            p99=None,
        )

    return DistributionSummary(
        sample_count=len(values),
        sample_unit=sample_unit,
        unit="ns",
        minimum=float(min(values)),
        maximum=float(max(values)),
        p50=linear_percentile(values, 0.50),
        p90=linear_percentile(values, 0.90),
        p95=linear_percentile(values, 0.95),
        p99=linear_percentile(values, 0.99),
    )


def _token_throughput(completed_runs: list, duration_ns: int | None) -> tuple:
    if not completed_runs:
        return (None, None, None, OutputTokenStatus.NO_SUCCESSFUL_REQUESTS.value)

    if any(run.usage.output_tokens is None for run in completed_runs):
        return (None, None, None, OutputTokenStatus.MISSING_OUTPUT_TOKEN_COUNTS.value)

    sources = {run.usage.source.value for run in completed_runs}
    if TokenCountSource.UNKNOWN.value in sources:
        return (None, None, None, OutputTokenStatus.UNKNOWN_TOKEN_COUNT_SOURCE.value)
    if len(sources) != 1:
        return (None, None, None, OutputTokenStatus.MIXED_TOKEN_COUNT_SOURCES.value)
    if duration_ns is None or duration_ns <= 0:
        return (None, None, None, OutputTokenStatus.NON_POSITIVE_MEASURED_DURATION.value)

    total = sum(run.usage.output_tokens for run in completed_runs)
    source = sources.pop()
    return (
        total,
        total * 1_000_000_000 / duration_ns,
        source,
        OutputTokenStatus.OK.value,
    )


def summarize_session(session: BenchmarkSession) -> SessionSummary:
    measured = session.measured_runs
    completed_runs = [
        request.run
        for request in measured
        if request.run.run_status == RunStatus.COMPLETED.value
    ]
    failed_runs = [
        request.run
        for request in measured
        if request.run.run_status == RunStatus.FAILED.value
    ]

    attempted = len(measured)
    completed = len(completed_runs)
    failed = len(failed_runs)
    attempts = AttemptSummary(
        attempted=attempted,
        completed=completed,
        failed=failed,
        success_rate=completed / attempted if attempted else 0.0,
        error_rate=failed / attempted if attempted else 0.0,
        errors_by_category=dict(Counter(
            run.error.category if run.error is not None else "unknown"
            for run in failed_runs
        )),
    )

    ttft = [
        run.metrics.client_ttft_ns
        for run in completed_runs
        if run.metrics.client_ttft_ns is not None
    ]
    e2e = [
        run.metrics.e2e_latency_ns
        for run in completed_runs
        if run.metrics.e2e_latency_ns is not None
    ]
    tpot = [
        run.metrics.tpot_ns
        for run in completed_runs
        if run.metrics.tpot_status == TpotStatus.OK.value
        and run.metrics.tpot_ns is not None
    ]
    inter_chunk = [
        interval
        for run in completed_runs
        for interval in run.metrics.inter_chunk_latencies_ns
    ]

    duration_ns = None
    if measured:
        duration_ns = (
            max(request.session_finish_offset_ns for request in measured)
            - min(request.session_start_offset_ns for request in measured)
        )

    attempted_per_s = None
    completed_per_s = None
    if duration_ns is not None and duration_ns > 0:
        attempted_per_s = attempted * 1_000_000_000 / duration_ns
        completed_per_s = completed * 1_000_000_000 / duration_ns

    token_total, tokens_per_s, token_source, token_status = _token_throughput(
        completed_runs, duration_ns
    )

    return SessionSummary(
        summary_version=SUMMARY_VERSION,
        phase="measured",
        percentile_method=PERCENTILE_METHOD,
        attempts=attempts,
        ttft=_distribution(ttft, "request"),
        e2e=_distribution(e2e, "request"),
        tpot=_distribution(tpot, "request"),
        inter_chunk=_distribution(inter_chunk, "chunk_interval"),
        throughput=ThroughputSummary(
            measured_duration_ns=duration_ns,
            attempted_requests_per_s=attempted_per_s,
            completed_requests_per_s=completed_per_s,
            output_tokens_total=token_total,
            output_tokens_per_s=tokens_per_s,
            output_token_source=token_source,
            output_token_status=token_status,
        ),
    )
