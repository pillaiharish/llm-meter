from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse, urlunparse

from llm_meter import __version__
from llm_meter.metrics import derive_metrics
from llm_meter.models import (
    BenchmarkRun,
    ClientMetrics,
    Completion,
    ErrorObservation,
    Provenance,
    RawObservations,
    RequestStart,
    ResponseEstablished,
    RunConfiguration,
    RunStatus,
    StreamEvent,
    TokenCountSource,
    Usage,
    dataclass_to_dict,
)

SCHEMA_VERSION = "1"

_SENSITIVE_QUERY_PARAMS = frozenset({
    "api_key",
    "key",
    "token",
    "access_token",
    "authorization",
    "signature",
    "sig",
})


class UnsupportedSchemaVersion(Exception):
    pass


def sanitize_endpoint(url: str) -> str:
    parsed = urlparse(url)

    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    if parsed.query:
        params = parse_qsl(parsed.query, keep_blank_values=True)
        redacted_params = [
            (k, "***REDACTED***" if k.lower() in _SENSITIVE_QUERY_PARAMS else v)
            for k, v in params
        ]
        query = "&".join(f"{k}={v}" for k, v in redacted_params)
    else:
        query = ""

    return urlunparse((
        parsed.scheme,
        netloc,
        parsed.path,
        parsed.params,
        query,
        parsed.fragment,
    ))


def build_run(
    *,
    run_id: str,
    started_at: str,
    configuration: RunConfiguration,
    observations: RawObservations,
) -> BenchmarkRun:
    metrics = derive_metrics(observations, observations.usage)
    provenance = Provenance(llm_meter_version=__version__)

    if observations.completion is not None and observations.error is None:
        run_status = RunStatus.COMPLETED.value
    else:
        run_status = RunStatus.FAILED.value

    sanitized_config = RunConfiguration(
        endpoint=sanitize_endpoint(configuration.endpoint),
        model=configuration.model,
        streaming=configuration.streaming,
        max_output_tokens=configuration.max_output_tokens,
    )

    return BenchmarkRun(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        started_at=started_at,
        run_status=run_status,
        configuration=sanitized_config,
        provenance=provenance,
        request_start=observations.request_start,
        response_established=observations.response_established,
        stream_events=observations.stream_events,
        completion=observations.completion,
        error=observations.error,
        usage=observations.usage,
        metrics=metrics,
    )


def to_json(run: BenchmarkRun) -> str:
    data = dataclass_to_dict(run)
    data["schema_version"] = run.schema_version
    data["run_status"] = run.run_status
    data["usage"]["source"] = run.usage.source.value
    data["metrics"]["tpot_status"] = run.metrics.tpot_status
    return json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)


def write_artifact(run: BenchmarkRun, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_json(run) + "\n", encoding="utf-8")
    return output_path


def from_json(data: str) -> BenchmarkRun:
    obj: dict[str, Any] = json.loads(data)
    encountered = obj.get("schema_version")
    if encountered is None:
        raise UnsupportedSchemaVersion(
            f"missing schema_version; expected {SCHEMA_VERSION!r}"
        )
    if encountered != SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported schema version: expected {SCHEMA_VERSION!r}, "
            f"got {encountered!r}"
        )
    return _dict_to_run(obj)


def _dict_to_run(obj: dict[str, Any]) -> BenchmarkRun:
    config_data = obj["configuration"]
    configuration = RunConfiguration(
        endpoint=config_data["endpoint"],
        model=config_data["model"],
        streaming=config_data["streaming"],
        max_output_tokens=config_data.get("max_output_tokens"),
    )

    request_start = None
    if obj.get("request_start"):
        rs = obj["request_start"]
        request_start = RequestStart(
            offset_ns=rs["offset_ns"],
            wall_clock_utc=rs["wall_clock_utc"],
        )

    response_established = None
    if obj.get("response_established"):
        re_data = obj["response_established"]
        response_established = ResponseEstablished(
            offset_ns=re_data["offset_ns"],
            status_code=re_data["status_code"],
            content_type=re_data.get("content_type"),
        )

    stream_events = [
        StreamEvent(
            sequence=se["sequence"],
            offset_ns=se["offset_ns"],
            event_type=se["event_type"],
            text_delta=se.get("text_delta"),
            finish_reason=se.get("finish_reason"),
            usage=se.get("usage"),
        )
        for se in obj.get("stream_events", [])
    ]

    completion = None
    if obj.get("completion"):
        c = obj["completion"]
        completion = Completion(
            offset_ns=c["offset_ns"],
            wall_clock_utc=c["wall_clock_utc"],
        )

    error = None
    if obj.get("error"):
        e = obj["error"]
        error = ErrorObservation(
            offset_ns=e["offset_ns"],
            category=e["category"],
            status=e.get("status"),
            exception_type=e.get("exception_type"),
            message=e.get("message", ""),
        )

    usage_data = obj.get("usage", {})
    usage = Usage(
        input_tokens=usage_data.get("input_tokens"),
        output_tokens=usage_data.get("output_tokens"),
        source=TokenCountSource(usage_data.get("source", "unknown")),
    )

    metrics_data = obj.get("metrics", {})
    metrics = ClientMetrics(
        client_ttft_ns=metrics_data.get("client_ttft_ns"),
        e2e_latency_ns=metrics_data.get("e2e_latency_ns"),
        inter_chunk_latencies_ns=metrics_data.get("inter_chunk_latencies_ns", []),
        tpot_ns=metrics_data.get("tpot_ns"),
        tpot_status=metrics_data.get("tpot_status", "no_token_count"),
    )

    provenance = Provenance(
        llm_meter_version=obj.get("provenance", {}).get("llm_meter_version", ""),
    )

    return BenchmarkRun(
        schema_version=obj["schema_version"],
        run_id=obj["run_id"],
        started_at=obj["started_at"],
        run_status=obj.get("run_status", RunStatus.FAILED.value),
        configuration=configuration,
        provenance=provenance,
        request_start=request_start,
        response_established=response_established,
        stream_events=stream_events,
        completion=completion,
        error=error,
        usage=usage,
        metrics=metrics,
    )
