from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

from llm_meter import __version__
from llm_meter.metrics import derive_metrics
from llm_meter.models import (
    BenchmarkRun,
    BenchmarkSession,
    ClientMetrics,
    Completion,
    ErrorObservation,
    Provenance,
    RawObservations,
    RequestStart,
    ResponseEstablished,
    RunConfiguration,
    RunStatus,
    SessionConfiguration,
    SessionRequest,
    StreamEvent,
    TokenCountSource,
    Usage,
    WorkloadProvenance,
    dataclass_to_dict,
)

SCHEMA_VERSION = "1"
SESSION_SCHEMA_VERSION = "1"

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
        def _quote(
            s: str,
            safe: str = "",
            encoding: str | None = None,
            errors: str | None = None,
        ) -> str:
            if s == "***REDACTED***":
                return s
            return quote(s, safe, encoding, errors)

        query = urlencode(
            redacted_params,
            doseq=True,
            quote_via=_quote,
        )
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


_URL_PATTERN = re.compile(r"https?://\S+")

_BEARER_PATTERN = re.compile(
    r"(Authorization\s*:\s*)([^\s]+(?:\s+[^\s]+)*)",
    re.IGNORECASE,
)


def sanitize_text(text: str) -> str:
    def _replace_url(match: re.Match[str]) -> str:
        return sanitize_endpoint(match.group(0))

    def _replace_bearer(match: re.Match[str]) -> str:
        return f"{match.group(1)}***REDACTED***"

    text = _URL_PATTERN.sub(_replace_url, text)
    return _BEARER_PATTERN.sub(_replace_bearer, text)


def build_run(
    *,
    run_id: str,
    started_at: str,
    configuration: RunConfiguration,
    observations: RawObservations,
    workload: WorkloadProvenance | None = None,
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

    sanitized_error = None
    if observations.error is not None:
        sanitized_error = ErrorObservation(
            offset_ns=observations.error.offset_ns,
            category=observations.error.category,
            status=observations.error.status,
            exception_type=observations.error.exception_type,
            message=sanitize_text(observations.error.message),
        )

    return BenchmarkRun(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        started_at=started_at,
        run_status=run_status,
        configuration=sanitized_config,
        provenance=provenance,
        workload=workload,
        request_start=observations.request_start,
        response_established=observations.response_established,
        stream_events=observations.stream_events,
        completion=observations.completion,
        error=sanitized_error,
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

    workload = None
    if obj.get("workload"):
        w = obj["workload"]
        workload = WorkloadProvenance(
            source=w["source"],
            seed=w["seed"],
            input_tokens_target=w.get("input_tokens_target"),
            output_tokens_target=w.get("output_tokens_target"),
            input_tokens_actual_local=w.get("input_tokens_actual_local"),
            resolution_status=w.get("resolution_status", "not_applicable"),
            prompt_sha256=w.get("prompt_sha256", ""),
            prompt_chars=w.get("prompt_chars", 0),
            tokenizer_provider=w.get("tokenizer_provider"),
            tokenizer_id=w.get("tokenizer_id"),
            tokenizer_revision=w.get("tokenizer_revision"),
        )

    return BenchmarkRun(
        schema_version=obj["schema_version"],
        run_id=obj["run_id"],
        started_at=obj["started_at"],
        run_status=obj.get("run_status", RunStatus.FAILED.value),
        configuration=configuration,
        provenance=provenance,
        workload=workload,
        request_start=request_start,
        response_established=response_established,
        stream_events=stream_events,
        completion=completion,
        error=error,
        usage=usage,
        metrics=metrics,
    )


def session_to_json(session: BenchmarkSession) -> str:
    data: dict[str, Any] = {
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "status": session.status,
        "configuration": {
            "endpoint": sanitize_endpoint(session.configuration.endpoint),
            "model": session.configuration.model,
            "warmup_requests": session.configuration.warmup_requests,
            "measured_requests": session.configuration.measured_requests,
            "concurrency": session.configuration.concurrency,
            "seed": session.configuration.seed,
            "seed_strategy": session.configuration.seed_strategy,
            "max_connections": session.configuration.max_connections,
            "max_keepalive_connections": session.configuration.max_keepalive_connections,
            "prompt_source": session.configuration.prompt_source,
            "input_tokens_target": session.configuration.input_tokens_target,
            "output_tokens_target": session.configuration.output_tokens_target,
            "tokenizer_id": session.configuration.tokenizer_id,
            "max_output_tokens": session.configuration.max_output_tokens,
        },
        "requests": [
            {
                "phase": req.phase,
                "ordinal": req.ordinal,
                "session_start_offset_ns": req.session_start_offset_ns,
                "session_finish_offset_ns": req.session_finish_offset_ns,
                "run": json.loads(to_json(req.run)),
            }
            for req in session.requests
        ],
        "provenance": {"llm_meter_version": session.provenance.llm_meter_version},
    }
    return json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)


def write_session(session: BenchmarkSession, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(session_to_json(session) + "\n", encoding="utf-8")
    return output_path


def session_from_json(data: str) -> BenchmarkSession:
    obj: dict[str, Any] = json.loads(data)
    encountered = obj.get("schema_version")
    if encountered is None:
        raise UnsupportedSchemaVersion(
            f"missing schema_version; expected {SESSION_SCHEMA_VERSION!r}"
        )
    if encountered != SESSION_SCHEMA_VERSION:
        raise UnsupportedSchemaVersion(
            f"unsupported session schema version: expected {SESSION_SCHEMA_VERSION!r}, "
            f"got {encountered!r}"
        )
    return _dict_to_session(obj)


def _dict_to_session(obj: dict[str, Any]) -> BenchmarkSession:
    config_data = obj["configuration"]
    configuration = SessionConfiguration(
        endpoint=config_data["endpoint"],
        model=config_data["model"],
        warmup_requests=config_data["warmup_requests"],
        measured_requests=config_data["measured_requests"],
        concurrency=config_data["concurrency"],
        seed=config_data["seed"],
        seed_strategy=config_data["seed_strategy"],
        max_connections=config_data["max_connections"],
        max_keepalive_connections=config_data["max_keepalive_connections"],
        prompt_source=config_data["prompt_source"],
        input_tokens_target=config_data.get("input_tokens_target"),
        output_tokens_target=config_data.get("output_tokens_target"),
        tokenizer_id=config_data.get("tokenizer_id"),
        max_output_tokens=config_data.get("max_output_tokens"),
    )

    requests: list[SessionRequest] = []
    for req_data in obj.get("requests", []):
        run_json = json.dumps(req_data["run"])
        run = from_json(run_json)
        requests.append(
            SessionRequest(
                phase=req_data["phase"],
                ordinal=req_data["ordinal"],
                session_start_offset_ns=req_data["session_start_offset_ns"],
                session_finish_offset_ns=req_data["session_finish_offset_ns"],
                run=run,
            )
        )

    provenance = Provenance(
        llm_meter_version=obj.get("provenance", {}).get("llm_meter_version", ""),
    )

    return BenchmarkSession(
        schema_version=obj["schema_version"],
        session_id=obj["session_id"],
        started_at=obj["started_at"],
        completed_at=obj["completed_at"],
        status=obj.get("status", "completed"),
        configuration=configuration,
        requests=requests,
        provenance=provenance,
    )
