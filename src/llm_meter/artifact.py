from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_meter import __version__
from llm_meter.metrics import derive_metrics
from llm_meter.models import (
    BenchmarkRun,
    ClientMetrics,
    Provenance,
    RawObservations,
    RunConfiguration,
    Usage,
    dataclass_to_dict,
)

SCHEMA_VERSION = "1"


def build_run(
    *,
    run_id: str,
    started_at: str,
    configuration: RunConfiguration,
    observations: RawObservations,
) -> BenchmarkRun:
    metrics = derive_metrics(observations, observations.usage)
    provenance = Provenance(llm_meter_version=__version__)

    return BenchmarkRun(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        started_at=started_at,
        configuration=configuration,
        request_start=observations.request_start,
        stream_events=observations.stream_events,
        completion=observations.completion,
        error=observations.error,
        usage=observations.usage,
        metrics=metrics,
        provenance=provenance,
    )


def to_json(run: BenchmarkRun) -> str:
    data = dataclass_to_dict(run)
    data["schema_version"] = run.schema_version
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
    return _dict_to_run(obj)


def _dict_to_run(obj: dict[str, Any]) -> BenchmarkRun:
    from llm_meter.models import (
        Completion,
        ErrorObservation,
        RequestStart,
        StreamEvent,
        TokenCountSource,
    )

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
        configuration=configuration,
        request_start=request_start,
        stream_events=stream_events,
        completion=completion,
        error=error,
        usage=usage,
        metrics=metrics,
        provenance=provenance,
    )