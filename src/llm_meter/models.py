from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class TokenCountSource(StrEnum):
    SERVER_REPORTED = "server_reported"
    ENGINE_REPORTED = "engine_reported"
    LOCALLY_TOKENIZED = "locally_tokenized"
    UNKNOWN = "unknown"


class TpotStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_TOKENS = "insufficient_tokens"
    NO_TOKEN_COUNT = "no_token_count"
    NO_TTFT = "no_ttft"
    NO_E2E = "no_e2e"


@dataclass
class RequestStart:
    offset_ns: int
    wall_clock_utc: str


@dataclass
class StreamEvent:
    sequence: int
    offset_ns: int
    event_type: str
    text_delta: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class Completion:
    offset_ns: int
    wall_clock_utc: str


@dataclass
class ErrorObservation:
    offset_ns: int
    category: str
    status: int | None = None
    exception_type: str | None = None
    message: str = ""


@dataclass
class RunConfiguration:
    endpoint: str
    model: str
    streaming: bool
    max_output_tokens: int | None = None


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    source: TokenCountSource = TokenCountSource.UNKNOWN


@dataclass
class ClientMetrics:
    client_ttft_ns: int | None = None
    e2e_latency_ns: int | None = None
    inter_chunk_latencies_ns: list[int] = field(default_factory=list)
    tpot_ns: int | None = None
    tpot_status: str = TpotStatus.NO_TOKEN_COUNT.value


@dataclass
class Provenance:
    llm_meter_version: str


@dataclass
class BenchmarkRun:
    schema_version: str
    run_id: str
    started_at: str
    configuration: RunConfiguration
    request_start: RequestStart | None = None
    stream_events: list[StreamEvent] = field(default_factory=list)
    completion: Completion | None = None
    error: ErrorObservation | None = None
    usage: Usage = field(default_factory=Usage)
    metrics: ClientMetrics = field(default_factory=ClientMetrics)
    provenance: Provenance = field(default_factory=Provenance)


@dataclass
class RawObservations:
    request_start: RequestStart
    stream_events: list[StreamEvent]
    completion: Completion | None = None
    error: ErrorObservation | None = None
    usage: Usage = field(default_factory=Usage)


def dataclass_to_dict(obj: Any) -> Any:
    if isinstance(obj, StrEnum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    return obj