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


class RunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class BenchmarkPhase(StrEnum):
    WARMUP = "warmup"
    MEASURED = "measured"


class SessionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"  # reserved: runner-level failures raise; no partial session is serialized


class SeedStrategy(StrEnum):
    BASE_PLUS_GLOBAL_ORDINAL = "base_plus_global_ordinal"


@dataclass
class RequestStart:
    offset_ns: int
    wall_clock_utc: str


@dataclass
class ResponseEstablished:
    offset_ns: int
    status_code: int
    content_type: str | None = None


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
class WorkloadProvenance:
    source: str
    seed: int
    input_tokens_target: int | None
    output_tokens_target: int | None
    input_tokens_actual_local: int | None = None
    resolution_status: str = "not_applicable"
    prompt_sha256: str = ""
    prompt_chars: int = 0
    tokenizer_provider: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None


@dataclass
class BenchmarkRun:
    schema_version: str
    run_id: str
    started_at: str
    run_status: str
    configuration: RunConfiguration
    provenance: Provenance
    workload: WorkloadProvenance | None = None
    request_start: RequestStart | None = None
    response_established: ResponseEstablished | None = None
    stream_events: list[StreamEvent] = field(default_factory=list)
    completion: Completion | None = None
    error: ErrorObservation | None = None
    usage: Usage = field(default_factory=Usage)
    metrics: ClientMetrics = field(default_factory=ClientMetrics)


@dataclass
class RawObservations:
    request_start: RequestStart
    stream_events: list[StreamEvent]
    response_established: ResponseEstablished | None = None
    completion: Completion | None = None
    error: ErrorObservation | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass
class SessionConfiguration:
    endpoint: str
    model: str
    warmup_requests: int
    measured_requests: int
    concurrency: int
    seed: int
    seed_strategy: str
    max_connections: int
    max_keepalive_connections: int
    prompt_source: str
    input_tokens_target: int | None
    output_tokens_target: int | None
    tokenizer_id: str | None
    max_output_tokens: int | None = None


@dataclass
class SessionRequest:
    phase: str
    ordinal: int
    session_start_offset_ns: int
    session_finish_offset_ns: int
    run: BenchmarkRun


@dataclass
class BenchmarkSession:
    schema_version: str
    session_id: str
    started_at: str
    completed_at: str
    status: str
    configuration: SessionConfiguration
    requests: list[SessionRequest]
    provenance: Provenance

    @property
    def warmup_runs(self) -> list[SessionRequest]:
        return [r for r in self.requests if r.phase == BenchmarkPhase.WARMUP.value]

    @property
    def measured_runs(self) -> list[SessionRequest]:
        return [r for r in self.requests if r.phase == BenchmarkPhase.MEASURED.value]


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
