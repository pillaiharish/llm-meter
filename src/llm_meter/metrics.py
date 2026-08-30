from __future__ import annotations

from llm_meter.models import (
    ClientMetrics,
    RawObservations,
    TokenCountSource,
    TpotStatus,
    Usage,
)


def derive_metrics(observations: RawObservations, usage: Usage | None = None) -> ClientMetrics:
    if usage is None:
        usage = observations.usage

    request_start = observations.request_start
    stream_events = observations.stream_events
    completion = observations.completion

    client_ttft_ns: int | None = None
    for event in stream_events:
        if event.event_type == "content":
            client_ttft_ns = event.offset_ns - request_start.offset_ns
            break

    e2e_latency_ns: int | None = None
    if completion is not None:
        e2e_latency_ns = completion.offset_ns - request_start.offset_ns

    inter_chunk_latencies_ns: list[int] = []
    for i in range(1, len(stream_events)):
        gap = stream_events[i].offset_ns - stream_events[i - 1].offset_ns
        inter_chunk_latencies_ns.append(gap)

    tpot_ns, tpot_status = _derive_tpot(
        e2e_latency_ns=e2e_latency_ns,
        client_ttft_ns=client_ttft_ns,
        output_tokens=usage.output_tokens,
        token_source=usage.source,
    )

    return ClientMetrics(
        client_ttft_ns=client_ttft_ns,
        e2e_latency_ns=e2e_latency_ns,
        inter_chunk_latencies_ns=inter_chunk_latencies_ns,
        tpot_ns=tpot_ns,
        tpot_status=tpot_status,
    )


def _derive_tpot(
    *,
    e2e_latency_ns: int | None,
    client_ttft_ns: int | None,
    output_tokens: int | None,
    token_source: TokenCountSource,
) -> tuple[int | None, str]:
    if e2e_latency_ns is None:
        return None, TpotStatus.NO_E2E.value
    if client_ttft_ns is None:
        return None, TpotStatus.NO_TTFT.value
    if output_tokens is None or token_source in (TokenCountSource.UNKNOWN,):
        return None, TpotStatus.NO_TOKEN_COUNT.value
    if output_tokens <= 1:
        return None, TpotStatus.INSUFFICIENT_TOKENS.value

    decode_window_ns = e2e_latency_ns - client_ttft_ns
    token_divisor = output_tokens - 1
    return decode_window_ns // token_divisor, TpotStatus.OK.value