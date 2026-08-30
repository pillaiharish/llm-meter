from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from llm_meter.models import (
    Completion,
    ErrorObservation,
    RawObservations,
    RequestStart,
    ResponseEstablished,
    StreamEvent,
    TokenCountSource,
    Usage,
)
from llm_meter.sse import DONE, SSEParseError, parse_sse_data


class Clock:
    def __init__(self) -> None:
        self._origin_ns: int | None = None

    def start(self) -> int:
        self._origin_ns = time.perf_counter_ns()
        return 0

    def now_offset_ns(self) -> int:
        if self._origin_ns is None:
            raise RuntimeError("Clock not started; call start() first")
        return time.perf_counter_ns() - self._origin_ns

    @staticmethod
    def wall_clock_utc() -> str:
        return datetime.now(UTC).isoformat()


class FakeClock(Clock):
    def __init__(self, intervals_ns: list[int]) -> None:
        super().__init__()
        self._intervals = list(intervals_ns)
        self._current_ns = 0

    def start(self) -> int:
        self._origin_ns = 0
        self._current_ns = 0
        return 0

    def now_offset_ns(self) -> int:
        if not self._intervals:
            return self._current_ns
        self._current_ns += self._intervals.pop(0)
        return self._current_ns


def _classify_event(
    data: dict[str, Any],
) -> tuple[str, str | None, str | None, dict[str, Any] | None]:
    event_type = "metadata"
    text_delta: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    choices = data.get("choices", [])
    if choices:
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        if content is None:
            pass
        elif content == "":
            text_delta = ""
        else:
            event_type = "content"
            text_delta = content
        fr = choice.get("finish_reason")
        if fr is not None:
            finish_reason = fr

    if "usage" in data and data["usage"] is not None:
        usage = data["usage"]

    return event_type, text_delta, finish_reason, usage


def _extract_usage(usage_data: dict[str, Any]) -> Usage:
    prompt_tokens = usage_data.get("prompt_tokens")
    completion_tokens = usage_data.get("completion_tokens")
    return Usage(
        input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        source=TokenCountSource.SERVER_REPORTED,
    )


async def stream_completion(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_output_tokens: int | None = None,
    api_key: str | None = None,
    clock: Clock | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RawObservations:
    if clock is None:
        clock = Clock()

    key = api_key or os.environ.get("LLM_METER_API_KEY")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    base = endpoint.rstrip("/")
    url = f"{base}/chat/completions"

    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if max_output_tokens is not None:
        body["max_tokens"] = max_output_tokens

    stream_events: list[StreamEvent] = []
    completion: Completion | None = None
    response_established: ResponseEstablished | None = None
    error: ErrorObservation | None = None
    usage = Usage(source=TokenCountSource.UNKNOWN)
    sequence = 0

    client_kwargs: dict[str, Any] = {}
    if transport is not None:
        client_kwargs["transport"] = transport

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            clock.start()
            request_start = RequestStart(
                offset_ns=0,
                wall_clock_utc=Clock.wall_clock_utc(),
            )

            async with client.stream(
                "POST",
                url,
                json=body,
                headers=headers,
                timeout=httpx.Timeout(60.0, connect=10.0),
            ) as response:
                response_established = ResponseEstablished(
                    offset_ns=clock.now_offset_ns(),
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type"),
                )

                if response.status_code >= 400:
                    error = ErrorObservation(
                        offset_ns=clock.now_offset_ns(),
                        category="http_error",
                        status=response.status_code,
                        message=f"HTTP {response.status_code}",
                    )
                    return RawObservations(
                        request_start=request_start,
                        stream_events=stream_events,
                        response_established=response_established,
                        completion=completion,
                        error=error,
                        usage=usage,
                    )

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    receive_offset_ns = clock.now_offset_ns()
                    try:
                        parsed = parse_sse_data(line)
                    except SSEParseError as exc:
                        error = ErrorObservation(
                            offset_ns=receive_offset_ns,
                            category="sse_parse",
                            exception_type=type(exc).__name__,
                            message=str(exc),
                        )
                        return RawObservations(
                            request_start=request_start,
                            stream_events=stream_events,
                            response_established=response_established,
                            completion=completion,
                            error=error,
                            usage=usage,
                        )

                    if parsed is DONE:
                        completion = Completion(
                            offset_ns=receive_offset_ns,
                            wall_clock_utc=Clock.wall_clock_utc(),
                        )
                        break
                    if not parsed:
                        continue

                    event_type, text_delta, finish_reason, event_usage = _classify_event(parsed)

                    if event_usage is not None:
                        usage = _extract_usage(event_usage)

                    stream_events.append(
                        StreamEvent(
                            sequence=sequence,
                            offset_ns=receive_offset_ns,
                            event_type=event_type,
                            text_delta=text_delta,
                            finish_reason=finish_reason,
                            usage=event_usage,
                        )
                    )
                    sequence += 1

                if completion is None:
                    error = ErrorObservation(
                        offset_ns=clock.now_offset_ns(),
                        category="stream_unexpected_end",
                        message="stream ended without [DONE]",
                    )

    except httpx.TransportError as exc:
        error = ErrorObservation(
            offset_ns=clock.now_offset_ns() if clock._origin_ns is not None else 0,
            category="transport",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
    except httpx.HTTPError as exc:
        error = ErrorObservation(
            offset_ns=clock.now_offset_ns() if clock._origin_ns is not None else 0,
            category="http_error",
            exception_type=type(exc).__name__,
            message=str(exc),
        )

    return RawObservations(
        request_start=request_start,
        stream_events=stream_events,
        response_established=response_established,
        completion=completion,
        error=error,
        usage=usage,
    )
