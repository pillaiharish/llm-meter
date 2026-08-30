from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from llm_meter import __version__
from llm_meter.models import (
    BenchmarkPhase,
    BenchmarkSession,
    Provenance,
    SeedStrategy,
    SessionConfiguration,
    SessionRequest,
    SessionStatus,
)
from llm_meter.workload import (
    PromptSource,
    RequestSpec,
    WorkloadSpec,
    resolve_workload,
)

_SENTINEL: Any = object()


@dataclass
class BenchmarkPlan:
    warmup_requests: int
    measured_requests: int
    concurrency: int
    workload: WorkloadSpec
    seed_strategy: SeedStrategy = SeedStrategy.BASE_PLUS_GLOBAL_ORDINAL

    def validate(self) -> None:
        if self.warmup_requests < 0:
            raise ValueError(
                f"warmup_requests must be >= 0, got {self.warmup_requests}"
            )
        if self.measured_requests <= 0:
            raise ValueError(
                f"measured_requests must be > 0, got {self.measured_requests}"
            )
        if self.concurrency <= 0:
            raise ValueError(
                f"concurrency must be > 0, got {self.concurrency}"
            )

    @property
    def total_requests(self) -> int:
        return self.warmup_requests + self.measured_requests


@dataclass
class WorkItem:
    phase: BenchmarkPhase
    ordinal: int
    global_ordinal: int
    request_seed: int


@runtime_checkable
class RequestExecutor(Protocol):
    async def __call__(
        self,
        request_spec: RequestSpec,
        *,
        client: httpx.AsyncClient,
        started_at: str,
        session_start_offset_ns: int,
        source: str,
    ) -> Any: ...


def _build_work_items(plan: BenchmarkPlan) -> list[WorkItem]:
    items: list[WorkItem] = []
    base_seed = plan.workload.seed

    is_builtin = plan.workload.prompt_source == PromptSource.BUILTIN.value

    for i in range(plan.warmup_requests):
        global_ordinal = i
        request_seed = base_seed + global_ordinal if is_builtin else base_seed
        items.append(
            WorkItem(
                phase=BenchmarkPhase.WARMUP,
                ordinal=i,
                global_ordinal=global_ordinal,
                request_seed=request_seed,
            )
        )

    for i in range(plan.measured_requests):
        global_ordinal = plan.warmup_requests + i
        request_seed = base_seed + global_ordinal if is_builtin else base_seed
        items.append(
            WorkItem(
                phase=BenchmarkPhase.MEASURED,
                ordinal=i,
                global_ordinal=global_ordinal,
                request_seed=request_seed,
            )
        )

    return items


def _resolve_request_spec(
    plan: BenchmarkPlan,
    item: WorkItem,
    tokenizer: Any,
    manual_prompt: str | None,
) -> RequestSpec:
    per_request_spec = WorkloadSpec(
        input_tokens_target=plan.workload.input_tokens_target,
        output_tokens_target=plan.workload.output_tokens_target,
        seed=item.request_seed,
        prompt_source=plan.workload.prompt_source,
        tokenizer_id=plan.workload.tokenizer_id,
    )
    return resolve_workload(
        per_request_spec, tokenizer, manual_prompt=manual_prompt
    )


async def _worker(
    queue: asyncio.Queue,
    results: dict[int, SessionRequest],
    plan: BenchmarkPlan,
    tokenizer: Any,
    manual_prompt: str | None,
    executor: RequestExecutor,
    client: httpx.AsyncClient,
    session_origin_ns: int,
) -> None:
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            queue.task_done()
            return

        idx, work_item = item
        request_spec = _resolve_request_spec(
            plan, work_item, tokenizer, manual_prompt
        )

        started_at = datetime.now(UTC).isoformat()
        start_offset = time.perf_counter_ns() - session_origin_ns

        run = await executor(
            request_spec,
            client=client,
            started_at=started_at,
            session_start_offset_ns=start_offset,
            source=plan.workload.prompt_source,
        )

        finish_offset = time.perf_counter_ns() - session_origin_ns

        results[idx] = SessionRequest(
            phase=work_item.phase.value,
            ordinal=work_item.ordinal,
            session_start_offset_ns=start_offset,
            session_finish_offset_ns=finish_offset,
            run=run,
        )
        queue.task_done()


async def _drain_queue(
    queue: asyncio.Queue,
    workers: list,
    num_workers: int,
) -> None:
    for _ in range(num_workers):
        await queue.put(_SENTINEL)
    await asyncio.gather(*workers)


async def run_session(
    plan: BenchmarkPlan,
    executor: RequestExecutor,
    *,
    endpoint: str,
    model: str,
    max_output_tokens: int | None,
    api_key: str | None,
    tokenizer: Any = None,
    manual_prompt: str | None = None,
) -> BenchmarkSession:
    plan.validate()

    work_items = _build_work_items(plan)
    warmup_items = work_items[: plan.warmup_requests]
    measured_items = work_items[plan.warmup_requests :]

    max_connections = plan.concurrency
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    session_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()
    session_origin_ns = time.perf_counter_ns()

    status = SessionStatus.COMPLETED.value
    session_requests: list[SessionRequest] = []

    try:
        async with httpx.AsyncClient(limits=limits, headers=headers) as client:
            results: dict[int, SessionRequest] = {}

            for phase_items in (warmup_items, measured_items):
                if not phase_items:
                    continue

                queue: asyncio.Queue = asyncio.Queue()
                base_idx = len(results)
                for offset, item in enumerate(phase_items):
                    queue.put_nowait((base_idx + offset, item))

                num_workers = min(plan.concurrency, len(phase_items))
                workers = [
                    asyncio.create_task(
                        _worker(
                            queue,
                            results,
                            plan,
                            tokenizer,
                            manual_prompt,
                            executor,
                            client,
                            session_origin_ns,
                        )
                    )
                    for _ in range(num_workers)
                ]

                await queue.join()
                await _drain_queue(queue, workers, num_workers)

            session_requests = [
                results[i] for i in range(len(work_items)) if i in results
            ]

    except Exception:
        status = SessionStatus.FAILED.value
        raise

    completed_at = datetime.now(UTC).isoformat()

    configuration = SessionConfiguration(
        endpoint=endpoint,
        model=model,
        warmup_requests=plan.warmup_requests,
        measured_requests=plan.measured_requests,
        concurrency=plan.concurrency,
        seed=plan.workload.seed,
        seed_strategy=plan.seed_strategy.value,
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
        prompt_source=plan.workload.prompt_source,
        input_tokens_target=plan.workload.input_tokens_target,
        output_tokens_target=plan.workload.output_tokens_target,
        tokenizer_id=plan.workload.tokenizer_id,
        max_output_tokens=max_output_tokens,
    )

    return BenchmarkSession(
        schema_version="1",
        session_id=session_id,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        configuration=configuration,
        requests=session_requests,
        provenance=Provenance(llm_meter_version=__version__),
    )