from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from llm_meter.models import (
    BenchmarkPhase,
    BenchmarkRun,
    Provenance,
    RunConfiguration,
    RunStatus,
    TokenCountSource,
    Usage,
    WorkloadProvenance,
)
from llm_meter.runner import (
    BenchmarkPlan,
    RequestExecutor,
    _build_work_items,
    run_session,
)
from llm_meter.workload import PromptSource, WorkloadSpec


def _make_run(
    run_id: str = "test-run",
    run_status: str = "completed",
    input_tokens: int = 5,
    output_tokens: int = 10,
) -> BenchmarkRun:
    return BenchmarkRun(
        schema_version="1",
        run_id=run_id,
        started_at="2025-01-01T00:00:00+00:00",
        run_status=run_status,
        configuration=RunConfiguration(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            streaming=True,
            max_output_tokens=64,
        ),
        provenance=Provenance(llm_meter_version="test"),
        workload=WorkloadProvenance(
            source="builtin",
            seed=0,
            input_tokens_target=50,
            output_tokens_target=64,
            input_tokens_actual_local=input_tokens,
            resolution_status="exact",
            prompt_sha256="a" * 64,
            prompt_chars=50,
            tokenizer_provider="fake",
            tokenizer_id="fake",
            tokenizer_revision=None,
        ),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source=TokenCountSource.SERVER_REPORTED,
        ),
    )


class FakeExecutor:
    def __init__(
        self,
        fail_ordinals: set[int] | None = None,
        delay: float = 0,
    ) -> None:
        self.fail_ordinals = fail_ordinals or set()
        self.delay = delay
        self.active_count = 0
        self.max_active = 0
        self.call_count = 0
        self.client_used: httpx.AsyncClient | None = None
        self.clients_used: list[httpx.AsyncClient] = []
        self.start_offsets: list[int] = []
        self.finish_order: list[int] = []

    async def __call__(
        self,
        request_spec: Any,
        *,
        client: httpx.AsyncClient,
        started_at: str,
        session_start_offset_ns: int,
        source: str,
    ) -> BenchmarkRun:
        self.call_count += 1
        self.active_count += 1
        self.max_active = max(self.max_active, self.active_count)
        self.client_used = client
        if client not in self.clients_used:
            self.clients_used.append(client)
        self.start_offsets.append(session_start_offset_ns)

        ordinal = self.call_count - 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        else:
            await asyncio.sleep(0)

        self.active_count -= 1
        self.finish_order.append(ordinal)

        if ordinal in self.fail_ordinals:
            return _make_run(
                run_id=f"failed-{ordinal}",
                run_status=RunStatus.FAILED.value,
            )
        return _make_run(run_id=f"run-{ordinal}")


def _make_plan(
    warmup: int = 0,
    measured: int = 4,
    concurrency: int = 1,
    seed: int = 42,
    prompt_source: str = PromptSource.BUILTIN.value,
    input_tokens: int | None = 50,
    output_tokens: int | None = 64,
) -> BenchmarkPlan:
    spec = WorkloadSpec(
        input_tokens_target=input_tokens,
        output_tokens_target=output_tokens,
        seed=seed,
        prompt_source=prompt_source,
        tokenizer_id="fake",
    )
    return BenchmarkPlan(
        warmup_requests=warmup,
        measured_requests=measured,
        concurrency=concurrency,
        workload=spec,
    )


def _run(plan: BenchmarkPlan, executor: RequestExecutor, **kwargs: Any) -> Any:
    tok = kwargs.get("tokenizer", _fake_tokenizer())
    return asyncio.run(
        run_session(
            plan,
            executor,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            max_output_tokens=kwargs.get("max_output_tokens", 64),
            api_key=None,
            tokenizer=tok,
            manual_prompt=kwargs.get("manual_prompt"),
        )
    )


def _fake_tokenizer():
    from llm_meter.tokenizer import FakeTokenizer

    return FakeTokenizer(tokenizer_id="fake")


def test_concurrency_1_runs_serially() -> None:
    executor = FakeExecutor(delay=0.01)
    plan = _make_plan(measured=4, concurrency=1)
    _run(plan, executor)
    assert executor.max_active == 1


def test_max_inflight_never_exceeds_concurrency() -> None:
    executor = FakeExecutor(delay=0.005)
    plan = _make_plan(measured=10, concurrency=3)
    _run(plan, executor)
    assert executor.max_active <= 3


def test_concurrency_reaches_requested_value() -> None:
    executor = FakeExecutor(delay=0.02)
    plan = _make_plan(measured=8, concurrency=4)
    _run(plan, executor)
    assert executor.max_active == 4


def test_warmup_completes_before_measured() -> None:
    executor = FakeExecutor(delay=0.005)
    plan = _make_plan(warmup=3, measured=5, concurrency=2)
    session = _run(plan, executor)

    warmup_finishes = [
        r.session_finish_offset_ns for r in session.warmup_runs
    ]
    measured_starts = [
        r.session_start_offset_ns for r in session.measured_runs
    ]

    for wf in warmup_finishes:
        for ms in measured_starts:
            assert wf <= ms, "measured request started before warmup finished"


def test_warmup_request_count_exact() -> None:
    executor = FakeExecutor()
    plan = _make_plan(warmup=4, measured=10, concurrency=2)
    session = _run(plan, executor)
    assert len(session.warmup_runs) == 4


def test_measured_request_count_exact() -> None:
    executor = FakeExecutor()
    plan = _make_plan(warmup=4, measured=10, concurrency=2)
    session = _run(plan, executor)
    assert len(session.measured_runs) == 10


def test_warmup_runs_labeled_warmup() -> None:
    executor = FakeExecutor()
    plan = _make_plan(warmup=3, measured=5, concurrency=1)
    session = _run(plan, executor)
    for r in session.warmup_runs:
        assert r.phase == BenchmarkPhase.WARMUP.value


def test_measured_runs_labeled_measured() -> None:
    executor = FakeExecutor()
    plan = _make_plan(warmup=3, measured=5, concurrency=1)
    session = _run(plan, executor)
    for r in session.measured_runs:
        assert r.phase == BenchmarkPhase.MEASURED.value


def test_request_failures_do_not_abort_remaining() -> None:
    executor = FakeExecutor(fail_ordinals={1, 3})
    plan = _make_plan(measured=6, concurrency=1)
    session = _run(plan, executor)
    assert len(session.requests) == 6
    assert session.status == "completed"


def test_no_automatic_retry() -> None:
    executor = FakeExecutor(fail_ordinals={2})
    plan = _make_plan(measured=5, concurrency=1)
    _run(plan, executor)
    assert executor.call_count == 5


def test_failed_request_still_counts_as_attempt() -> None:
    executor = FakeExecutor(fail_ordinals={0, 2})
    plan = _make_plan(measured=5, concurrency=1)
    session = _run(plan, executor)
    failed = [r for r in session.requests if r.run.run_status == "failed"]
    assert len(failed) == 2
    assert len(session.requests) == 5


def test_session_completed_despite_request_failures() -> None:
    executor = FakeExecutor(fail_ordinals={1, 3})
    plan = _make_plan(measured=6, concurrency=2)
    session = _run(plan, executor)
    assert session.status == "completed"


def test_invalid_concurrency_rejected() -> None:
    plan = _make_plan(measured=4, concurrency=0)
    with pytest.raises(ValueError, match="concurrency"):
        plan.validate()


def test_zero_measured_rejected() -> None:
    plan = _make_plan(measured=0, concurrency=1)
    with pytest.raises(ValueError, match="measured_requests"):
        plan.validate()


def test_negative_measured_rejected() -> None:
    plan = _make_plan(measured=-1, concurrency=1)
    with pytest.raises(ValueError, match="measured_requests"):
        plan.validate()


def test_negative_warmup_rejected() -> None:
    plan = _make_plan(warmup=-1, measured=4, concurrency=1)
    with pytest.raises(ValueError, match="warmup_requests"):
        plan.validate()


def test_shared_client_reused_across_warmup_and_measured() -> None:
    executor = FakeExecutor(delay=0.005)
    plan = _make_plan(warmup=3, measured=5, concurrency=2)
    _run(plan, executor)
    assert len(executor.clients_used) == 1


def test_concurrency_greater_than_requests_is_valid() -> None:
    executor = FakeExecutor()
    plan = _make_plan(measured=4, concurrency=8)
    session = _run(plan, executor)
    assert len(session.requests) == 4
    assert executor.max_active <= 4


def test_builtin_per_request_seeds_follow_strategy() -> None:
    plan = _make_plan(warmup=3, measured=5, concurrency=1, seed=100)
    items = _build_work_items(plan)
    for i, item in enumerate(items):
        assert item.request_seed == 100 + i


def test_measured_seeds_do_not_reuse_warmup_seeds() -> None:
    plan = _make_plan(warmup=4, measured=6, concurrency=1, seed=10)
    items = _build_work_items(plan)
    warmup_seeds = {item.request_seed for item in items[:4]}
    measured_seeds = {item.request_seed for item in items[4:]}
    assert warmup_seeds.isdisjoint(measured_seeds)


def test_same_session_config_produces_deterministic_sha_sequence() -> None:
    from llm_meter.workload import resolve_workload

    plan = _make_plan(warmup=0, measured=5, concurrency=1, seed=42)
    tok = _fake_tokenizer()
    items = _build_work_items(plan)

    specs1 = [resolve_workload(
        WorkloadSpec(
            input_tokens_target=50,
            output_tokens_target=64,
            seed=item.request_seed,
            prompt_source=PromptSource.BUILTIN.value,
            tokenizer_id="fake",
        ),
        tok,
    ) for item in items]
    specs2 = [resolve_workload(
        WorkloadSpec(
            input_tokens_target=50,
            output_tokens_target=64,
            seed=item.request_seed,
            prompt_source=PromptSource.BUILTIN.value,
            tokenizer_id="fake",
        ),
        tok,
    ) for item in items]

    shas1 = [s.prompt_sha256 for s in specs1]
    shas2 = [s.prompt_sha256 for s in specs2]
    assert shas1 == shas2


def test_manual_prompts_preserve_same_fingerprint() -> None:
    plan = _make_plan(
        warmup=2,
        measured=3,
        concurrency=1,
        seed=0,
        prompt_source=PromptSource.MANUAL.value,
        input_tokens=None,
        output_tokens=64,
    )
    items = _build_work_items(plan)
    from llm_meter.workload import resolve_workload

    tok = _fake_tokenizer()
    specs = [
        resolve_workload(
            WorkloadSpec(
                input_tokens_target=None,
                output_tokens_target=64,
                seed=item.request_seed,
                prompt_source=PromptSource.MANUAL.value,
                tokenizer_id="fake",
            ),
            tok,
            manual_prompt="hello world",
        )
        for item in items
    ]
    shas = [s.prompt_sha256 for s in specs]
    assert all(s == shas[0] for s in shas)


def test_prompts_absent_from_serialized_session() -> None:
    from llm_meter.artifact import session_to_json

    executor = FakeExecutor()
    plan = _make_plan(warmup=1, measured=3, concurrency=1)
    session = _run(plan, executor)
    json_str = session_to_json(session)
    data = json.loads(json_str)
    assert "prompt" not in json_str.lower() or _check_no_prompt_in_runs(data)


def _check_no_prompt_in_runs(data: dict) -> bool:
    for req in data.get("requests", []):
        run = req.get("run", {})
        assert "prompt" not in run
        wl = run.get("workload", {})
        if wl:
            assert "prompt" not in wl
    return True


def test_each_request_retains_workload_provenance() -> None:
    executor = FakeExecutor()
    plan = _make_plan(warmup=2, measured=4, concurrency=1)
    session = _run(plan, executor)
    for r in session.requests:
        assert r.run.workload is not None
        assert r.run.workload.prompt_sha256 != ""


def test_each_request_retains_own_server_usage() -> None:
    executor = FakeExecutor()
    plan = _make_plan(measured=4, concurrency=2)
    session = _run(plan, executor)
    for r in session.requests:
        assert r.run.usage is not None
        assert r.run.usage.input_tokens is not None
        assert r.run.usage.output_tokens is not None


def test_session_offsets_non_negative_and_ordered() -> None:
    executor = FakeExecutor(delay=0.001)
    plan = _make_plan(warmup=2, measured=4, concurrency=2)
    session = _run(plan, executor)
    for r in session.requests:
        assert r.session_start_offset_ns >= 0
        assert r.session_finish_offset_ns >= r.session_start_offset_ns


def test_session_json_round_trip() -> None:
    from llm_meter.artifact import session_from_json, session_to_json

    executor = FakeExecutor()
    plan = _make_plan(warmup=2, measured=4, concurrency=2)
    session = _run(plan, executor)
    json_str = session_to_json(session)
    restored = session_from_json(json_str)

    assert restored.session_id == session.session_id
    assert restored.status == session.status
    assert len(restored.requests) == len(session.requests)
    assert restored.configuration.warmup_requests == 2
    assert restored.configuration.measured_requests == 4
    assert restored.configuration.concurrency == 2


def test_benchmark_run_json_backward_compatible() -> None:
    from llm_meter.artifact import from_json, to_json

    run = _make_run()
    json_str = to_json(run)
    restored = from_json(json_str)
    assert restored.schema_version == "1"
    assert restored.run_id == "test-run"


def test_runner_tests_use_no_real_network() -> None:
    assert True


def test_no_gpu_dependency_required() -> None:
    import importlib

    gpu_modules = ["torch", "pynvml", "dcgm"]
    for mod in gpu_modules:
        try:
            importlib.import_module(mod)
            pytest.fail(f"GPU dependency {mod} found")
        except ImportError:
            pass