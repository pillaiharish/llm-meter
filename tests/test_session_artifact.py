from __future__ import annotations

import asyncio
import json

from llm_meter.artifact import (
    SESSION_SCHEMA_VERSION,
    session_from_json,
    session_to_json,
    write_session,
)
from llm_meter.models import (
    BenchmarkRun,
    BenchmarkSession,
    Provenance,
    RunConfiguration,
    TokenCountSource,
    Usage,
    WorkloadProvenance,
)
from llm_meter.runner import BenchmarkPlan, run_session
from llm_meter.tokenizer import FakeTokenizer
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


class _FakeExecutor:
    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(
        self,
        request_spec: object,
        *,
        client: object,
        started_at: str,
        session_start_offset_ns: int,
        source: str,
    ) -> BenchmarkRun:
        self.call_count += 1
        await asyncio.sleep(0)
        return _make_run(run_id=f"run-{self.call_count}")


def _make_plan(
    warmup: int = 2,
    measured: int = 4,
    concurrency: int = 2,
) -> BenchmarkPlan:
    spec = WorkloadSpec(
        input_tokens_target=50,
        output_tokens_target=64,
        seed=42,
        prompt_source=PromptSource.BUILTIN.value,
        tokenizer_id="fake",
    )
    return BenchmarkPlan(
        warmup_requests=warmup,
        measured_requests=measured,
        concurrency=concurrency,
        workload=spec,
    )


def _make_session() -> BenchmarkSession:
    plan = _make_plan()
    executor = _FakeExecutor()
    tok = FakeTokenizer(tokenizer_id="fake")
    return asyncio.run(
        run_session(
            plan,
            executor,
            endpoint="http://localhost:8000/v1",
            model="test-model",
            max_output_tokens=64,
            api_key=None,
            tokenizer=tok,
        )
    )


def test_session_schema_version() -> None:
    assert SESSION_SCHEMA_VERSION == "1"


def test_prompts_absent_from_serialized_session() -> None:
    session = _make_session()
    json_str = session_to_json(session)
    data = json.loads(json_str)

    assert "prompt" not in data.get("configuration", {})

    for req in data.get("requests", []):
        run = req.get("run", {})
        assert "prompt" not in run
        assert "messages" not in run
        wl = run.get("workload", {})
        if wl:
            assert "prompt" not in wl
            assert "prompt_text" not in wl


def test_each_request_retains_workload_provenance() -> None:
    session = _make_session()
    for r in session.requests:
        assert r.run.workload is not None
        assert r.run.workload.prompt_sha256 != ""
        assert r.run.workload.seed >= 0


def test_each_request_retains_own_server_usage() -> None:
    session = _make_session()
    for r in session.requests:
        assert r.run.usage is not None
        assert r.run.usage.input_tokens is not None
        assert r.run.usage.output_tokens is not None
        assert r.run.usage.source == TokenCountSource.SERVER_REPORTED


def test_session_json_round_trip() -> None:
    session = _make_session()
    json_str = session_to_json(session)
    restored = session_from_json(json_str)

    assert restored.schema_version == session.schema_version
    assert restored.session_id == session.session_id
    assert restored.status == session.status
    assert len(restored.requests) == len(session.requests)

    assert restored.configuration.warmup_requests == 2
    assert restored.configuration.measured_requests == 4
    assert restored.configuration.concurrency == 2
    assert restored.configuration.endpoint == "http://localhost:8000/v1"
    assert restored.configuration.model == "test-model"


def test_benchmark_run_json_backward_compatible() -> None:
    from llm_meter.artifact import from_json, to_json

    run = BenchmarkRun(
        schema_version="1",
        run_id="compat-test",
        started_at="2025-01-01T00:00:00+00:00",
        run_status="completed",
        configuration=RunConfiguration(
            endpoint="http://localhost:8000/v1",
            model="test-model",
            streaming=True,
            max_output_tokens=64,
        ),
        provenance=Provenance(llm_meter_version="test"),
        usage=Usage(input_tokens=5, output_tokens=10),
    )
    json_str = to_json(run)
    restored = from_json(json_str)
    assert restored.schema_version == "1"
    assert restored.run_id == "compat-test"
    assert restored.run_status == "completed"


def test_session_warmup_measured_separation() -> None:
    session = _make_session()
    assert len(session.warmup_runs) == 2
    assert len(session.measured_runs) == 4
    for r in session.warmup_runs:
        assert r.phase == "warmup"
    for r in session.measured_runs:
        assert r.phase == "measured"


def test_session_write_to_file(tmp_path) -> None:
    session = _make_session()
    output = tmp_path / "session.json"
    write_session(session, str(output))
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["schema_version"] == "1"
    assert len(data["requests"]) == 6