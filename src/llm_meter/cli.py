from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

from llm_meter import __version__
from llm_meter.artifact import build_run, write_artifact, write_session
from llm_meter.client import Clock, stream_completion
from llm_meter.models import RunConfiguration, WorkloadProvenance
from llm_meter.runner import BenchmarkPlan, RequestExecutor, run_session
from llm_meter.tokenizer import load_tokenizer
from llm_meter.workload import (
    PromptSource,
    RequestSpec,
    WorkloadSpec,
    resolve_workload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-meter",
        description=(
            "Transparent, reproducible metrology for LLM inference. "
            "Under active development; V1 is being designed."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"llm-meter {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_one = subparsers.add_parser(
        "run-one",
        help="Execute a single streaming request and save a BenchmarkRun artifact (experimental).",
    )
    run_one.add_argument(
        "--endpoint", required=True, help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)"
    )
    run_one.add_argument("--model", required=True, help="Model name")
    run_one.add_argument(
        "--prompt", default=None, help="Prompt text (mutually exclusive with --input-tokens)"
    )
    run_one.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer ID for local token counting / prompt construction",
    )
    run_one.add_argument(
        "--input-tokens",
        type=int,
        default=None,
        help="Target input token count (requires --tokenizer, mutually exclusive with --prompt)",
    )
    run_one.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens (also used as workload output_tokens_target)",
    )
    run_one.add_argument(
        "--seed", type=int, default=0, help="Deterministic workload seed (default: 0)"
    )
    run_one.add_argument(
        "--output", default="run.json", help="Output artifact path (default: run.json)"
    )

    run_batch = subparsers.add_parser(
        "run-batch",
        help="Execute warmup + measured requests at fixed concurrency (experimental).",
    )
    run_batch.add_argument(
        "--endpoint", required=True, help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)"
    )
    run_batch.add_argument("--model", required=True, help="Model name")
    run_batch.add_argument(
        "--prompt", default=None, help="Prompt text (mutually exclusive with --input-tokens)"
    )
    run_batch.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer ID for local token counting / prompt construction",
    )
    run_batch.add_argument(
        "--input-tokens",
        type=int,
        default=None,
        help="Target input token count (requires --tokenizer, mutually exclusive with --prompt)",
    )
    run_batch.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens (also used as workload output_tokens_target)",
    )
    run_batch.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Number of warmup requests (default: 0)",
    )
    run_batch.add_argument(
        "--requests",
        type=int,
        required=True,
        help="Number of measured requests (must be > 0)",
    )
    run_batch.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum simultaneous in-flight requests (default: 1)",
    )
    run_batch.add_argument(
        "--seed", type=int, default=0, help="Deterministic workload seed (default: 0)"
    )
    run_batch.add_argument(
        "--output",
        default="session.json",
        help="Output session artifact path (default: session.json)",
    )

    workload_parser = subparsers.add_parser("workload", help="Workload specification tools")
    workload_subparsers = workload_parser.add_subparsers(dest="workload_command")

    inspect_parser = workload_subparsers.add_parser(
        "inspect",
        help="Inspect a resolved workload specification",
    )
    inspect_parser.add_argument(
        "--tokenizer", required=True, help="Tokenizer ID (Hugging Face model id)"
    )
    inspect_parser.add_argument(
        "--input-tokens", type=int, required=True, help="Target input token count"
    )
    inspect_parser.add_argument(
        "--output-tokens", type=int, required=True, help="Target output token count"
    )
    inspect_parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic seed (default: 0)"
    )
    inspect_parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Show the full generated prompt text",
    )

    return parser


def _format_ms(ns: int | None) -> str:
    if ns is None:
        return "N/A"
    return f"{ns / 1_000_000:.2f} ms"


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _build_workload_provenance(
    request_spec: RequestSpec, source: str
) -> WorkloadProvenance:
    tokenizer_prov = request_spec.tokenizer_provenance
    return WorkloadProvenance(
        source=source,
        seed=request_spec.workload_seed,
        input_tokens_target=request_spec.input_tokens_target,
        output_tokens_target=request_spec.max_output_tokens,
        input_tokens_actual_local=request_spec.input_tokens_actual_local,
        resolution_status=request_spec.resolution_status.value,
        prompt_sha256=request_spec.prompt_sha256,
        prompt_chars=request_spec.prompt_chars,
        tokenizer_provider=tokenizer_prov.provider if tokenizer_prov else None,
        tokenizer_id=tokenizer_prov.tokenizer_id if tokenizer_prov else None,
        tokenizer_revision=tokenizer_prov.revision if tokenizer_prov else None,
    )


def _validate_cli_inputs(args: argparse.Namespace) -> None:
    if args.prompt is not None and args.input_tokens is not None:
        _fail("--prompt and --input-tokens are mutually exclusive")

    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        _fail("--max-output-tokens must be positive")

    if args.input_tokens is not None and args.input_tokens <= 0:
        _fail("--input-tokens must be positive")


def _build_workload_spec(
    args: argparse.Namespace,
) -> tuple[WorkloadSpec, str | None, str]:
    if args.prompt is not None:
        spec = WorkloadSpec(
            input_tokens_target=None,
            output_tokens_target=args.max_output_tokens,
            seed=args.seed,
            prompt_source=PromptSource.MANUAL.value,
            tokenizer_id=args.tokenizer,
        )
        return spec, args.prompt, PromptSource.MANUAL.value

    if args.input_tokens is not None:
        if not args.tokenizer:
            _fail("--input-tokens requires --tokenizer")
        if args.max_output_tokens is None:
            _fail("--max-output-tokens is required with --input-tokens")
        spec = WorkloadSpec(
            input_tokens_target=args.input_tokens,
            output_tokens_target=args.max_output_tokens,
            seed=args.seed,
            prompt_source=PromptSource.BUILTIN.value,
            tokenizer_id=args.tokenizer,
        )
        return spec, None, PromptSource.BUILTIN.value

    _fail("either --prompt or --input-tokens is required")
    raise SystemExit(1)


def _resolve_prompt(args: argparse.Namespace) -> tuple[str, RequestSpec, str]:
    _validate_cli_inputs(args)
    spec, manual_prompt, source = _build_workload_spec(args)
    tokenizer = load_tokenizer(args.tokenizer)
    request_spec = resolve_workload(spec, tokenizer, manual_prompt=manual_prompt)
    prompt = request_spec.prompt
    return prompt, request_spec, source


def _run_one(args: argparse.Namespace) -> int:
    prompt, request_spec, source = _resolve_prompt(args)

    api_key = os.environ.get("LLM_METER_API_KEY")

    configuration = RunConfiguration(
        endpoint=args.endpoint,
        model=args.model,
        streaming=True,
        max_output_tokens=args.max_output_tokens,
    )

    workload_prov = _build_workload_provenance(request_spec, source)

    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()

    observations = asyncio.run(
        stream_completion(
            endpoint=args.endpoint,
            model=args.model,
            prompt=prompt,
            max_output_tokens=args.max_output_tokens,
            api_key=api_key,
            clock=Clock(),
        )
    )

    run = build_run(
        run_id=run_id,
        started_at=started_at,
        configuration=configuration,
        observations=observations,
        workload=workload_prov,
    )

    output_path = write_artifact(run, args.output)

    print(f"run_id:           {run.run_id}")
    print(f"schema_version:   {run.schema_version}")
    print(f"run_status:       {run.run_status}")
    print(f"artifact:         {output_path}")
    print(f"client_ttft:      {_format_ms(run.metrics.client_ttft_ns)}")
    print(f"e2e_latency:      {_format_ms(run.metrics.e2e_latency_ns)}")
    print(f"inter_chunk_lags: {len(run.metrics.inter_chunk_latencies_ns)} intervals")
    out_tokens = run.usage.output_tokens if run.usage.output_tokens is not None else "N/A"
    print(f"output_tokens:    {out_tokens}")
    print(f"token_source:     {run.usage.source.value}")
    print(f"tpot:             {_format_ms(run.metrics.tpot_ns)} ({run.metrics.tpot_status})")
    if run.error is not None:
        print(f"error:            {run.error.category}: {run.error.message}")

    return 0 if run.error is None else 1


def _make_production_executor(
    endpoint: str,
    model: str,
    api_key: str | None,
) -> RequestExecutor:
    async def execute(
        request_spec: RequestSpec,
        *,
        client: Any,
        started_at: str,
        session_start_offset_ns: int,
        source: str,
    ) -> Any:
        from llm_meter.client import Clock

        observations = await stream_completion(
            endpoint=endpoint,
            model=model,
            prompt=request_spec.prompt,
            max_output_tokens=request_spec.max_output_tokens,
            api_key=api_key,
            clock=Clock(),
            client=client,
        )

        configuration = RunConfiguration(
            endpoint=endpoint,
            model=model,
            streaming=True,
            max_output_tokens=request_spec.max_output_tokens,
        )
        workload_prov = _build_workload_provenance(request_spec, source)

        return build_run(
            run_id=str(uuid.uuid4()),
            started_at=started_at,
            configuration=configuration,
            observations=observations,
            workload=workload_prov,
        )

    return execute


def _validate_batch_inputs(args: argparse.Namespace) -> None:
    if args.requests <= 0:
        _fail("--requests must be > 0")

    if args.warmup_requests < 0:
        _fail("--warmup-requests must be >= 0")

    if args.concurrency <= 0:
        _fail("--concurrency must be > 0")

    _validate_cli_inputs(args)


def _run_batch(args: argparse.Namespace) -> int:
    _validate_batch_inputs(args)

    spec, manual_prompt, source = _build_workload_spec(args)
    tokenizer = load_tokenizer(args.tokenizer)

    plan = BenchmarkPlan(
        warmup_requests=args.warmup_requests,
        measured_requests=args.requests,
        concurrency=args.concurrency,
        workload=spec,
    )

    api_key = os.environ.get("LLM_METER_API_KEY")
    executor = _make_production_executor(
        endpoint=args.endpoint,
        model=args.model,
        api_key=api_key,
    )

    session = asyncio.run(
        run_session(
            plan,
            executor,
            endpoint=args.endpoint,
            model=args.model,
            api_key=api_key,
            tokenizer=tokenizer,
            manual_prompt=manual_prompt,
        )
    )

    output_path = write_session(session, args.output)

    warmup_count = len(session.warmup_runs)
    measured_count = len(session.measured_runs)
    completed_count = sum(
        1 for r in session.requests if r.run.run_status == "completed"
    )
    failed_count = sum(
        1 for r in session.requests if r.run.run_status == "failed"
    )

    print(f"session_id:       {session.session_id}")
    print(f"schema_version:   {session.schema_version}")
    print(f"status:           {session.status}")
    print(f"warmup_requests:  {warmup_count}")
    print(f"measured_requests: {measured_count}")
    print(f"completed:        {completed_count}")
    print(f"failed:           {failed_count}")
    print(f"concurrency:      {session.configuration.concurrency}")
    print(f"artifact:         {output_path}")

    return 0 if session.status == "completed" else 1


def _workload_inspect(args: argparse.Namespace) -> int:
    if args.input_tokens <= 0:
        _fail("--input-tokens must be positive")
    if args.output_tokens <= 0:
        _fail("--output-tokens must be positive")

    spec = WorkloadSpec(
        input_tokens_target=args.input_tokens,
        output_tokens_target=args.output_tokens,
        seed=args.seed,
        prompt_source=PromptSource.BUILTIN.value,
        tokenizer_id=args.tokenizer,
    )
    tokenizer = load_tokenizer(args.tokenizer)
    request_spec = resolve_workload(spec, tokenizer)

    print("source:                builtin")
    print(f"input_tokens_target:   {request_spec.input_tokens_target}")
    print(f"input_tokens_actual:   {request_spec.input_tokens_actual_local}")
    print(f"resolution:            {request_spec.resolution_status.value}")
    print(f"output_tokens_target:  {request_spec.max_output_tokens}")
    if request_spec.tokenizer_provenance:
        tp = request_spec.tokenizer_provenance
        print(
            f"tokenizer:              {tp.provider} / {tp.tokenizer_id} "
            f"/ {tp.revision or 'null'}"
        )
    else:
        print("tokenizer:              none")
    print(f"prompt_sha256:          {request_spec.prompt_sha256}")
    print(f"prompt_chars:           {request_spec.prompt_chars}")
    if args.show_prompt:
        print("--- prompt ---")
        print(request_spec.prompt)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-one":
        return _run_one(args)
    if args.command == "run-batch":
        return _run_batch(args)
    if args.command == "workload":
        if args.workload_command == "inspect":
            return _workload_inspect(args)
        parser.error("workload subcommand required")

    return 0


if __name__ == "__main__":
    sys.exit(main())
