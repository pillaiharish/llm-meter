from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import UTC, datetime

from llm_meter import __version__
from llm_meter.artifact import build_run, write_artifact
from llm_meter.client import Clock, stream_completion
from llm_meter.models import RunConfiguration, WorkloadProvenance
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

    workload_parser = subparsers.add_parser("workload", help="Workload specification tools")
    workload_subparsers = workload_parser.add_subparsers(dest="workload_command")

    inspect_parser = workload_subparsers.add_parser(
        "inspect",
        help="Inspect a resolved workload specification",
    )
    inspect_parser.add_argument("--tokenizer", default="fake", help="Tokenizer ID (default: fake)")
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
        input_tokens_actual_local=request_spec.input_tokens_actual,
        resolution_status=request_spec.resolution_status.value,
        prompt_sha256=request_spec.prompt_sha256,
        prompt_chars=request_spec.prompt_chars,
        tokenizer_provider=tokenizer_prov.provider if tokenizer_prov else None,
        tokenizer_id=tokenizer_prov.tokenizer_id if tokenizer_prov else None,
        tokenizer_revision=tokenizer_prov.revision if tokenizer_prov else None,
    )


def _resolve_prompt(args: argparse.Namespace) -> tuple[str, RequestSpec, str]:
    if args.prompt is not None and args.input_tokens is not None:
        _fail("--prompt and --input-tokens are mutually exclusive")

    if args.prompt is not None:
        output_target = args.max_output_tokens or 1
        spec = WorkloadSpec(
            input_tokens_target=0,
            output_tokens_target=output_target,
            seed=args.seed,
            prompt_source=PromptSource.MANUAL.value,
            tokenizer_id=args.tokenizer,
        )
        tokenizer = load_tokenizer(args.tokenizer)
        request_spec = resolve_workload(spec, tokenizer, manual_prompt=args.prompt)
        return args.prompt, request_spec, PromptSource.MANUAL.value

    if args.input_tokens is not None:
        if not args.tokenizer:
            _fail("--input-tokens requires --tokenizer")
        output_target = args.max_output_tokens or 0
        if output_target <= 0:
            _fail("--max-output-tokens is required with --input-tokens")
        spec = WorkloadSpec(
            input_tokens_target=args.input_tokens,
            output_tokens_target=output_target,
            seed=args.seed,
            prompt_source=PromptSource.BUILTIN.value,
            tokenizer_id=args.tokenizer,
        )
        tokenizer = load_tokenizer(args.tokenizer)
        request_spec = resolve_workload(spec, tokenizer)
        return request_spec.prompt, request_spec, PromptSource.BUILTIN.value

    _fail("either --prompt or --input-tokens is required")


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


def _workload_inspect(args: argparse.Namespace) -> int:
    if args.input_tokens <= 0:
        raise SystemExit("error: --input-tokens must be positive")
    if args.output_tokens <= 0:
        raise SystemExit("error: --output-tokens must be positive")

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
    print(f"input_tokens_actual:   {request_spec.input_tokens_actual}")
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
    if args.command == "workload":
        if args.workload_command == "inspect":
            return _workload_inspect(args)
        parser.error("workload subcommand required")

    return 0


if __name__ == "__main__":
    sys.exit(main())