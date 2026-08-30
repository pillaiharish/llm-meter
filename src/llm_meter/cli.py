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
from llm_meter.models import RunConfiguration


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
    run_one.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)")
    run_one.add_argument("--model", required=True, help="Model name")
    run_one.add_argument("--prompt", required=True, help="Prompt text")
    run_one.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens",
    )
    run_one.add_argument(
        "--output",
        default="run.json",
        help="Output artifact path (default: run.json)",
    )

    return parser


def _format_ms(ns: int | None) -> str:
    if ns is None:
        return "N/A"
    return f"{ns / 1_000_000:.2f} ms"


def _run_one(args: argparse.Namespace) -> int:
    api_key = os.environ.get("LLM_METER_API_KEY")

    configuration = RunConfiguration(
        endpoint=args.endpoint,
        model=args.model,
        streaming=True,
        max_output_tokens=args.max_output_tokens,
    )

    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC).isoformat()

    observations = asyncio.run(
        stream_completion(
            endpoint=args.endpoint,
            model=args.model,
            prompt=args.prompt,
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
    )

    output_path = write_artifact(run, args.output)

    print(f"run_id:           {run.run_id}")
    print(f"schema_version:   {run.schema_version}")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-one":
        return _run_one(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())