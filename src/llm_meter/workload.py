from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import StrEnum

from llm_meter.tokenizer import Tokenizer, TokenizerProvenance


class ResolutionStatus(StrEnum):
    EXACT = "exact"
    NEAREST = "nearest"
    UNRESOLVABLE = "unresolvable"


class PromptSource(StrEnum):
    BUILTIN = "builtin"
    MANUAL = "manual"


@dataclass
class WorkloadSpec:
    input_tokens_target: int
    output_tokens_target: int
    seed: int = 0
    prompt_source: str = PromptSource.BUILTIN.value
    tokenizer_id: str | None = None


@dataclass
class RequestSpec:
    prompt: str
    prompt_sha256: str
    prompt_chars: int
    input_tokens_target: int
    input_tokens_actual: int | None
    max_output_tokens: int
    resolution_status: ResolutionStatus
    tokenizer_provenance: TokenizerProvenance | None
    workload_seed: int


_CORPUS = [
    "The inference engine processes requests through a scheduler.",
    "Latency measurements must capture both queue wait and prefill time.",
    "Token throughput depends on batch composition and decode efficiency.",
    "Reproducible benchmarks require explicit workload and environment provenance.",
    "GPU memory pressure affects KV cache utilization during sustained inference.",
    "The time to first token reflects prefill computation and scheduling overhead.",
    "Inter-token latency varies with batch size and available GPU resources.",
    "Dynamic batching improves throughput by grouping concurrent requests.",
    "Benchmark artifacts should preserve raw observations for later analysis.",
    "Serving engines may apply chat templates that change tokenization.",
    "Prefill latency is dominated by prompt length and available compute.",
    "Decode latency depends on model architecture and batch composition.",
    "Concurrency control is essential for controlled benchmark measurements.",
    "Engine-neutral measurement requires adapters for runtime-specific metrics.",
    "Performance analysis must separate observation from interpretation.",
    "KV cache occupancy influences scheduling decisions in serving engines.",
    "Streaming responses arrive as SSE events with variable chunk sizes.",
    "Token counting provenance distinguishes server-reported from local estimates.",
    "Workload specifications define the intent of a benchmark measurement.",
    "Monotonic clocks provide reliable timing for latency measurements.",
]

_MAX_ITERATIONS = 20


def fingerprint_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _encode_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_to_tokens(text: str, target: int, tokenizer: Tokenizer) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= target:
        return text
    truncated = tokens[:target]
    return tokenizer.decode(truncated)


def _truncate_piece(piece: str, remaining: int, tokenizer: Tokenizer) -> str:
    tokens = tokenizer.encode(piece, add_special_tokens=False)
    if len(tokens) <= remaining:
        return piece
    truncated = tokens[:remaining]
    return tokenizer.decode(truncated)


def _deterministic_prompt(target_tokens: int, seed: int, tokenizer: Tokenizer) -> str:
    rng = random.Random(seed)
    pieces = list(_CORPUS)
    rng.shuffle(pieces)

    prompt_parts: list[str] = []
    for piece in pieces:
        candidate = " ".join(prompt_parts + [piece])
        if _encode_count(tokenizer, candidate) >= target_tokens:
            current = " ".join(prompt_parts)
            current_tokens = _encode_count(tokenizer, current)
            if current_tokens >= target_tokens:
                return _truncate_to_tokens(current, target_tokens, tokenizer)
            remaining = target_tokens - current_tokens
            truncated_piece = _truncate_piece(piece, remaining, tokenizer)
            if truncated_piece:
                return " ".join(prompt_parts + [truncated_piece])
            return " ".join(prompt_parts)
        prompt_parts.append(piece)

    current = " ".join(prompt_parts)
    for _ in range(_MAX_ITERATIONS):
        if _encode_count(tokenizer, current) >= target_tokens:
            return _truncate_to_tokens(current, target_tokens, tokenizer)
        current = current + " " + " ".join(pieces)

    return _truncate_to_tokens(current, target_tokens, tokenizer)


def resolve_workload(
    spec: WorkloadSpec,
    tokenizer: Tokenizer | None,
    *,
    manual_prompt: str | None = None,
) -> RequestSpec:
    if spec.output_tokens_target <= 0:
        raise ValueError(
            f"output_tokens_target must be positive, got {spec.output_tokens_target}"
        )

    if spec.prompt_source == PromptSource.MANUAL.value:
        if manual_prompt is None:
            raise ValueError("manual prompt_source requires a manual_prompt argument")

        prompt = manual_prompt
        prompt_sha256 = fingerprint_prompt(prompt)
        prompt_chars = len(prompt)

        if tokenizer is not None:
            actual = _encode_count(tokenizer, prompt)
            if spec.input_tokens_target > 0 and actual == spec.input_tokens_target:
                status = ResolutionStatus.EXACT
            else:
                status = ResolutionStatus.NEAREST
            tok_prov = tokenizer.provenance()
        else:
            actual = None
            status = ResolutionStatus.UNRESOLVABLE
            tok_prov = None

        return RequestSpec(
            prompt=prompt,
            prompt_sha256=prompt_sha256,
            prompt_chars=prompt_chars,
            input_tokens_target=spec.input_tokens_target,
            input_tokens_actual=actual,
            max_output_tokens=spec.output_tokens_target,
            resolution_status=status,
            tokenizer_provenance=tok_prov,
            workload_seed=spec.seed,
        )

    if spec.input_tokens_target <= 0:
        raise ValueError(
            f"input_tokens_target must be positive for builtin prompts, "
            f"got {spec.input_tokens_target}"
        )

    if tokenizer is None:
        return RequestSpec(
            prompt="",
            prompt_sha256=fingerprint_prompt(""),
            prompt_chars=0,
            input_tokens_target=spec.input_tokens_target,
            input_tokens_actual=None,
            max_output_tokens=spec.output_tokens_target,
            resolution_status=ResolutionStatus.UNRESOLVABLE,
            tokenizer_provenance=None,
            workload_seed=spec.seed,
        )

    prompt = _deterministic_prompt(spec.input_tokens_target, spec.seed, tokenizer)
    prompt_sha256 = fingerprint_prompt(prompt)
    prompt_chars = len(prompt)
    actual = _encode_count(tokenizer, prompt)

    if actual == spec.input_tokens_target:
        status = ResolutionStatus.EXACT
    else:
        status = ResolutionStatus.NEAREST

    return RequestSpec(
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        prompt_chars=prompt_chars,
        input_tokens_target=spec.input_tokens_target,
        input_tokens_actual=actual,
        max_output_tokens=spec.output_tokens_target,
        resolution_status=status,
        tokenizer_provenance=tokenizer.provenance(),
        workload_seed=spec.seed,
    )