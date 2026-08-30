from __future__ import annotations

import hashlib

import pytest

from llm_meter.tokenizer import FakeTokenizer
from llm_meter.workload import (
    PromptSource,
    ResolutionStatus,
    WorkloadSpec,
    fingerprint_prompt,
    resolve_workload,
)


def _make_tokenizer() -> FakeTokenizer:
    return FakeTokenizer(tokenizer_id="fake-test")


def _make_spec(
    input_tokens: int | None = 100,
    output_tokens: int | None = 64,
    seed: int = 42,
    prompt_source: str = PromptSource.BUILTIN.value,
    tokenizer_id: str | None = "fake-test",
) -> WorkloadSpec:
    return WorkloadSpec(
        input_tokens_target=input_tokens,
        output_tokens_target=output_tokens,
        seed=seed,
        prompt_source=prompt_source,
        tokenizer_id=tokenizer_id,
    )


def test_same_spec_same_tokenizer_same_seed_identical_prompt() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(seed=42)
    req1 = resolve_workload(spec, tok)
    req2 = resolve_workload(spec, tok)
    assert req1.prompt == req2.prompt


def test_same_workload_identical_sha256() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(seed=42)
    req1 = resolve_workload(spec, tok)
    req2 = resolve_workload(spec, tok)
    assert req1.prompt_sha256 == req2.prompt_sha256
    assert req1.prompt_sha256 == hashlib.sha256(req1.prompt.encode("utf-8")).hexdigest()


def test_actual_token_count_measured_not_copied() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=100, seed=42)
    req = resolve_workload(spec, tok)
    assert req.input_tokens_actual_local == len(
        tok.encode(req.prompt, add_special_tokens=False)
    )
    assert req.input_tokens_actual_local is not None
    assert req.input_tokens_target == 100


def test_exact_resolution_deterministic() -> None:
    tok = _make_tokenizer()
    target = 50
    spec = _make_spec(input_tokens=target, seed=42)
    req = resolve_workload(spec, tok)
    assert req.input_tokens_actual_local == target
    assert req.resolution_status == ResolutionStatus.EXACT


def test_nearest_resolution_deterministic() -> None:
    tok = FakeTokenizer(tokenizer_id="fake-odd", tokens_per_char=1.3)
    target = 50
    spec = WorkloadSpec(
        input_tokens_target=target,
        output_tokens_target=64,
        seed=42,
        tokenizer_id="fake-odd",
    )
    req = resolve_workload(spec, tok)
    assert req.input_tokens_actual_local != target
    assert req.resolution_status == ResolutionStatus.NEAREST


def test_no_infinite_adjustment_loop() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=1_000_000, seed=42)
    req = resolve_workload(spec, tok)
    assert req.input_tokens_actual_local is not None
    assert req.input_tokens_actual_local < 1_000_000
    assert req.resolution_status in (ResolutionStatus.EXACT, ResolutionStatus.NEAREST)


def test_builtin_zero_input_tokens_rejected() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=0, seed=42)
    with pytest.raises(ValueError):
        resolve_workload(spec, tok)


def test_builtin_negative_input_tokens_rejected() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=-10, seed=42)
    with pytest.raises(ValueError):
        resolve_workload(spec, tok)


def test_builtin_zero_output_tokens_rejected() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=50, output_tokens=0, seed=42)
    with pytest.raises(ValueError):
        resolve_workload(spec, tok)


def test_builtin_negative_output_tokens_rejected() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=50, output_tokens=-5, seed=42)
    with pytest.raises(ValueError):
        resolve_workload(spec, tok)


def test_builtin_none_input_tokens_rejected() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=None, seed=42)
    with pytest.raises(ValueError):
        resolve_workload(spec, tok)


def test_builtin_none_output_tokens_rejected() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=50, output_tokens=None, seed=42)
    with pytest.raises(ValueError):
        resolve_workload(spec, tok)


def test_output_target_is_target_not_actual() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=50, output_tokens=128, seed=42)
    req = resolve_workload(spec, tok)
    assert req.max_output_tokens == 128
    assert req.input_tokens_actual_local != req.max_output_tokens


def test_generated_prompt_without_tokenizer_raises() -> None:
    spec = _make_spec(tokenizer_id=None)
    with pytest.raises(ValueError, match="builtin workload requires a tokenizer"):
        resolve_workload(spec, None)


def test_manual_prompt_deterministic_fingerprint() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(
        input_tokens=None,
        prompt_source=PromptSource.MANUAL.value,
    )
    req1 = resolve_workload(spec, tok, manual_prompt="Hello world")
    req2 = resolve_workload(spec, tok, manual_prompt="Hello world")
    assert req1.prompt_sha256 == req2.prompt_sha256
    assert req1.prompt_sha256 == hashlib.sha256(b"Hello world").hexdigest()


def test_manual_prompt_with_tokenizer_measures_and_not_applicable() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(
        input_tokens=None,
        prompt_source=PromptSource.MANUAL.value,
        tokenizer_id="fake-test",
    )
    req = resolve_workload(spec, tok, manual_prompt="Hello")
    assert req.input_tokens_target is None
    assert req.input_tokens_actual_local is not None
    assert req.input_tokens_actual_local == 5
    assert req.resolution_status == ResolutionStatus.NOT_APPLICABLE
    assert req.tokenizer_provenance is not None
    assert req.tokenizer_provenance.provider == "fake"


def test_manual_prompt_without_tokenizer_not_applicable() -> None:
    spec = _make_spec(
        input_tokens=None,
        prompt_source=PromptSource.MANUAL.value,
        tokenizer_id=None,
    )
    req = resolve_workload(spec, None, manual_prompt="Hello")
    assert req.input_tokens_target is None
    assert req.input_tokens_actual_local is None
    assert req.resolution_status == ResolutionStatus.NOT_APPLICABLE
    assert req.prompt_sha256 == hashlib.sha256(b"Hello").hexdigest()
    assert req.prompt_chars == 5
    assert req.tokenizer_provenance is None


def test_manual_prompt_without_output_target_does_not_fabricate() -> None:
    tok = _make_tokenizer()
    spec = WorkloadSpec(
        input_tokens_target=None,
        output_tokens_target=None,
        seed=0,
        prompt_source=PromptSource.MANUAL.value,
        tokenizer_id="fake-test",
    )
    req = resolve_workload(spec, tok, manual_prompt="Hello")
    assert req.max_output_tokens is None
    assert req.input_tokens_target is None
    assert req.input_tokens_actual_local is not None
    assert req.resolution_status == ResolutionStatus.NOT_APPLICABLE


def test_manual_prompt_with_explicit_input_target_rejected() -> None:
    tok = _make_tokenizer()
    spec = WorkloadSpec(
        input_tokens_target=50,
        output_tokens_target=None,
        seed=0,
        prompt_source=PromptSource.MANUAL.value,
        tokenizer_id="fake-test",
    )
    with pytest.raises(ValueError, match="input_tokens_target must be None"):
        resolve_workload(spec, tok, manual_prompt="Hello")


def test_different_seed_deterministic() -> None:
    tok = _make_tokenizer()
    req1 = resolve_workload(_make_spec(seed=1), tok)
    req2 = resolve_workload(_make_spec(seed=2), tok)
    assert req1.prompt == resolve_workload(_make_spec(seed=1), tok).prompt
    assert req2.prompt == resolve_workload(_make_spec(seed=2), tok).prompt


def test_fingerprint_prompt_function() -> None:
    fp = fingerprint_prompt("test prompt")
    assert fp == hashlib.sha256(b"test prompt").hexdigest()
    assert len(fp) == 64


def test_tokenizer_provenance_in_request_spec() -> None:
    tok = _make_tokenizer()
    spec = _make_spec(input_tokens=50, seed=42)
    req = resolve_workload(spec, tok)
    assert req.tokenizer_provenance is not None
    assert req.tokenizer_provenance.provider == "fake"
    assert req.tokenizer_provenance.tokenizer_id == "fake-test"
    assert req.tokenizer_provenance.revision is None


def test_tokenizer_revision_none_stays_none() -> None:
    tok = FakeTokenizer(tokenizer_id="fake-rev-test", revision=None)
    spec = WorkloadSpec(
        input_tokens_target=50,
        output_tokens_target=64,
        seed=42,
        tokenizer_id="fake-rev-test",
    )
    req = resolve_workload(spec, tok)
    assert req.tokenizer_provenance is not None
    assert req.tokenizer_provenance.revision is None


def test_final_reencode_determines_exact_or_nearest() -> None:
    tok = _make_tokenizer()
    target = 50
    spec = _make_spec(input_tokens=target, seed=42)
    req = resolve_workload(spec, tok)
    final_count = len(tok.encode(req.prompt, add_special_tokens=False))
    if final_count == target:
        assert req.resolution_status == ResolutionStatus.EXACT
    else:
        assert req.resolution_status == ResolutionStatus.NEAREST