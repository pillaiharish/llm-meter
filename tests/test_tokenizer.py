from __future__ import annotations

from llm_meter.tokenizer import FakeTokenizer, TokenizerProvenance, load_tokenizer


def test_fake_tokenizer_encode_decode_roundtrip() -> None:
    tok = FakeTokenizer()
    text = "hello world"
    ids = tok.encode(text)
    assert len(ids) == len(text)
    decoded = tok.decode(ids)
    assert len(decoded) == len(text)


def test_fake_tokenizer_provenance() -> None:
    tok = FakeTokenizer(tokenizer_id="fake-test", revision="v1")
    prov = tok.provenance()
    assert prov.provider == "fake"
    assert prov.tokenizer_id == "fake-test"
    assert prov.revision == "v1"


def test_fake_tokenizer_revision_null_by_default() -> None:
    tok = FakeTokenizer()
    prov = tok.provenance()
    assert prov.revision is None


def test_fake_tokenizer_add_special_tokens_ignored() -> None:
    tok = FakeTokenizer()
    ids1 = tok.encode("hello", add_special_tokens=False)
    ids2 = tok.encode("hello", add_special_tokens=True)
    assert ids1 == ids2


def test_fake_tokenizer_tokens_per_char() -> None:
    tok = FakeTokenizer(tokens_per_char=0.5)
    ids = tok.encode("hello world")
    assert len(ids) == 5


def test_tokenizer_provenance_dataclass() -> None:
    prov = TokenizerProvenance(
        provider="huggingface",
        tokenizer_id="Qwen/Qwen3-8B",
        revision="abc123",
    )
    assert prov.provider == "huggingface"
    assert prov.tokenizer_id == "Qwen/Qwen3-8B"
    assert prov.revision == "abc123"


def test_tokenizer_provenance_revision_null() -> None:
    prov = TokenizerProvenance(
        provider="huggingface",
        tokenizer_id="Qwen/Qwen3-8B",
    )
    assert prov.revision is None


def test_load_tokenizer_fake() -> None:
    tok = load_tokenizer("fake")
    assert tok is not None
    assert tok.provenance().provider == "fake"


def test_load_tokenizer_fake_prefix() -> None:
    tok = load_tokenizer("fake-custom")
    assert tok is not None
    assert tok.provenance().tokenizer_id == "fake-custom"


def test_load_tokenizer_none() -> None:
    tok = load_tokenizer(None)
    assert tok is None