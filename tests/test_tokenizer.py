from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_meter.tokenizer import (
    FakeTokenizer,
    HuggingFaceTokenizer,
    TokenizerProvenance,
    load_tokenizer,
)


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


def _make_mock_hf_tokenizer() -> MagicMock:
    mock_tok = MagicMock()
    mock_encoding = MagicMock()
    mock_encoding.ids = [1, 2, 3]
    mock_tok.encode.return_value = mock_encoding
    mock_tok.decode.return_value = "decoded text"
    return mock_tok


def test_hf_tokenizer_id_forwarded() -> None:
    with patch("tokenizers.Tokenizer.from_pretrained", return_value=_make_mock_hf_tokenizer()):
        tok = HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B")
        prov = tok.provenance()
        assert prov.provider == "huggingface"
        assert prov.tokenizer_id == "Qwen/Qwen3-8B"


def test_hf_tokenizer_explicit_revision_forwarded() -> None:
    with patch(
        "tokenizers.Tokenizer.from_pretrained",
        return_value=_make_mock_hf_tokenizer(),
    ) as mock_fp:
        HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B", revision="abc123")
        mock_fp.assert_called_once_with("Qwen/Qwen3-8B", revision="abc123")


def test_hf_tokenizer_revision_none_preserved() -> None:
    with patch(
        "tokenizers.Tokenizer.from_pretrained",
        return_value=_make_mock_hf_tokenizer(),
    ) as mock_fp:
        tok = HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B", revision=None)
        mock_fp.assert_called_once_with("Qwen/Qwen3-8B", revision=None)
        assert tok.provenance().revision is None


def test_hf_tokenizer_encode_forwards_add_special_tokens_false() -> None:
    mock_tok = _make_mock_hf_tokenizer()
    with patch("tokenizers.Tokenizer.from_pretrained", return_value=mock_tok):
        tok = HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B")
        ids = tok.encode("hello world", add_special_tokens=False)
        mock_tok.encode.assert_called_once_with("hello world", add_special_tokens=False)
        assert ids == [1, 2, 3]
        assert isinstance(ids, list)


def test_hf_tokenizer_encode_returns_list_of_int() -> None:
    mock_tok = _make_mock_hf_tokenizer()
    mock_tok.encode.return_value.ids = [10, 20, 30]
    with patch("tokenizers.Tokenizer.from_pretrained", return_value=mock_tok):
        tok = HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B")
        ids = tok.encode("test")
        assert ids == [10, 20, 30]
        assert all(isinstance(i, int) for i in ids)


def test_hf_tokenizer_decode_delegates() -> None:
    mock_tok = _make_mock_hf_tokenizer()
    mock_tok.decode.return_value = "hello decoded"
    with patch("tokenizers.Tokenizer.from_pretrained", return_value=mock_tok):
        tok = HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B")
        result = tok.decode([1, 2, 3])
        mock_tok.decode.assert_called_once_with([1, 2, 3])
        assert result == "hello decoded"


def test_hf_tokenizer_provenance_fields() -> None:
    with patch("tokenizers.Tokenizer.from_pretrained", return_value=_make_mock_hf_tokenizer()):
        tok = HuggingFaceTokenizer(tokenizer_id="Qwen/Qwen3-8B", revision="main")
        prov = tok.provenance()
        assert prov.provider == "huggingface"
        assert prov.tokenizer_id == "Qwen/Qwen3-8B"
        assert prov.revision == "main"


def test_load_tokenizer_huggingface_with_mock() -> None:
    with patch("tokenizers.Tokenizer.from_pretrained", return_value=_make_mock_hf_tokenizer()):
        tok = load_tokenizer("Qwen/Qwen3-8B")
        assert tok is not None
        assert isinstance(tok, HuggingFaceTokenizer)
        assert tok.provenance().provider == "huggingface"