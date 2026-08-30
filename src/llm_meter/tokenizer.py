from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TokenizerProvenance:
    provider: str
    tokenizer_id: str
    revision: str | None = None


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

    def provenance(self) -> TokenizerProvenance:
        ...


class FakeTokenizer:
    def __init__(
        self,
        tokenizer_id: str = "fake-tokenizer",
        tokens_per_char: float = 1.0,
        revision: str | None = None,
    ) -> None:
        self._id = tokenizer_id
        self._tokens_per_char = tokens_per_char
        self._revision = revision

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self._tokens_per_char == 1.0:
            return list(range(len(text)))
        count = max(1, int(len(text) * self._tokens_per_char))
        return list(range(count))

    def decode(self, tokens: list[int]) -> str:
        if self._tokens_per_char == 1.0:
            return "".join(chr(t % 128) for t in tokens)
        count = max(1, int(len(tokens) / self._tokens_per_char))
        return "".join(chr(t % 128) for t in tokens[:count])

    def provenance(self) -> TokenizerProvenance:
        return TokenizerProvenance(
            provider="fake",
            tokenizer_id=self._id,
            revision=self._revision,
        )


class HuggingFaceTokenizer:
    def __init__(self, tokenizer_id: str, revision: str | None = None) -> None:
        from tokenizers import Tokenizer as HFTokenizer

        self._tokenizer_id = tokenizer_id
        self._revision = revision
        self._tokenizer = HFTokenizer.from_pretrained(tokenizer_id, revision=revision)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def provenance(self) -> TokenizerProvenance:
        return TokenizerProvenance(
            provider="huggingface",
            tokenizer_id=self._tokenizer_id,
            revision=self._revision,
        )


def load_tokenizer(tokenizer_id: str | None) -> Tokenizer | None:
    if tokenizer_id is None:
        return None
    if tokenizer_id == "fake" or tokenizer_id.startswith("fake-"):
        return FakeTokenizer(tokenizer_id=tokenizer_id)
    return HuggingFaceTokenizer(tokenizer_id=tokenizer_id)