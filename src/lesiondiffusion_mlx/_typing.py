"""Native array and dependency contracts used by the source implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from lesiondiffusion_mlx._arrays import HostArray as HostArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mlx.core as mx
    from tokenizers import Encoding

    type Array = mx.array
    type Dtype = mx.Dtype

    class TokenizerContract(Protocol):
        def enable_truncation(self, max_length: int) -> None:
            """Enable tokenizer truncation at the requested length."""
            ...

        def enable_padding(self, *, length: int, pad_id: int, pad_token: str) -> None:
            """Configure tokenizer padding and its token ID."""
            ...

        def encode_batch(self, inputs: Sequence[str]) -> list[Encoding]:
            """Encode a batch of input strings."""
            ...
