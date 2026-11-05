"""Native array and dependency contracts used by the source implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from lesiondiffusion_mlx._arrays import HostArray as HostArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mlx.core as mx
    from tokenizers import Encoding

    type Array = mx.array
    type Dtype = mx.Dtype
    type ArrayValue = bool | int | float | complex | list[ArrayValue] | tuple[ArrayValue, ...]

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

    class MetalKernel(Protocol):
        def __call__(
            self,
            *,
            inputs: Sequence[Array],
            template: Sequence[tuple[str, int | bool | Dtype]],
            grid: tuple[int, int, int],
            threadgroup: tuple[int, int, int],
            output_shapes: Sequence[Sequence[int]],
            output_dtypes: Sequence[Dtype],
        ) -> list[Array]:
            """Launch a Metal kernel and return its output arrays."""
            ...

    Mlx = Any


# Defined at runtime because callers pass it to ``typing.cast`` unquoted.
class ArrayConstructor(Protocol):
    def __call__(
        self,
        value: HostArray | Array | ArrayValue,
        /,
        dtype: Dtype | None = None,
    ) -> Array:
        """Construct a native array from values and an optional dtype."""
        ...
